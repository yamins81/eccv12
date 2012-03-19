import sys
import numpy as np
import Image


class ImgLoaderResizer(object):
    """ Load 250x250 greyscale images, return normalized 200x200 float32 ones.
    """
    def __init__(self, shape=None, ndim=None, dtype='float32', normalize=True,
                 crop=None):
        assert len(shape) == 2
        shape = tuple(shape)
        assert len(crop) == 4
        crop = tuple(crop)
        l, t, r, b = crop
        assert 0 <= l < r <= 250
        assert 0 <= t < b <= 250 
        self._crop = crop   
        assert dtype == 'float32'
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.normalize = normalize

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        assert im.size == (250, 250)
        if self._crop != (0, 0, 250, 250):
            im = im.crop(self._crop)
        l, t, r, b = self._crop
        assert im.size == (r - l, b - t)
        if max(im.size) != self._shape[0]:
            m = self._shape[0]/float(max(im.size))
            new_shape = (int(round(im.size[0]*m)), int(round(im.size[1]*m)))
            im = im.resize(new_shape, Image.ANTIALIAS)
        imval = np.asarray(im, 'float32')
        rval = np.zeros(self._shape)
        ctr = self._shape[0]/2
        cxmin = ctr - imval.shape[0] / 2
        cxmax = ctr - imval.shape[0] / 2 + imval.shape[0]
        cymin = ctr - imval.shape[1] / 2
        cymax = ctr - imval.shape[1] / 2 + imval.shape[1]
        rval[cxmin:cxmax,cymin:cymax] = imval
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
        assert rval.shape == self._shape
        return rval


DOT_MAX_NDIMS = 10000
MEAN_MAX_NPOINTS = 2000
STD_MAX_NPOINTS = 2000


import theano
sX = theano.tensor.matrix(dtype='float32')
sY = theano.tensor.matrix(dtype='float32')
dot32 = theano.function([sX, sY], theano.tensor.dot(sX, sY))
sX = theano.tensor.matrix(dtype='float64')
sY = theano.tensor.matrix(dtype='float64')
dot64 = theano.function([sX, sY], theano.tensor.dot(sX, sY))


def dot(A, B):
    _dot = dict(float32=dot32, float64=dot64)[str(A.dtype)]
    return _dot(A, B)


def chunked_linear_kernel(Xs, Ys, use_theano, symmetric):
    """Compute a linear kernel in blocks so that it can use a GPU with limited
    memory.

    Xs is a list of feature matrices
    Ys ia  list of feature matrices

    This function computes the kernel matrix with
        \sum_i len(Xs[i]) rows
        \sum_j len(Ys[j]) cols
    """

    dtype = Xs[0].dtype

    def _dot(A, B):
        if K < DOT_MAX_NDIMS:
            return dot(A, B)
        else:
            out = dot(A[:,:DOT_MAX_NDIMS], B[:DOT_MAX_NDIMS])
            ndims_done = DOT_MAX_NDIMS            
            while ndims_done < K:
                out += dot(
                    A[:,ndims_done : ndims_done + DOT_MAX_NDIMS], 
                    B[ndims_done : ndims_done + DOT_MAX_NDIMS])
                ndims_done += DOT_MAX_NDIMS
            return out

    R = sum([len(X) for X in Xs])
    C = sum([len(Y) for Y in Ys])
    K = Xs[0].shape[1]

    rval = np.zeros((R, C), dtype=dtype)

    if symmetric:
        assert R == C

    print 'Computing gram matrix',

    ii0 = 0
    for ii, X_i in enumerate(Xs):
        sys.stdout.write('.')
        sys.stdout.flush()
        ii1 = ii0 + len(X_i) # -- upper bound of X block

        jj0 = 0
        for jj, Y_j in enumerate(Ys):
            jj1 = jj0 + len(Y_j) # -- upper bound of Y block

            r_ij = rval[ii0:ii1, jj0:jj1]

            if symmetric and jj < ii:
                r_ji = rval[jj0:jj1, ii0:ii1]
                r_ij[:] = r_ji.T
            else:
                r_ij[:] = _dot(X_i, Y_j.T)

            jj0 = jj1

        ii0 = ii1

    print 'done!'
    return rval


def linear_kernel(X, Y, use_theano, block_size=1000):
    """Compute a linear kernel in blocks so that it can use a GPU with limited
    memory.

    Xs is a list of feature matrices
    Ys ia  list of feature matrices

    This function computes the kernel matrix with
        \sum_i len(Xs[i]) rows
        \sum_j len(Ys[j]) cols
    """

    def chunk(Z):
        Zs = []
        ii = 0
        while len(Z[ii:ii + block_size]):
            Zs.append(Z[ii:ii + block_size])
            ii += block_size
        return Zs

    Xs = chunk(X)
    Ys = chunk(Y)

    assert sum([len(xi) for xi in Xs]) == len(X)
    assert sum([len(yi) for yi in Ys]) == len(Y)
    return chunked_linear_kernel(Xs, Ys, use_theano, symmetric=(X is Y))


def mean_and_std(X):
    fshape = X.shape
    X.shape = fshape[0], -1
    npoints, ndims = X.shape

    if npoints < MEAN_MAX_NPOINTS:
        fmean = X.mean(0)
    else:
        sel = X[:MEAN_MAX_NPOINTS]
        fmean = np.empty_like(sel[0,:])

        np.add.reduce(sel, axis=0, dtype="float32", out=fmean)

        # -- sum up the features in blocks to reduce rounding error
        curr = np.empty_like(fmean)
        npoints_done = MEAN_MAX_NPOINTS
        while npoints_done < npoints:
            
            sel = X[npoints_done : npoints_done + MEAN_MAX_NPOINTS]
            np.add.reduce(sel, axis=0, dtype="float32", out=curr)
            np.add(fmean, curr, fmean)
            npoints_done += MEAN_MAX_NPOINTS                
 
        fmean /= npoints

    if npoints < STD_MAX_NPOINTS:
        fstd = X.std(0)
    else:
        sel = X[:MEAN_MAX_NPOINTS]

        mem = np.empty_like(sel)
        curr = np.empty_like(mem[0,:])

        seln = sel.shape[0]
        np.subtract(sel, fmean, mem[:seln])
        np.multiply(mem[:seln], mem[:seln], mem[:seln])
        fstd = np.add.reduce(mem[:seln], axis=0, dtype="float32")

        npoints_done = MEAN_MAX_NPOINTS
        # -- loop over by blocks for improved numerical accuracy
        while npoints_done < npoints:

            sel = X[npoints_done : npoints_done + MEAN_MAX_NPOINTS]
            seln = sel.shape[0]
            np.subtract(sel, fmean, mem[:seln])
            np.multiply(mem[:seln], mem[:seln], mem[:seln])
            np.add.reduce(mem[:seln], axis=0, dtype="float32", out=curr)
            np.add(fstd, curr, fstd)

            npoints_done += MEAN_MAX_NPOINTS

        fstd = np.sqrt(fstd/npoints)

    fstd[fstd == 0] = 1
    return fmean, fstd


