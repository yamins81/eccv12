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


def linear_kernel(X, Y, use_theano):
    """Compute a linear kernel in blocks so that it can use a GPU with limited
    memory
    """

    if use_theano:
        import theano
        sX = theano.tensor.matrix(dtype=X.dtype)
        sY = theano.tensor.matrix(dtype=X.dtype)
        dot = theano.function([sX, sY], theano.tensor.dot(sX, sY))
    else:
        dot = np.dot

    n_blocks = 10
    if len(X) % n_blocks:
        raise NotImplementedError()
    if len(Y) % n_blocks:
        raise NotImplementedError()
    x_block_size = len(X) / n_blocks
    y_block_size = len(Y) / n_blocks

    rval = np.zeros((len(X), len(Y)), dtype=X.dtype)

    for ii in range(n_blocks):
        x_i = X[ii * x_block_size : (ii + 1) * x_block_size]

        for jj in range(n_blocks):
            r_ij = rval[ii * x_block_size : (ii + 1) * x_block_size,
                    jj * y_block_size : (jj + 1) * y_block_size]

            y_j = Y[jj * y_block_size : (jj + 1) * y_block_size]

            if jj >= ii or (X is not Y):
                r_ij[:] = dot(x_i, y_j.T)
            else:
                # compute a SYRK
                r_ji = rval[jj * y_block_size : (jj + 1) * y_block_size,
                        ii * x_block_size : (ii + 1) * x_block_size]
                r_ij[:] = r_ji.T

    return rval

