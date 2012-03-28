import os
import time

import numpy as np
import scipy.io

from skdata.utils.glviewer import glumpy_viewer, command, glumpy

#import pyll
from .cifar10 import CF10

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from pyll_slm import boxconv

from .utils import mean_and_std


#@pyll.scope.define
def random_patches(images, N, R, C):
    """Return a stack of N image patches"""
    n_imgs, iR, iC, iF = images.shape
    rval = np.empty((N, R, C, iF), dtype=images.dtype)
    rng = np.random.RandomState(1234)
    for rv_i in rval:
        src = rng.randint(n_imgs)
        roffset = rng.randint(iR - R)
        coffset = rng.randint(iC - C)
        rv_i[:] = images[src, roffset: roffset + R, coffset: coffset + C]
    return rval


def contrast_normalize(patches):
    X = patches
    N = X.shape[1]
    unbias = float(N) / (float(N) - 1)
    X = (X - X.mean(1)[:, None]) / np.sqrt(unbias * X.var(1)[:, None] + 10)
    return X


#@pyll.scope.define_info(o_len=2)
def patch_whitening_filterbank(patches, retX=False, reshape=True):
    """
    Image patches of uint8 pixels

    """
    # Algorithm from Coates' sc_vq_demo.m
    assert str(patches.dtype) == 'uint8'

    # -- patches -> column vectors
    X = patches.reshape(len(patches), -1).astype('float64')

    X = contrast_normalize(X)

    # -- ZCA whitening (with low-pass)
    M, _std = mean_and_std(X)
    #M = X.mean(0)  -- less numerically accurate?
    Xm = X - M
    assert Xm.shape == X.shape
    C = np.dot(Xm.T, Xm) / (Xm.shape[0] - 1)
    D, V = np.linalg.eigh(C)
    P = np.dot(np.sqrt(1.0 / (D + 0.1)) * V, V.T)

    # -- return to image space
    if reshape:
        M = M.reshape(patches.shape[1:])
        P = P.reshape((P.shape[0],) + patches.shape[1:])

    if retX:
        return M, P, X
    else:
        return M, P


#@pyll.scope.define_info(o_len=2)
def cifar10_img_classification_task(dtype='float32'):
    imgs, labels = CF10.img_classification_task(dtype='float32')
    return imgs, labels


def im2col(img, (R, C)):
    H, W, F = img.shape
    rval = np.zeros(((H - R + 1), (W - C + 1), R, C, F), dtype=img.dtype)
    for ii in xrange(rval.shape[0]):
        for jj in xrange(rval.shape[1]):
            rval[ii, jj] = img[ii:ii + R, jj: jj+ C]
    return rval


def show_centroids(D):
    D = D.copy()
    for di in D:
        di -= di.min()
        di /= di.max()
    glumpy_viewer(
            img_array=D.astype('float32'),
            arrays_to_print=[],
            )


import line_profiler
profile = line_profiler.LineProfiler()
import time


#@profile
def extract_features(imgs, D, M, P, alpha, R, C,
        internal_dtype='float64'):
    tt = time.time()
    N, H, W, F = imgs.shape
    numBases = len(D)
    M = M.flatten().astype(internal_dtype)
    P = P.reshape((R * C * 3, R * C * 3)).astype(internal_dtype)
    XC = np.zeros((N, len(D), 2, 2, 2), dtype='float32')
    PD = np.dot(P, D.reshape(len(D), -1).T).astype(internal_dtype)
    for i in xrange(len(imgs)):
        print 'PY ITER', i
        tt = time.time()
        if 0 == i % 100:
            #profile.print_stats()
            print i, (time.time() - tt)
        patches = im2col(imgs[i], (R, C)).astype(internal_dtype)
        patches = patches.transpose(0, 1, 4, 2, 3).reshape(
                (patches.shape[0] * patches.shape[1], -1))

        patches_cn = contrast_normalize(patches)
        z = np.dot(patches_cn - M, PD)

        if 0:
            aPDx = p_scale[:, None] * np.dot(patches, PD)
            aPDmmm = (p_scale * p_mean)[:, None] * PD.sum(0)


            tmp_1 = np.dot((patches - p_mean[:, None]) * p_scale[:, None], PD)
            tmp_2 = aPDx - aPDmmm
            #print 'tmp1', tmp_1
            #print 'tmp2', tmp_2
            #assert np.allclose(tmp_1, tmp_2)

            PDM = np.dot(M, PD)

            Z = aPDx - aPDmmm - PDM
            for foo in [
                    #aPDx,
                    #-aPDmmm,
                    #-PDM,
                    #aPDx - aPDmmm,
                    #-aPDmmm - PDM,
                    z,
                    #Z,
                    ]:
                print '->', foo.min(), foo.max(), foo.mean(), foo.shape
                #print foo.flatten()[:0]
                #print foo
                pass
            #print z
            #print Z
            print 'PY ATOL', abs(z - Z).max()
            print 'PY RTOL', (abs(z - Z) / ((1e-12 + abs(z) + abs(Z)))).max()

        prows = H - R + 1
        pcols = W - C + 1
        z = z.reshape((prows, pcols, numBases))
        hr = int(np.round(prows / 2.))
        hc = int(np.round(pcols / 2.))
        XC[i, :, 0, 0, 0] = np.maximum(z[:hr, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 0, 0, 1] = -np.maximum(-z[:hr, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 0, 1, 0] = np.maximum(z[:hr, hc:] - alpha, 0).sum(1).sum(0)
        XC[i, :, 0, 1, 1] = -np.maximum(-z[:hr, hc:] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 0, 0] = np.maximum(z[hr:, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 0, 1] = -np.maximum(-z[hr:, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 1, 0] = np.maximum(z[hr:, hc:] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 1, 1] = -np.maximum(-z[hr:, hc:] - alpha, 0).sum(1).sum(0)
    return XC


def extract_features_theano(imgs, D, M, P, alpha, batchsize=10):
    """
    imgs, D, P are 4-tensors in channel-minor layout
    M is the shape of a single patch (6, 6, 3)
    alpha is a scalar
    """

    tt = time.time()
    N, H, W, F = imgs.shape
    R, C = M.shape[:2]
    numBases = len(D)
    M = M.astype('float32')
    P = P.astype('float32')
    D = D.astype('float32')
    XC = np.zeros((N, len(D), 2, 2, 2), dtype='float32')

    s_imgs = theano.shared(imgs[:batchsize].astype('float32'))

    x_shp = (batchsize, F, H, W)
    ker_shape = (R, C)

    # -- calculate patch means, patch variances
    p_sum, _shp = boxconv((s_imgs, x_shp), ker_shape, channels=True)
    p_ssq, _shp = boxconv((s_imgs ** 2, x_shp), ker_shape, channels=True)
    p_mean = p_sum / (R * C * F)
    p_var = p_ssq / (R * C * F - 1) - (p_sum / (R * C * F)) ** 2
    p_scale = 1.0 / tensor.sqrt(p_var + 10)
    assert p_mean.ndim == 4
    assert p_scale.ndim == 4
    assert p_mean.broadcastable[1]
    assert p_scale.broadcastable[1]

    # --
    # from whitening, we have a shift and linear transform (P)
    # for each patch (as vector).
    #
    # let m be the vector [m m m m] that replicates p_mean
    # let a be the scalar p_scale
    # let x be an image patch from s_imgs
    #
    # Whitening means applying the affine transformation
    #   (c - M) P
    # to contrast-normalized patch c = a (x - m),
    # where a = p_scale and m = p_mean.
    #
    # We also want to extract features in dictionary D
    #
    #   (c - M) P D
    #   = (a (x - [m,m,m]) - M) P D
    #   = (a x - a [m,m,m] - M) P D
    #   = a x P D - a [m,m,m] P D - M P D
    #
    M1 = M.flatten()
    P2 = P.reshape((108, 108))
    assert D.shape[1:] == (6, 6, 3)
    D2 = D.reshape(numBases, 108)

    PD2 = np.dot(P2, D2.T)  # -- N.B. P is symmetric
    PD_kerns = PD2.reshape(6, 6, 3, numBases)\
            .transpose(3, 2, 0, 1)[:, :, ::-1, ::-1]
    s_PD_kerns = theano.shared(np.asarray(PD_kerns, order='C'))

    PDx = conv.conv2d(
            s_imgs,
            s_PD_kerns,
            image_shape=x_shp,
            filter_shape=(numBases, 3, 6, 6),
            border_mode='valid')

    s_PD2_sum = theano.shared(PD2.sum(0))
    aPDmmm = p_scale * p_mean * s_PD2_sum.dimshuffle(0, 'x', 'x')

    s_PDM = theano.shared(np.dot(M1, PD2))  # -- vector

    z = p_scale * PDx - aPDmmm - s_PDM.dimshuffle(0, 'x', 'x')
    assert z.ndim == 4

    s_foo = [
            #p_scale * PDx,
            #- aPDmmm,
            #- s_PDM.dimshuffle(0, 'x', 'x'),
            #p_scale * PDx - aPDmmm,
            #aPDmmm - s_PDM.dimshuffle(0, 'x', 'x'),
            #z,
            ]

    sXC_base = sXC = theano.shared(XC[:2])

    hr = int(np.round((H - R + 1) / 2.))
    hc = int(np.round((W - C + 1) / 2.))

    sXC = tensor.set_subtensor(sXC[:, :, 0, 0, 0],
        tensor.maximum(z[:, :, :hr, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 0, 0, 1],
        -tensor.maximum(-z[:, :, :hr, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 0, 1, 0],
        tensor.maximum(z[:, :, :hr, hc:] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 0, 1, 1],
        -tensor.maximum(-z[:, :, :hr, hc:] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 0, 0],
        tensor.maximum(z[:, :, hr:, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 0, 1],
        -tensor.maximum(-z[:, :, hr:, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 1, 0],
        tensor.maximum(z[:, :, hr:, hc:] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 1, 1],
        -tensor.maximum(-z[:, :, hr:, hc:] - alpha, 0).sum([2, 3]))

    z_fn = theano.function([], s_foo,
            updates={sXC_base: sXC})

    i = 0
    sXC_base.set_value(XC[i:i + batchsize])
    while i < len(imgs):
        #print 'THEANO ITER', i
        tt = time.time()
        s_imgs.set_value(
                imgs[i:i + batchsize].transpose(0, 3, 1, 2).astype('float32'))
        vfoo = z_fn()
        for foo in vfoo:
            print ' -> ', foo.min(), foo.max(), foo.mean(), foo.shape
            #print foo
        XC[i:i + batchsize] = sXC_base.get_value()
        i += batchsize
        #print 'TIME', time.time() - tt

    XC = XC.reshape((len(XC), -1))
    return XC


def assert_allclose(a, b, rtol=1e-05, atol=1e-08):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        adiff = abs(a - b).max(),
        rdiff = (abs(a - b) / (abs(a) + abs(b) + 1e-15)).max()
        raise ValueError('not close enough', (adiff, rdiff, {
            'amax': a.max(),
            'bmax': b.max(),
            'amin': a.min(),
            'bmin': b.min(),
            'asum': a.sum(),
            'bsum': b.sum(),
            }))


def track_matlab():
    """
    Mar 28 - this function gets exactly the same features as Adam Coates'
    matlab code.
    """
    def mpath(name):
        return os.path.join(
            '/home/bergstra/.VENV/eccv12/src/sc_vq_demo/',
            name)
    m_patches = scipy.io.loadmat(mpath('patches.mat'))['patches']
    m_patches_rci = scipy.io.loadmat(mpath('patches_rci.mat'))['patches_rci']
    print 'm_patches',
    print m_patches.shape, m_patches.min(), m_patches.max()

    imgs, labels = CF10.img_classification_task(dtype='uint8')
    patches = []
    for j, (r, c, i) in enumerate(m_patches_rci):
        patch = imgs[i - 1, c - 1:c + 5, r - 1:r + 5]
        flatpatch = patch.transpose(2, 0, 1).flatten()
        patches.append(flatpatch)
        assert np.allclose(flatpatch, m_patches[j])
    patches = np.asarray(patches) # -- matches m_patches
    assert np.allclose(patches, m_patches)
    M, P, X = patch_whitening_filterbank(patches, retX=True, reshape=False)

    X = np.dot(X - M, P)

    foo = scipy.io.loadmat(mpath('patches_final.mat'))
    assert np.allclose(M, foo['M'].flatten())
    assert np.allclose(P, foo['P'])
    if 0:
        print X.shape, foo['patches'].shape
        print X[0].min(), foo['patches'][0].min()
        print X[0].max(), foo['patches'][0].max()
        print X[0].sum(), foo['patches'][0].sum()
    assert_allclose(X[0], foo['patches'][0], atol=1e-6, rtol=1e-3)
    assert_allclose(X, foo['patches'], atol=1e-6, rtol=1e-3)

    foo = scipy.io.loadmat(mpath('dictionary.mat'))
    print foo['dictionary_elems'][:, 0]
    dictionary = X[foo['dictionary_elems'].flatten() - 1]
    dictionary = dictionary / (np.sqrt((dictionary ** 2).sum(axis=1))[:, None] + 1e-20);

    if 0:
        show_centroids(
                dictionary.reshape(len(dictionary), 3, 6, 6).transpose(0, 3, 2,
                    1))
    # tolerances are getting a little wacky, but the images look right
    assert_allclose(dictionary, foo['dictionary'], atol=1e-6, rtol=0.03)

    m_trainXC = scipy.io.loadmat(mpath('trainXC5.mat'))['trainXC']
    nF = len(dictionary)
    trainXC = extract_features(imgs[:5], dictionary, M, P, .25, 6, 6)
    trainXCp = trainXC.transpose(0, 2, 3, 4, 1).reshape(5, -1)
    assert_allclose(trainXCp, m_trainXC, atol=1e-4, rtol=1e-4)


def demo():
    imgs, labels = CF10.img_classification_task(dtype='uint8')

    try:
        train_XC = np.load('train_XC.npy')
    except IOError:
        tt = time.time()
        patches = random_patches(imgs[:10000], 50000, 6, 6)
        print 'extracting patches took', time.time() - tt
        M, P = patch_whitening_filterbank(patches)
        print 'ZCA took', time.time() - tt


        print 'P range', P.min(), P.max()
        if 0:
            P = (P - P.min()) / (P.max() - P.min())
            glumpy_viewer(
                    img_array=P,
                    arrays_to_print=[],
                    )

        # -- use 'patches' dictionary-learning algorithm
        D = np.dot(
                patches[:1600].reshape((1600, -1)) - M.reshape(108),
                P.reshape((108, 108)))
        D = D / (np.sqrt(np.sum(D ** 2, axis=1)) + 1e-20)[:, None]
        D = D.reshape(1600, 6, 6, 3)

        print 'D range', D.min(), D.max()
        if 0:
            D = (D - D.min()) / (D.max() - D.min())
            glumpy_viewer(
                    img_array=D,
                    arrays_to_print=[],
                    )

        # keep just 3 filters
        #D = D[:10]
        train_XC = extract_features_theano(imgs[:50000], D, M, P, alpha=.25,
                batchsize=100)
        if 0:
            train_XC_theano = train_XC
            train_XC = extract_features(imgs[:100], D, M, P, alpha=.25)
            print 'XC th', train_XC_theano.min(), train_XC_theano.max()
            print 'XC py', train_XC.min(), train_XC.max()
            z = train_XC
            Z = train_XC_theano
            #print z
            #print Z
            print 'ATOL', abs(z - Z).max()
            print 'RTOL', (abs(z - Z) / ((1e-12 + abs(z) + abs(Z)))).max()
            assert np.allclose(train_XC_theano, train_XC)

        np.save('train_XC.npy', train_XC)

    try:
        test_XC = np.load('test_XC.npy')
    except IOError:
        test_XC = extract_features_theano(imgs[50000:], D, M, P, alpha=.25,
                batchsize=100)
        np.save('test_XC.npy', test_XC)

    from pyll_slm import fit_linear_svm, model_predict, error_rate
    from .utils import linear_kernel, mean_and_std

    xmean, xstd = mean_and_std(train_XC, remove_std0=False)
    xscale = 1.0 / np.sqrt(xstd ** 2 + 0.01)
    train_XC -= xmean
    train_XC *= xscale
    test_XC -= xmean
    test_XC *= xscale
    svm = fit_linear_svm((train_XC, labels[:len(train_XC)]),
        solver=('asgd.SparseUpdateRankASGD',
            {
                'sgd_step_size0': 0.01,
                'dtype': 'float64',
                'fit_n_partial': len(train_XC),
                'fit_verbose': 1,
                'cost_fn': "L2Huber",
                #'cost_fn': "Hinge",
                'max_observations': 1 * 50000,
                }),
        l2_regularization=5e-7)

    pred = model_predict(svm, train_XC)
    print 'TRAIN ERR', error_rate(pred, labels[:50000])

    pred = model_predict(svm, test_XC)
    print 'TEST ERR', error_rate(pred, labels[50000:])


