import time
import numpy as np

import theano

from skdata import larray
from skdata.cifar10 import CIFAR10

# install some functions from pyll.scope into the module symbol table
import pyll

############
############ CODE THAT SHOULD BE UPSTREAM: HYPEROPT?
############

#@pyll.scope.define
def pyll_theano_batched_lmap((in_rows, out_rows), seq, batchsize,
        _debug_call_counts=None, print_progress=False):

    in_shp = (batchsize,) + seq.shape[1:]
    batch = np.zeros(in_shp, dtype='float32')
    s_ibatch = theano.shared(batch)
    s_xi = (s_ibatch * 1).type() # get a TensorType w right ndim
    s_N = s_xi.shape[0]
    s_X = theano.tensor.set_subtensor(s_ibatch[:s_N], s_xi)
    memo = {in_rows: (s_X, in_shp)}
    s_obatch, oshp = pyll.rec_eval(out_rows, memo=memo)
    assert oshp[0] == batchsize

    # Compile a function that takes a variable number of elements in,
    # returns the same number of processed elements out,
    # but does all internal computations using a fixed number of elements,
    # because convolutions are fastest when they're hard-coded to a certain
    # size.
    fn = theano.function([s_xi], s_obatch[:s_N],
            updates={
                s_ibatch: s_X, # this allows the inc_subtensor to be in-place
                })

    def fn_1(x):
        if _debug_call_counts:
            _debug_call_counts['fn_1'] += 1
        return fn(x[None, :, :, :])[0]

    attrs = {
            'shape': oshp[1:],
            'ndim': len(oshp) -1,
            'dtype': s_obatch.dtype }
    def rval_getattr(attr, objs):
        # -- objs don't matter to the structure of the return value
        try:
            return attrs[attr]
        except KeyError:
            raise AttributeError(attr)

    fn_1.rval_getattr = rval_getattr

    def f_map(X):
        if _debug_call_counts:
            _debug_call_counts['f_map'] += 1
        rval = np.zeros((len(X),) + oshp[1:], dtype=s_obatch.dtype)
        offset = 0
        while offset < len(X):
            if print_progress:
                print 'pyll_theano_batched_lmap.f_map', offset, len(X)
            xi = X[offset: offset + batchsize]
            rval[offset:offset + len(xi)] = fn(xi)
            offset += len(xi)
        return rval

    return larray.lmap(fn_1, seq, f_map=f_map)


@pyll.scope.define
def linsvm_train_valid_test(features, labels, n_train, n_test,
        C=1.0, allow_inplace=False, normalize_cols=True, shuffle=False):
    return {'result': 'awesome'}


############
############
############

def restrict_scope(scope, *args, **kwargs):
    rval = {}
    for k in args:
        rval[k] = getattr(scope, k)
    for k, origk in kwargs.items():
        rval[k] = getattr(scope, origk)
    return rval


import pyll_slm
globals().update(restrict_scope(pyll.scope,
            #'pyll_theano_batched_lmap',
            'linsvm_train_valid_test',
            'uniform',
            'quniform',
            'loguniform',
            'qloguniform',
            'one_of',
            'choice',
            **{
                'lnorm': 'slm_lnorm',
                'lpool': 'slm_lpool',
                'fbcorr': 'slm_fbcorr',
                's_int': 'int',
                }))


def rfilter_size(smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(quniform(smin - q + 1e-5, smax, q))


def logu_range(lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return loguniform(np.log(lower), np.log(upper))


def experiment1(namebase):
    img_i = pyll.Literal()

    layer0 = lnorm(img_i,
            ker_size = rfilter_size(2, 6),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    layer1 = lnorm(
            lpool(
                fbcorr(layer0,
                    ker_size=rfilter_size(2, 6),
                    n_filters=s_int(
                        qloguniform(
                            np.log(16 / 2.0),
                            np.log(64),
                            q=16)),
                    generate=('random:uniform',
                        {'rseed': choice(range(10, 15))})),
                ker_size=rfilter_size(2, 5),
                order=loguniform(np.log(1), np.log(10)),
                stride=1),
            ker_size = rfilter_size(2, 5),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    layer2 = lnorm(
            lpool(
                fbcorr(layer1,
                    ker_size=rfilter_size(2, 6),
                    n_filters=s_int(
                        qloguniform(
                            np.log(16 / 2.0),
                            np.log(128),
                            q=16)),
                    generate=('random:uniform',
                        {'rseed': choice(range(20, 25))})),
                ker_size=rfilter_size(2, 5),
                order=loguniform(np.log(1), np.log(10)),
                stride=1),
            ker_size = rfilter_size(2, 5),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    layer3 = lnorm(
            lpool(
                fbcorr(layer2,
                    ker_size=rfilter_size(2, 6),
                    n_filters=s_int(
                        qloguniform(
                            np.log(16 / 2.0),
                            np.log(384),
                            q=16)),
                    generate=('random:uniform',
                        {'rseed': choice(range(30, 35))})),
                ker_size=rfilter_size(2, 5),
                order=loguniform(np.log(1), np.log(10)),
                stride=1),
            ker_size = rfilter_size(2, 5),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    return locals()

    result = linsvm_train_valid_test(
            image_features, labels,
            50000, 10000,
            C=1.0,
            allow_inplace=True,
            shuffle=False)

    return locals()


################################################################################
#                                  TESTS
################################################################################

import unittest

class TestPTBL(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(234)
        dummy = pyll.Literal()
        layer0 = lnorm(dummy,
                ker_size = rfilter_size(2, 10),
                remove_mean=one_of(0, 1),
                stretch=logu_range(.1/3, 10.*3),
                threshold=logu_range(.1/3, 10.*3))
        pyll.stochastic.recursive_set_rng_kwarg(layer0, rng)
        self.call_counts = {'fn_1': 0, 'f_map': 0}
        self.features = pyll_theano_batched_lmap((dummy, layer0),
                np.random.rand(101, 3, 32, 32).astype('float32'),
                batchsize=10,
                _debug_call_counts=self.call_counts)

    def test0(self):
        ff = self.features[:]
        print self.features.shape
        print ff.shape
        assert self.features.shape == ff.shape


    def test1(self):
        assert self.call_counts == {'fn_1': 0, 'f_map': 0}
        self.features[0]
        assert self.call_counts == {'fn_1': 1, 'f_map': 0}
        self.features[:]
        assert self.call_counts == {'fn_1': 1, 'f_map': 1}
        self.features[:1]
        assert self.call_counts == {'fn_1': 1, 'f_map': 2}
        self.features[:11]
        assert self.call_counts == {'fn_1': 1, 'f_map': 3}

    def test2(self):
        ff1 = self.features[:]
        ff2 = self.features[:]

        # sanity check
        assert ff1.shape == ff2.shape

        assert np.all(ff1 == ff2)
        ff2[:] = 0
        # assert no aliasing
        assert np.all(ff1 != ff2)


def test_cifar10():
    allvars = experiment1('memmap_test_cifar10')
    rng = np.random.RandomState(234)
    img_i = allvars['img_i']
    layer3 = allvars['layer3']
    imgs, labels = CIFAR10().img_classification_task(dtype='float32')
    # put imgs channel-major for Theano conv
    print imgs.dtype
    imgs = imgs.transpose(0, 3, 1, 2)
    pyll.stochastic.recursive_set_rng_kwarg(layer3, rng)
    i = 0
    while True:
        try:
            image_features = pyll_theano_batched_lmap((img_i, layer3),
                    imgs, batchsize=1000)
            break
        except pyll_slm.InvalidDescription:
            print 'Invalid', i
            i += 1
            continue

    print 'Extracting features'
    print image_features.shape
    t0 = time.time()
    image_features[:]
    t1 = time.time()
    print 'took', t1 - t0

