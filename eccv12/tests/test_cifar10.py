from eccv12.cifar10 import *

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


def test_partial_callpipe():
    n_train = 50
    n_valid = 50
    batchsize = 10
    HP = hyperopt_param
    lnorm0 = partial(lnorm,
            ker_size=HP('n0_size', rfilter_size(2, 6)),
            remove_mean=HP('n0_remove_mean', one_of(0, 1)),
            stretch=HP('n0_stretch', logu_range(.1/3, 10.*3)),
            threshold=HP('n0_thresh', logu_range(.1/3, 10.*3)))

    fbcorr1 = partial(fbcorr,
            ker_size=HP('f1_size', rfilter_size(2, 6)),
            n_filters=16,
            generate=('random:uniform',
                {'rseed': HP('f1_seed', choice(range(10, 15)))}))

    screen_features = pyll_theano_batched_lmap(
            partial('callpipe1', [lnorm0, fbcorr1]),
            np.random.randn(20, 3, 32, 32),
            batchsize=batchsize)

    feats = pyll.stochastic.sample(screen_features, np.random.RandomState(1))

    assert feats.shape[0] == 20
    assert feats.shape[1] == 16
    assert feats.shape[2] < 32
    assert feats.shape[3] < 32


def test_bandit1_sample():
    bandit = Cifar10Bandit1('foo', 10, 10, 10)
    print 'TEMPLATE', bandit.template

    config = pyll.stochastic.sample(bandit.template,
            np.random.RandomState(34))

    print 'CONFIG', config

    bandit.evaluate(config, ctrl=None)


