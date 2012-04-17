import cPickle
import unittest

import nose.plugins.attrib
from eccv12.cifar10 import *
from eccv12 import pyll_slm

import hyperopt
import hyperopt.plotting
from hyperopt import Random, Experiment, Trials

class TestPTBL(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(234)
        layer0 = partial(slm_lnorm,
                ker_size = rfilter_size('ksize', 2, 10),
                remove_mean=hp_choice('rmean', [0, 1]),
                stretch=logu_range('stretch', .1/3, 10.*3),
                threshold=logu_range('thresh', .1/3, 10.*3))
        pyll.stochastic.recursive_set_rng_kwarg(layer0, rng)
        self.call_counts = {'fn_1': 0, 'f_map': 0}
        self.features = pyll_slm.pyll_theano_batched_lmap(layer0,
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
    lnorm0 = partial(slm_lnorm,
            ker_size=rfilter_size('n0_size', 2, 6),
            remove_mean=hp_choice('n0_remove_mean', [0, 1]),
            stretch=logu_range('n0_stretch', .1/3, 10.*3),
            threshold=logu_range('n0_thresh', .1/3, 10.*3))

    size=rfilter_size('ksize', 2, 6)
    fbcorr1 = partial(slm_fbcorr,
            kerns= alloc_random_uniform_filterbank(
                    n_filters=16,
                    height=size,
                    width=size,
                    channels=3,
                    dtype='float32',
                    rseed=hp_choice('f1_seed', range(10, 15))))

    screen_features = pyll_theano_batched_lmap(
            partial(callpipe1, [lnorm0, fbcorr1]),
            np.random.randn(20, 3, 32, 32),
            batchsize=batchsize)

    feats = pyll.stochastic.sample(screen_features, np.random.RandomState(1))

    assert feats.shape[0] == 20
    assert feats.shape[1] == 16
    assert 22 <= feats.shape[2] < 32
    assert 22 <= feats.shape[3] < 32


def test_fuzz(test_len=1, catch_exceptions=False):
    bandit = cifar10bandit(
            # -- all-different numbers, small enough to be fast hopefully
            n_train=100,
            n_valid=50,
            n_test=40,
            n_imgs_for_patches=100,
            n_patches=1000,
            batchsize=10,
            # -- one-layer model tests the pipeline_exits and
            # pipeline_extension code
            max_layer_sizes=[64])
    algo = Random(bandit)
    # -- based on test_sampling_distribution this is
    trials = Trials()
    for i in range(test_len):
        try:
            trials = cPickle.load(open('test_fuzz_trials_%i.pkl' % i))
        except IOError:
            pass
            exp = Experiment(trials, algo)
            exp.catch_bandit_exceptions = catch_exceptions
            print '-' * 80
            print 'FUZZ ROUND', i
            print '-' * 80
            exp.run(1)
            cPickle.dump(trials, open('test_fuzz_trials_%i.pkl' % i, 'w'), -1)
    hyperopt.plotting.main_plot_vars(trials, bandit)

def test_foo():
    trials = cPickle.load(open('test_fuzz_trials_%i.pkl' % 80))
    bandit = cifar10bandit(
            # -- all-different numbers, small enough to be fast hopefully
            n_train=100,
            n_valid=50,
            n_test=40,
            n_imgs_for_patches=100,
            n_patches=1000,
            batchsize=10,
            # -- one-layer model tests the pipeline_exits and
            # pipeline_extension code
            max_layer_sizes=[64])
    #hyperopt.plotting.main_plot_vars(trials, bandit)
    hyperopt.plotting.main_plot_history(trials, bandit)

@nose.plugins.attrib.attr('slow')
def test_fuzz100():
    test_fuzz(100, catch_exceptions=False)

def test_sampling_distribution1():
    def evaluate(config, ctrl):
        return  {'loss': 9.0, 'status': hyperopt.STATUS_OK}

    trials = Trials()
    bandit = cifar10bandit(max_layer_sizes=[])
    bandit.evaluate = evaluate
    algo = Random(bandit)
    exp = Experiment(trials, algo)
    exp.catch_bandit_exceptions = False

    exp.run(100)

    hyperopt.plotting.main_plot_vars(trials, bandit)


def test_sampling_distribution2():
    def evaluate(config, ctrl):
        return  {'loss': 9.0, 'status': hyperopt.STATUS_OK}

    trials = Trials()
    bandit = cifar10bandit(max_layer_sizes=[64])
    bandit.evaluate = evaluate
    algo = Random(bandit)
    exp = Experiment(trials, algo)
    exp.catch_bandit_exceptions = False

    exp.run(100)

    hyperopt.plotting.main_plot_vars(trials, bandit)

def test_sampling_distribution3():
    def evaluate(config, ctrl):
        return  {'loss': 9.0, 'status': hyperopt.STATUS_OK}

    trials = Trials()
    bandit = cifar10bandit(max_layer_sizes=[64, 128])
    bandit.evaluate = evaluate
    algo = Random(bandit)
    exp = Experiment(trials, algo)
    exp.catch_bandit_exceptions = False

    exp.run(100)

    hyperopt.plotting.main_plot_vars(trials, bandit)
