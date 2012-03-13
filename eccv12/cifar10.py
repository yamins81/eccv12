import time
import numpy as np

import pyll
from pyll.base import partial1
from pyll.base import Lambda
from pyll import scope
import pyll_slm

pyll.scope.import_(globals(),
    'pyll_theano_batched_lmap',
    'cifar10_img_classification_task',
    'linsvm_train_test',
    'hyperopt_set_loss',
    'np_transpose',
    # /begin distributions that hyperopt can tune
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'one_of',
    'choice',
    # /end distributions...
    **{
    #NEW NAME:  ORIG NAME
    'lnorm': 'slm_lnorm',
    'lpool': 'slm_lpool',
    'fbcorr': 'slm_fbcorr',
    's_int': 'int',
    })


def rfilter_size(smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(quniform(smin - q + 1e-5, smax, q))


def logu_range(lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return loguniform(np.log(lower), np.log(upper))


def experiment1(namebase, n_train, n_valid, n_test,
        use_l3='no',  # can be yes, no, maybe
        batchsize=10,
        C=1.0):
    # -- set up the img_i -> top_layer SLM feature pipeline

    lnorm0 = partial1(lnorm,
            ker_size=rfilter_size(2, 6),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    fbcorr1 = partial1(fbcorr,
            ker_size=rfilter_size(2, 6),
            n_filters=s_int(
                qloguniform(
                    np.log(16 / 2.0),
                    np.log(64),
                    q=16)),
            generate=('random:uniform',
                {'rseed': choice(range(10, 15))}))

    lpool1 = partial1(lpool,
            ker_size=rfilter_size(2, 5),
            order=loguniform(np.log(1), np.log(10)),
            stride=1)

    lnorm1 = partial1(lnorm,
            ker_size = rfilter_size(2, 5),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    fbcorr2 = partial1(fbcorr,
            ker_size=rfilter_size(2, 6),
            n_filters=s_int(
                qloguniform(
                    np.log(16 / 2.0),
                    np.log(128),
                    q=16)),
            generate=('random:uniform',
                {'rseed': choice(range(20, 25))}))

    lpool2 = partial1(lpool,
            ker_size=rfilter_size(2, 5),
            order=loguniform(np.log(1), np.log(10)),
            stride=1)

    lnorm2 = partial1(lnorm,
            ker_size = rfilter_size(2, 5),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    fbcorr3 = partial1(fbcorr,
            ker_size=rfilter_size(2, 6),
            n_filters=s_int(
                qloguniform(
                    np.log(16 / 2.0),
                    np.log(384),
                    q=16)),
            generate=('random:uniform',
                {'rseed': choice(range(30, 35))}))

    lpool3 = partial1(lpool,
            ker_size=rfilter_size(2, 5),
            order=loguniform(np.log(1), np.log(10)),
            stride=1)

    lnorm3 = partial1(lnorm,
            ker_size = rfilter_size(2, 5),
            remove_mean=one_of(0, 1),
            stretch=logu_range(.1/3, 10.*3),
            threshold=logu_range(.1/3, 10.*3))

    pipeline = [lnorm0,
            fbcorr1, lpool1, lnorm1,
            fbcorr2, lpool2, lnorm2,
            ]

    if 0:
        top_layer = dict(
                yes=layer3,
                no=layer2,
                maybe=one_of(layer2, layer3)
                )[use_l3]

    assert n_train + n_valid < 50000
    assert n_test < 10000

    # -- map cifar10 through the pipeline
    all_imgs, all_labels = cifar10_img_classification_task()
    all_imgs = np_transpose(all_imgs, (0, 3, 1, 2))
    screen_features = pyll_theano_batched_lmap(
            pipeline,
            all_imgs[:n_train + n_valid],
            batchsize=batchsize)
    test_features = pyll_theano_batched_lmap(
            pipeline,
            all_imgs[50000:50000 + n_test],
            batchsize=batchsize)

    result = linsvm_train_test(
            train=(screen_features[:n_train], all_labels[:n_train]),
            test_sets = dict(
                valid=(
                    screen_features[n_train:n_train + n_valid],
                    all_labels[n_train:n_train + n_valid]
                    ),
                test=(
                    test_features,
                    all_labels[50000:50000 + n_test],
                    ),
                ),
            C=C,
            normalize_cols=True,
            allow_inplace=True,
            report={
                'col_affine': 0,
                'svm': 0,
                'train.error_rate': 1,
                'train.fit_time': 1,
                'test.valid.error_rate': 1,
                'test.valid.accuracy': 1,
                'test.test.error_rate': 1,
                'test.test.accuracy': 1,
                'train.sgd_step_size0': 1,
                })

    result_w_loss = hyperopt_set_loss(result, 'test.valid.error_rate')

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
    allvars = experiment1('memmap_test_cifar10',
            n_train=10, n_valid=10,  n_test=10,
            use_l3='no',
            batchsize=10)
    result_w_loss = allvars['result_w_loss']
    # make the experiment fully runnable by installing a random generator
    rng = np.random.RandomState(234)
    pyll.stochastic.recursive_set_rng_kwarg(result_w_loss, rng)
    i = 0
    while True:
        try:
            result = pyll.rec_eval(result_w_loss)
            break
        except pyll_slm.InvalidDescription:
            print 'Invalid Description', i
            i += 1
            continue

    print result

