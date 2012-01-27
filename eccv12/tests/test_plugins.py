import cPickle
import numpy as np

import skdata.lfw
from hyperopt.base import Ctrl

#from eccv12.fson import fson_print, fson_eval, register

#from eccv12.plugins import fetch_train_decisions
#from eccv12.plugins import fetch_test_decisions
#from eccv12.plugins import lfw_images
#from eccv12.plugins import slm_memmap
from eccv12 import plugins
from eccv12.fson import fson_print, fson_eval


def ctrl_stub():
    ctrl = Ctrl()
    rng = np.random.RandomState(5234)
    dataset = skdata.lfw.Aligned()
    for split in 'DevTrain', 'DevTest':
        n_examples = len(dataset.raw_verification_task(split)[0])
        ctrl.attachments['decisions_%s' % split] = cPickle.dumps(
            rng.randn(n_examples))
    return ctrl


def test_print_screening_program():
    fson_print(plugins.screening_program(
        slm_desc='slm_desc',
        comparison='sumsqdiff',
        namebase='namebase'))


def test_get_images():
    X = plugins.get_images(dtype='float32')
    # XXX: are these faces supposed to be greyscale?
    assert X.dtype == 'float32'
    print X[0].sum()
    assert X[0].sum() != 0
    assert X.shape == (13233, 200, 200), X.shape


def test_verification_pairs_0():
    l, r = plugins._verification_pairs_helper(
        ['a', 'b', 'z', 'c'],
        ['b', 'z', 'a'],
        ['c', 'b', 'z'])
    assert list(l) == [1, 2, 0]
    assert list(r) == [3, 1, 2]

def test_verification_pairs_1():
    l, r, m = plugins.verification_pairs('DevTest')
    print set(m)
    # XXX: is there really just 1000 verification pairs in DevTest??
    # That's so hard for doing screening!?
    assert len(l) == len(r) == len(m) == 1000
    assert set(m) == set([-1, 1])


slm_desc = [
    # -- Layer 0
    [('lnorm', {'kwargs': {'inker_shape': (3, 3)}}),
     ('lpool', {'kwargs': {'stride': 2}}),
    ],

    # -- Layer 1
    [('fbcorr', {'kwargs': {'min_out': 0},
                 'initialize': {
                     'n_filters': 16,
                     'filter_shape': (3, 3),
                     'generate': ('random:uniform', {'rseed': 42}),
                 },
                }),
     ('lpool', {'kwargs': {'ker_shape': (3, 3), 'stride':2}}),
     ('lnorm', {'kwargs': {'inker_shape': (3, 3)}}),
    ],

    # -- Layer 2
    [('fbcorr', {'kwargs': {'min_out': 0},
                 'initialize': {
                     'n_filters': 16,
                     'filter_shape': (3, 3),
                     'generate': ('random:uniform', {'rseed': 42}),
                 },
                }),
     ('lpool', {'kwargs': {'ker_shape': (3, 3), 'stride': 2}}),
     ('lnorm', {'kwargs': {'inker_shape': (3, 3)}}),
    ],
]

def test_slm_memmap():
    rng = np.random.RandomState(123)
    feats = plugins.slm_memmap(slm_desc, rng.randn(5, 200, 200), 'test_foo')
    try:
        assert feats[3].sum() != 0.0
        assert feats[3].shape[-1] == 16
        assert np.all(feats._valid == [0, 0, 0, 1, 0])

    finally:
        feats.delete_files()
    # XXX: test that using feats after delete_files does something
    # sensible - either it works, or it raises an error. This should be
    # tested in skdata.

def test_pairs_memmap_mult():
    lidxs = [0, 2, 4]
    ridxs = [5, 3, 1]
    matches = [1, 1, -1]
    X = [[1, 2],
         [3, 4],
         [5, 6],
         [7, 8],
         [9, 10],
         [11, 12]]
    pf, labels = plugins.pairs_memmap(
        (lidxs, ridxs, matches),
        np.asarray(X),
        'mult',
        'test_foo')
    try:
        assert np.all(matches == labels)
        pf.populate()
        print pf
        print pf[:]
        assert np.all(pf[:] == [
            [11, 24],
            [35, 48],
            [27, 40]])
    finally:
        pf.delete_files()

def test_train_linear_svm_w_decisions():
    """
    """
    rng = np.random.RandomState(123)

    train_X = rng.randn(10, 4)
    train_y = np.sign(rng.randn(10))
    decisions = rng.randn(10)
    
    model = plugins.train_linear_svm_w_decisions(
        (train_X, train_y),
        l2_regularization=1e-4,
        decisions=decisions)

    print model.asgd_weights
    print model.asgd_bias
    #XXX: test something better

    assert model.asgd_bias != 0.0


def test_results_binary_classifier_stats():
    scope = {}
    plugins.results_binary_classifier_stats(
        train_data=(None, [-1, -1, 1, 1]),
        test_data=(None, [-1, 1]),
        train_decisions=[.1, -.3, .3, .2],
        test_decisions=[.2, .3],
        scope=scope,
        )
    print scope
    result = scope['result']
    assert result['train_accuracy'] == 75.
    assert result['test_accuracy'] == 50.
    assert result['loss'] == .5


def test_prog_1():
    """
    This tests the screening program as far as image feature extraction.
    """
    dct = plugins.screening_program(slm_desc, 'sqrtabsdiff', 'foobase')
    scope = {}
    X = fson_eval(dct['image_features'], scope=scope)
    assert len(X) == 13233
    print X[0].sum()
    print X[2100:].sum()
    # XXX: more asserts


def test_prog_2():
    """
    This tests the screening program as far as creating an svm training set.
    """
    dct = plugins.screening_program(slm_desc, 'sqrtabsdiff', 'foobase')
    scope = {}
    X, y = fson_eval(dct['train_verification_dataset'], scope=scope)
    assert len(X) == 2200 == len(y)
    print X[0].sum()
    print X[2100:].sum()
    # XXX: more asserts


def test_prog_all():
    """
    This actually runs an entire screening experiment based on slm_desc
    """
    dct = plugins.screening_program(slm_desc, 'sqrtabsdiff', 'foobase')
    scope = dict(ctrl=ctrl_stub())
    fson_eval(dct['rval'], scope=scope)
    print scope
    # XXX: more asserts

