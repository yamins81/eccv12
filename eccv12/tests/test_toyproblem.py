import cPickle

from nose.plugins.attrib import attr
import numpy as np

from eccv12 import toyproblem
from hyperopt import Ctrl
import pyll

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

config_tiny=dict(
        n_examples_train=200,
        n_examples_test=100,
        n_folds=2,
        feat_spec=dict(seed=1, n_features=5, scale=2.2),
        decisions=None,
        svm_l2_regularization=1e-3,
        svm_max_observations=1e3,
        save_svms=True,
        )


def test_random_train_test_idxs():
    for n_examples in 66, 100:
        train, test = toyproblem.random_train_test_idxs(n_examples, n_fold=4, split_idx=2)
        assert len(set(train)) == len(train)
        assert len(set(test)) == len(test)
        assert not set(train).intersection(test)

    # -- the test sets cover the whole range, and do not overlap
    tests = [toyproblem.random_train_test_idxs(50, 5, ii)[1] for ii in range(5)]
    tests = np.asarray(tests)
    assert tests.size == 50
    assert set(tests.flatten()) == set(range(50))

    # -- when the number of examples doesn't fit evenly into the folds,
    #    some examples are dropped.
    tests = [toyproblem.random_train_test_idxs(54, 5, ii)[1] for ii in range(5)]
    tests = np.asarray(tests)
    assert tests.size == 50
    assert len(set(tests.flatten())) == 50


def test_screening_prog_for_smoke():
    # smoke test
    prog = toyproblem.screening_prog(ctrl=Ctrl(None), **config_tiny)
    sprog = str(prog)
    #print sprog
    rval = pyll.rec_eval(prog)
    #print rval
    assert 'loss' in rval
    assert 'decisions' in rval
    assert len(rval['splits']) == 2
    assert rval['splits'][0] != rval['splits'][1]


def test_replicable():
    ctrls = [Ctrl(None) for ii in range(3)]

    def foo(ctrl):
        BI = toyproblem.BoostableDigits()
        return BI.evaluate(config_tiny, ctrl)
    r0, r1, r2 = map(foo, ctrls)

    for f in [
            lambda r: r['splits'][0]['train_prediction'],
            lambda r: r['splits'][1]['train_prediction'],
            lambda r: r['splits'][0]['test_prediction'],
            lambda r: r['splits'][1]['test_prediction'],
            ] :
        assert f(r0) == f(r1)
        assert f(r0) == f(r2)

    svm0 = cPickle.loads(ctrls[0].attachments['svm_0'])
    svm1 = cPickle.loads(ctrls[1].attachments['svm_0'])
    assert np.allclose(svm0['weights'], svm1['weights'])

    assert r0 == r1


def test_decisions_do_something():
    ctrl0 = Ctrl(None)
    ctrl1 = Ctrl(None)

    B0 = toyproblem.BoostableDigits()
    B0.split_decisions = None
    r0 = B0.evaluate(config_tiny, ctrl0)

    rstate = np.random.RandomState(32)
    B1 = toyproblem.BoostableDigits()
    config_decisions = dict(config_tiny)
    config_decisions['decisions'] = rstate.randn(
            config_tiny['n_folds'],
            config_tiny['n_examples_train'])
    r1 = B1.evaluate(config_decisions, ctrl1)

    # contrast test_replicable where they come out the same
    svm0 = cPickle.loads(ctrl0.attachments['svm_0'])
    svm1 = cPickle.loads(ctrl1.attachments['svm_0'])
    assert not np.allclose(svm0['weights'], svm1['weights'])


@attr('slow')
def test_boosting_margin_goes_down():
    n_examples = 1750
    X, y = toyproblem.digits_xy(0, n_examples)
    n_rounds = 8
    margins = []
    decisions = None
    for round_ii in range(n_rounds):
        config = dict(
            n_examples_train=n_examples,
            n_examples_test=0,
            n_folds=5,
            feat_spec=dict(seed=round_ii, n_features=16, scale=2.2),
            decisions=decisions,
            save_svms=False,       # good to test False sometimes
            svm_l2_regularization=1e-3,
            svm_max_observations=1e3,
            )
        ctrl = Ctrl(None)
        BI = toyproblem.BoostableDigits()
        result = BI.evaluate(config, ctrl)
        decisions = np.asarray(result['decisions'])
        assert decisions.shape == (5, 1750)
        print 'mean abs decisions', abs(decisions).mean(),
        margins.append(1 - np.minimum(decisions * y, 1).mean())
        for key in 'train_accuracy', 'test_accuracy':
            print key, np.mean([rr[key] for rr in result['splits']]),
        print ''
    print margins
    print list(reversed(margins))
    print list(sorted(margins))
    assert list(reversed(margins)) == list(sorted(margins))


def mean_acc(result):
    tr_acc = np.mean([r['train_accuracy'] for r in result['splits']])
    te_acc = np.mean([r['test_accuracy'] for r in result['splits']])
    return tr_acc, te_acc


def train_run(n_examples, n_features, decisions_in, rseed=1):
    # -- need y for margin computation
    X, y = toyproblem.digits_xy(0, n_examples)
    assert len(y) == n_examples
    config = dict(
        n_examples_train=n_examples,
        n_examples_test=0,
        n_folds=5,
        feat_spec=dict(seed=rseed,
            n_features=n_features,
            scale=2.2),
        decisions=decisions_in, # gets clobbered by ctrl attachment
        save_svms=False,      # good to test False sometimes
        svm_l2_regularization=1e-3,
        svm_max_observations=1e5,
        )
    ctrl = Ctrl(None)
    BI = toyproblem.BoostableDigits()
    result = BI.evaluate(config, ctrl)
    decisions = np.asarray(result['decisions'])
    assert decisions.shape == (5, n_examples)
    print decisions.shape
    print 'mean abs decisions', abs(decisions).mean(),
    print 'mean margins', 1 - np.minimum(decisions * y, 1).mean(),
    tr_acc, te_acc = mean_acc(result)
    print 'train_accuracy', tr_acc,
    print 'test_accuracy', te_acc,
    print ''
    return decisions, tr_acc, te_acc


@attr('slow')
def test_boosting_for_smoke():
    n_examples = 1790
    X, y = toyproblem.digits_xy(0, n_examples)
    assert len(y) == n_examples

    n_rounds = 16
    n_features_per_round = 16

    print 'Training jointly'
    _, joint_tr_acc, joint_te_acc = train_run(
            n_examples,
            n_rounds * n_features_per_round,
            None)

    print 'Training one round'
    _, one_tr_acc, one_te_acc = train_run(
            n_examples,
            n_features_per_round,
            None)

    # -- train in rounds
    print 'Training in rounds'
    decisions = None
    for round_ii in range(n_rounds):
        decisions, tr_acc, te_acc = train_run(
                n_examples,
                n_features_per_round,
                decisions,
                rseed=round_ii)

    # assert that round-training and joint training are both way better than
    # training just one
    assert joint_tr_acc > 95
    assert joint_te_acc > 88
    assert one_tr_acc < 72
    assert one_te_acc < 72
    assert tr_acc > 90
    assert te_acc > 88

