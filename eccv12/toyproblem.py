import cPickle
import sys

import numpy as np

from hyperopt import Ctrl
from hyperopt.genson_helpers import choice
from hyperopt.genson_helpers import uniform

from skdata.digits import Digits

from .bandits import BaseBandit
from .plugins import train_linear_svm_w_decisions
from .plugins import result_binary_classifier_stats
from .plugins import svm_decisions
from .plugins import attach_svmasgd
from .plugins import register


@register()
def digits_xy(n_splits=5):
    """
    Turn digits to binary classification: 0-4 vs. 5-9
    """
    X, y = Digits().classification_task()
    y = (y < 5) * 2 - 1 # -- convert to +-1
    n_usable = (len(y) // n_splits) * n_splits
    return X[:n_usable], y[:n_usable]


@register(call_w_scope=True)
def scope_lookup(key, scope):
    return scope[key]


@register()
def digits_split(split_num, n_examples):
    rstate = np.random.RandomState(42)
    splits = rstate.permutation(n_examples).reshape(5, -1)
    result = dict(split=[])

    test_idxs = splits[split_num]
    train_idxs = np.asarray([splits[ii]
        for ii in range(5) if ii != split_num]).flatten()

    return train_idxs, test_idxs


@register()
def Xy_split(Xy, decisions, idxs):
    X, y = Xy
    train_idxs, test_idxs = idxs

    return [
            (X[train_idxs], y[train_idxs], decisions[train_idxs]),
            (X[test_idxs], y[test_idxs], decisions[test_idxs])
            ]

@register()
def normalize_Xcols(train_test):
    train, test = train_test
    X_train, y_train, d_train = train
    X_test, y_test, d_test = test

    Xm = X_train.mean(axis=0)
    Xs = X_train.std(axis=0) + 1e-7

    # think and add tests before working in-place here
    X_train = (X_train - Xm) / Xs
    X_test = (X_test - Xm) / Xs

    return [
            (X_train, y_train, d_train),
            (X_test, y_test, d_test),
            ]

@register()
def features(config, train_test):
    # -- unpack args
    train, test = train_test
    X_train, y_train, d_train = train
    X_test, y_test, d_test = test

    # -- allocate our model
    rstate = np.random.RandomState(config['seed'])
    W = rstate.randn(X_train.shape[1], config['n_features']) * config['scale']
    b = rstate.randn(config['n_features']) * config['scale']

    # -- apply it
    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))
    f_train = sigmoid(np.dot(X_train, W) + b)
    f_test = sigmoid(np.dot(X_test, W) + b)

    return [
            (f_train, y_train, d_train),
            (f_test, y_test, d_test),
            ]


class BoostableDigits(BaseBandit):
    param_gen = dict(
            seed=choice([1, 2, 3, 4, 5]),
            n_features=choice([1, 5, 10]),
            scale=uniform(0, 5),
            )

    def evaluate(self, config, ctrl):
        n_splits = 5

        X, y = Xy = digits_xy(n_splits)

        if self.split_decisions is None:
            split_decisions = [np.zeros(len(y), dtype=X.dtype) for ii in range(n_splits)]
        else:
            split_decisions = self.split_decisions

        new_decisions = np.zeros_like(split_decisions)

        scope=dict(
                ctrl=ctrl,
                result={'split':[]},
                )

        for split_idx in range(n_splits):
            split_result = {}
            decisions = np.asarray(split_decisions[split_idx])
            scope['decisions'] = decisions

            train_test_idxs = digits_split(split_idx, len(y))
            train_test = Xy_split(Xy, decisions, train_test_idxs)
            train_test_norm = normalize_Xcols(train_test)
            train_test_feat = features(config, train_test_norm)
            train_test_feat_norm = normalize_Xcols(train_test_feat)

            (f_train, y_train, d_train) = train_test_feat_norm[0]
            (f_test, y_test, d_test) = train_test_feat_norm[1]

            svm = train_linear_svm_w_decisions(
                    (f_train, y_train),
                    l2_regularization=1e-3,
                    decisions=d_train)

            new_d_train = svm_decisions(
                    svm,
                    (f_train, y_train),
                    d_train)

            new_d_test = svm_decisions(
                    svm,
                    (f_test, y_test),
                    d_test)

            result_binary_classifier_stats(
                    (f_train, y_train),
                    (f_test, y_test),
                    new_d_train,
                    new_d_test,
                    scope=dict(result=split_result)
                    )

            attach_svmasgd(svm, 'svm', scope=dict(ctrl=ctrl))

            # -- update result
            split_result['test_idxs'] = list(train_test_idxs[1])
            scope['result']['split'].append(split_result)

            # -- update new_decisions
            train_idxs, test_idxs = train_test_idxs
            new_decisions[split_idx][train_idxs] = new_d_train
            new_decisions[split_idx][test_idxs] = new_d_test

        scope['result']['decisions'] = [list(dd) for dd in new_decisions]
        def rr_loss(rr):
            rr['loss']
        def test_margin_mean(rr):
            labels = y[rr['test_idxs']]
            margin = np.asarray(rr['test_decisions']) * labels
            return 1 - np.minimum(margin, 1).mean()
        # XXX: is it right to return test margin mean?
        loss_fn = test_margin_mean
        scope['result']['loss'] = np.mean([loss_fn(rr)
            for rr in scope['result']['split']])
        return scope['result']


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_for_smoke():
    BI = BoostableDigits()
    BI.split_decisions = None
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=5,
                scale=2.2),
            ctrl=Ctrl())

    assert 'loss' in result
    assert 'decisions' in result


def test_replicable():
    config=dict(
            seed=1,
            n_features=5,
            scale=2.2)
    ctrls = [Ctrl() for ii in range(3)]

    def foo(ctrl):
        BI = BoostableDigits()
        BI.split_decisions = None
        return BI.evaluate(config, ctrl)
    r0, r1, r2 = map(foo, ctrls)

    for f in [
            lambda r: r['split'][0]['train_prediction'],
            lambda r: r['split'][4]['train_prediction'],
            lambda r: r['split'][2]['test_prediction'],
            lambda r: r['split'][3]['test_prediction'],
            ] :
        assert f(r0) == f(r1)
        assert f(r0) == f(r2)

    svm0 = cPickle.loads(ctrls[0].attachments['svm'])
    svm1 = cPickle.loads(ctrls[1].attachments['svm'])
    assert np.allclose(svm0['weights'], svm1['weights'])

    assert r0 == r1


def test_decisions_do_something():
    config=dict(seed=1, n_features=5, scale=2.2)
    X, y = digits_xy()

    ctrl0 = Ctrl()
    ctrl1 = Ctrl()

    B0 = BoostableDigits()
    B0.split_decisions = None
    r0 = B0.evaluate(config, ctrl0)

    B1 = BoostableDigits()
    B1.split_decisions = np.random.RandomState(32).randn(5, len(y))
    r1 = B1.evaluate(config, ctrl1)

    # contrast test_replicable where they come out the same
    svm0 = cPickle.loads(ctrl0.attachments['svm'])
    svm1 = cPickle.loads(ctrl1.attachments['svm'])
    assert not np.allclose(svm0['weights'], svm1['weights'])


def test_boosting_margin_goes_down():
    X, y = digits_xy()
    n_rounds = 8
    n_features_per_round = 16
    split_decisions = None
    margins = []
    for round_ii in range(n_rounds):
        BI = BoostableDigits()
        BI.split_decisions = split_decisions
        result = BI.evaluate(
                config=dict(
                    seed=round_ii,
                    n_features=n_features_per_round,
                    scale=2.2),
                ctrl=Ctrl())
        split_decisions = np.asarray(result['decisions'])
        assert len(split_decisions) == 5
        assert split_decisions.ndim == 2
        print 'mean abs decisions', abs(split_decisions).mean(),
        margins.append(1 - np.minimum(split_decisions * y, 1).mean())
        for key in 'train_accuracy', 'test_accuracy':
            print key, np.mean([r[key] for r in result['split']]),
        print ''
    print margins
    print list(reversed(margins))
    print list(sorted(margins))
    assert list(reversed(margins)) == list(sorted(margins))


def test_boosting_for_smoke():
    X, y = digits_xy()
    print len(y)

    n_rounds = 16
    n_features_per_round = 16

    # -- train jointly
    print 'Training jointly'
    BI = BoostableDigits()
    BI.split_decisions = None
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=n_rounds * n_features_per_round,
                scale=2.2),
            ctrl=Ctrl())
    decisions = np.asarray(result['decisions'])
    print decisions.shape
    print 'mean abs decisions', abs(decisions).mean(),
    print 'mean margins', 1 - np.minimum(decisions * y, 1).mean(),
    tr_acc = np.mean([r['train_accuracy'] for r in result['split']])
    te_acc = np.mean([r['test_accuracy'] for r in result['split']])
    print 'train_accuracy', tr_acc,
    print 'test_accuracy', te_acc,
    print ''

    # -- train just one
    print 'Training one round'
    BI = BoostableDigits()
    BI.split_decisions = None
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=n_features_per_round,
                scale=2.2),
            ctrl=Ctrl())
    decisions = np.asarray(result['decisions'])
    print 'mean abs decisions', abs(decisions).mean(),
    print 'mean margins', 1 - np.minimum(decisions * y, 1).mean(),
    one_tr_acc = np.mean([r['train_accuracy'] for r in result['split']])
    one_te_acc = np.mean([r['test_accuracy'] for r in result['split']])
    print 'train_accuracy', one_tr_acc,
    print 'test_accuracy', one_te_acc,
    print ''

    # -- train in rounds
    print 'Training in rounds'
    split_decisions = None
    for round_ii in range(n_rounds):
        BI = BoostableDigits()
        BI.split_decisions = split_decisions
        result = BI.evaluate(
                config=dict(
                    seed=round_ii,
                    n_features=n_features_per_round,
                    scale=2.2),
                ctrl=Ctrl())
        split_decisions = np.asarray(result['decisions'])
        assert len(split_decisions) == 5
        assert split_decisions.ndim == 2
        print 'mean abs decisions', abs(split_decisions).mean(),
        print 'mean margins', 1 - np.minimum(split_decisions * y, 1).mean(),
        round_tr_acc = np.mean([r['train_accuracy'] for r in result['split']])
        round_te_acc = np.mean([r['test_accuracy'] for r in result['split']])
        print 'train_accuracy', round_tr_acc,
        print 'test_accuracy', round_te_acc,
        print ''

    # assert that round-training and joint training are both way better than
    # training just one
    assert tr_acc > 90
    assert round_tr_acc > 90
    assert one_tr_acc < 70


def test_random_search_boosting():
    X, y = digits_xy()
    print len(y)

    def foo(n_candidates):

        n_rounds = 16
        rstate = np.random.RandomState(123)  # for sampling configs

        # On every round of boosting,
        # draw some candidates, and select the best among them to join the
        # ensemble.

        selected = []
        for round_ii in range(n_rounds):
            print 'ROUND', round_ii
            candidates = []
            for candidate_ii in range(n_candidates):
                print ' CANDIDATE', candidate_ii
                BI = BoostableDigits()
                if selected:
                    BI.split_decisions = selected[-1]['decisions']
                else:
                    BI.split_decisions = None
                result = BI.evaluate(
                        config=dict(
                            seed=int(rstate.randint(2**31)),
                            n_features=int(rstate.randint(2, 32)),
                            scale=float(np.exp(rstate.randn())),
                            ),
                        ctrl=Ctrl())
                print '  loss', result['loss']
                candidates.append(result)

            # loss is the average test set err across the internal folds
            loss = np.array([rr['loss'] for rr in candidates])
            selected_ind = loss.argmin()
            best = candidates[selected_ind]
            if len(selected):
                # XXX: what are we supposed to do here?
                # among the pool of candidates it is not necessarily the case
                # that any of them makes an improvement on the validation set
                if best['loss'] < selected[-1]['loss']:
                    selected.append(best)
                else:
                    print 'SKIPPING CRAPPY BEST CANDIDATE'
                    pass
            else:
                selected.append(best)

            split_decisions = best['decisions']
            print 'mean margins', 1 - np.minimum(split_decisions * y, 1).mean(),
            round_tr_acc = np.mean([rr['train_accuracy'] for rr in best['split']])
            round_te_acc = np.mean([rr['test_accuracy'] for rr in best['split']])
            print 'train_accuracy', round_tr_acc,
            print 'test_accuracy', round_te_acc,
            print ''
        return selected

    r1 = foo(1)
    r2 = foo(2)
    r5 = foo(5)

    assert r1[-1]['loss'] > r2[-1]['loss']
    assert r2[-1]['loss'] > r5[-1]['loss']

