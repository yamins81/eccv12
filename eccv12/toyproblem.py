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
def digits_xy():
    """
    Turn digits to binary classification: 0-4 vs. 5-9
    """
    X, y = Digits().classification_task()
    y = (y < 5) * 2 - 1 # -- convert to +-1
    return X, y


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
        X, y = Xy = digits_xy()
        print len(y)

        split_decisions = getattr(self, 'split_decisions', None)
        if split_decisions is None:
            split_decisions = [np.zeros_like(y) for ii in range(5)]
        scope=dict(
                ctrl=ctrl,
                result={'split':[]},
                )

        for split_idx in range(5):
            split_result = {}
            decisions = np.asarray(split_decisions[split_idx])
            scope['decisions'] = decisions

            train_test_idxs = digits_split(split_idx, (len(y) // 5) * 5)
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

            split_result['test_idxs'] = list(train_test_idxs[1])
            split_result['new_d_train'] = list(new_d_train)
            split_result['new_d_test'] = list(new_d_test)
            scope['result']['split'].append(split_result)

        decisions = np.zeros((5, len(y)))
        for ii, rr in enumerate(scope['result']['split']):
            train_idxs, test_idxs = digits_split(ii, (len(y) // 5) * 5)
            decisions[ii][train_idxs] = rr['new_d_train']
            decisions[ii][test_idxs] = rr['new_d_test']
        scope['result']['decisions'] = [list(dd) for dd in decisions]
        return scope['result']


def test_replicable():
    config=dict(
            seed=1,
            n_features=5,
            scale=2.2)
    ctrls = [Ctrl() for ii in range(3)]
    r0, r1, r2 = [BoostableDigits().evaluate(config, ctrl) for ctrl in ctrls]

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


def test_indexing():
    BI = BoostableDigits()
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=5,
                scale=2.2),
            ctrl=Ctrl())
    decisions = result['decisions']

    for rr in result['split']:
        print rr.keys()
        ti0, ti1 = rr['test_idxs'][:2]
        td0, td1 = rr['new_d_test'][:2]
        ti2, ti3 = rr['test_idxs'][-2:]
        td2, td3 = rr['new_d_test'][-2:]

        assert decisions[ti0] == td0
        assert decisions[ti1] == td1
        assert decisions[ti2] == td2
        assert decisions[ti3] == td3



def test_decisions_do_something():
    config=dict(seed=1, n_features=5, scale=2.2)

    ctrl0 = Ctrl()
    ctrl1 = Ctrl()

    B0 = BoostableDigits()
    r0 = B0.evaluate(config, ctrl0)

    B1 = BoostableDigits()
    B1.decisions = np.random.RandomState(32).randn(150)
    r1 = B1.evaluate(config, ctrl1)

    # contrast test_replicable where they come out the same
    svm0 = cPickle.loads(ctrl0.attachments['svm'])
    svm1 = cPickle.loads(ctrl1.attachments['svm'])
    assert not np.allclose(svm0['weights'], svm1['weights'])


def test_overfitting():
    for log_n_features in range(7, 13):
        BI = BoostableDigits()
        result = BI.evaluate(
                config=dict(
                    seed=1,
                    n_features=2 ** log_n_features,
                    scale=2.2),
                ctrl=Ctrl())
        for key in 'train_accuracy', 'test_accuracy':
            print 2 ** log_n_features, key,
            print np.mean([r[key] for r in result['split']])

    # XXX: assert something ... what does this prove?


def test_boosting_for_smoke():

    n_rounds = 16
    n_features_per_round = 16

    # -- train jointly
    print 'Training jointly'
    BI = BoostableDigits()
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=n_rounds * n_features_per_round,
                scale=2.2),
            ctrl=Ctrl())
    for key in 'train_accuracy', 'test_accuracy':
        print key, np.mean([r[key] for r in result['split']])

    # -- train just one
    print 'Training one round'
    BI = BoostableDigits()
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=n_features_per_round,
                scale=2.2),
            ctrl=Ctrl())
    for key in 'train_accuracy', 'test_accuracy':
        print key, np.mean([r[key] for r in result['split']])

    # -- train in rounds
    print 'Training in rounds'
    decisions = None
    for round_ii in range(n_rounds):
        BI = BoostableDigits()
        BI.decisions = decisions
        result = BI.evaluate(
                config=dict(
                    seed=round_ii,
                    n_features=n_features_per_round,
                    scale=2.2),
                ctrl=Ctrl())
        if decisions is None:
            decisions = np.asarray(result['decisions'])
        else:
            decisions += result['decisions']
        assert len(decisions) == 5
        assert decisions.ndim == 2
        print 'abs decisions', abs(decisions).mean()
        for key in 'train_accuracy', 'test_accuracy':
            print key, np.mean([r[key] for r in result['split']])

