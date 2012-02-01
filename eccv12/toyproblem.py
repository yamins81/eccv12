import sys

import numpy as np

from hyperopt.genson_helpers import choice
from hyperopt.genson_helpers import uniform

from skdata.iris import Iris

from .bandits import BaseBandit
from .plugins import train_linear_svm_w_decisions
from .plugins import result_binary_classifier_stats
from .plugins import svm_decisions

class BoostableIris(BaseBandit):
    param_gen = dict(
            seed=choice([1, 2, 3, 4, 5]),
            n_features=choice([1, 5, 10]),
            scale=uniform(0, 5),
            )

    def evaluate(self, config, ctrl):
        X, y = Iris().classification_task()
        y = (y == 0) * 2 - 1 # -- convert to +-1
        print len(y)
        decisions = getattr(self, 'decisions', None)
        if decisions is None:
            decisions = np.zeros_like(y)
        rstate = np.random.RandomState(42)
        splits = rstate.permutation(len(y)).reshape(5, -1)
        result = dict(split=[])

        split_decisions = []

        for split_idx in range(5):
            split_result = {}
            test_idxs = splits[split_idx]
            train_idxs = np.asarray([splits[ii]
                for ii in range(5) if ii != split_idx]).flatten()

            split_result['test_idxs'] = list(test_idxs)

            X_train = X[train_idxs]
            y_train = y[train_idxs]
            d_train = decisions[train_idxs]
            X_test = X[test_idxs]
            y_test = y[test_idxs]
            d_test = decisions[test_idxs]

            # XXX: where is sklearn plugin for this?
            Xm = X_train.mean(axis=0)
            Xs = X_train.std(axis=0) + 1e-7

            X_train = (X_train - Xm) / Xs
            X_test = (X_test - Xm) / Xs


            # -- allocate our model
            rstate = np.random.RandomState(config['seed'])
            W = rstate.randn(X.shape[1], config['n_features']) * config['scale']
            b = rstate.randn(config['n_features']) * config['scale']

            # -- apply it
            def sigmoid(X):
                return 1.0 / (1.0 + np.exp(-X))
            f_train = sigmoid(np.dot(X_train, W) + b)
            f_test = sigmoid(np.dot(X_test, W) + b)

            svm = train_linear_svm_w_decisions(
                    (X_train, y_train),
                    l2_regularization=1e-3,
                    decisions=d_train)

            new_d_train = svm_decisions(
                    svm,
                    (X_train, y_train),
                    d_train)

            new_d_test = svm_decisions(
                    svm,
                    (X_test, y_test),
                    d_test)

            result_binary_classifier_stats(
                    (X_train, y_train),
                    (X_test, y_test),
                    new_d_train,
                    new_d_test,
                    scope=dict(result=split_result)
                    )

            split_decisions.append(new_d_test)
            result['split'].append(split_result)

        all_idxs = splits.flatten()
        all_decisions = np.asarray(split_decisions).flatten()

        decisions = all_decisions[np.argsort(all_idxs)]
        # XXX: how to test that this indexing is right?
        result['decisions'] = list(decisions)
        return result


def test_for_smoke():
    BI = BoostableIris()
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=5,
                scale=2.2),
            ctrl=None)
    assert 'decisions' in result


def test_replicable():
    config=dict(
            seed=1,
            n_features=5,
            scale=2.2)
    r0, r1, r2 = [BoostableIris().evaluate(config, None) for ii in range(3)]

    for f in [
            lambda r: r['split'][0]['train_prediction'],
            lambda r: r['split'][4]['train_prediction'],
            lambda r: r['split'][2]['test_prediction'],
            lambda r: r['split'][3]['test_prediction'],
            ] :
        assert f(r0) == f(r1)
        assert f(r0) == f(r2)

    assert r0 == r1


def test_decisions_do_something():
    config=dict(seed=1, n_features=5, scale=2.2)

    B0 = BoostableIris()
    r0 = B0.evaluate(config, None)

    B1 = BoostableIris()
    B1.decisions = np.random.RandomState(32).randn(150)
    r1 = B1.evaluate(config, None)

    for f in [
            lambda r: r['split'][0]['train_prediction'],
            lambda r: r['split'][4]['train_prediction'],
            lambda r: r['split'][2]['test_prediction'],
            lambda r: r['split'][3]['test_prediction'],
            ] :
        assert f(r0) != f(r1)


def test_boosting_for_smoke():

    decisions = np.zeros(150)
    for i in range(100):
        BI = BoostableIris()
        BI.decisions = decisions
        result = BI.evaluate(
                config=dict(
                    seed=1,
                    n_features=5,
                    scale=2.2),
                ctrl=None)
        decisions = decisions + result['decisions']
        print abs(decisions).mean()


