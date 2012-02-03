import cPickle
import sys

import numpy as np

from hyperopt import Ctrl
from hyperopt.genson_helpers import choice
from hyperopt.genson_helpers import uniform

from skdata.digits import Digits

from genson import lazy
from genson import lazyinfo
from genson import JSONFunction

from .bandits import BaseBandit
from .classifier import get_result
from .margin_asgd import MarginBinaryASGD
from .margin_asgd import binary_fit
from .plugins import train_linear_svm_w_decisions
from .plugins import attach_svmasgd


@lazyinfo(len=2)
def digits_xy(n_examples):
    """
    Turn digits to binary classification: 0-4 vs. 5-9
    Returns, X, y as in classification task.
    """
    X, y = Digits().classification_task()
    y = (y < 5) * 2 - 1 # -- convert to +-1
    assert X.shape[1] == 64, X.shape[1]
    return X[:n_examples], y[:n_examples]


@lazyinfo(len=2)
def random_train_test_idxs(n_examples, n_fold, split_idx, rseed=42):
    """ Returns train_idxs, test_idxs of n_fold cross-validation.

    The examples are divided evenly into the folds, so that the train/test
    proportions are always the same, and up to n_fold - 1 examples may be
    wasted.

    """
    if not (0 <= split_idx < n_fold):
        raise ValueError('invalid split', (n_fold, split_idx))
    rstate = np.random.RandomState(rseed)
    n_usable = (n_examples // n_fold) * n_fold
    splits = rstate.permutation(n_examples)[:n_usable].reshape(n_fold, -1)

    test_idxs = splits[split_idx]
    train_idxs = np.asarray([splits[ii]
        for ii in range(n_fold) if ii != split_idx]).flatten()

    return train_idxs, test_idxs


@lazyinfo(len=3)
def slice_Xyd(Xy, decisions, idxs):
    """
    """
    X, y = Xy
    assert X.ndim == 2
    assert y.ndim == 1
    assert decisions.ndim == 1
    return X[idxs], y[idxs], decisions[idxs]



@lazyinfo(len=2)
def normalize_Xcols(train_Xyd, test_Xyd):
    X_train, y_train, d_train = train_Xyd
    X_test, y_test, d_test = test_Xyd

    Xm = X_train.mean(axis=0)
    Xs = X_train.std(axis=0) + 1e-7

    # think and add tests before working in-place here
    X_train = (X_train - Xm) / Xs
    X_test = (X_test - Xm) / Xs

    return [
            (X_train, y_train, d_train),
            (X_test, y_test, d_test),
            ]


@lazy
def features(Xyd, config):
    """Map from (X, y, d) -> (f(X), y, d)
    """
    X, y, d = Xyd

    rstate = np.random.RandomState(config['seed'])
    W = rstate.randn(X.shape[1], config['n_features']) * config['scale']
    b = rstate.randn(config['n_features']) * config['scale']

    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))
    feat = sigmoid(np.dot(X, W) + b)

    return (feat, y, d)


@lazy
def train_svm(Xyd, l2_regularization,):
    """
    Return a sklearn-like classification model.
    """
    train_X, train_y, decisions = Xyd
    if train_X.ndim != 2:
        raise ValueError('train_X must be matrix')
    assert len(train_X) == len(train_y) == len(decisions)
    svm = MarginBinaryASGD(
        n_features=train_X.shape[1],
        l2_regularization=l2_regularization,
        dtype=train_X.dtype,
        rstate=np.random.RandomState(123))
    binary_fit(svm, (train_X, train_y, np.asarray(decisions)))
    return svm


@lazy
def svm_decisions(svm, Xyd):
    X, y, d = Xyd
    inc = svm.decision_function(X)
    return d + inc

@lazy
def result_binary_classifier_stats(
        train_data,
        test_data,
        train_decisions,
        test_decisions,
        result):
    """
    Compute classification statistics and store them to scope['result']

    The train_decisions / test_decisions are the real-valued confidence
    score whose sign indicates the predicted class for binary
    classification.

    """
    result = dict(result)
    stats = get_result(train_data[1],
                             test_data[1],
                             np.sign(train_decisions),
                             np.sign(test_decisions),
                             [-1, 1])
    result.update(stats)
    result['loss'] = float(1 - result['test_accuracy']/100.)
    result['train_decisions'] = list(train_decisions)
    result['test_decisions'] = list(test_decisions)
    return result


@lazy
def combine_results(split_results, tt_idxs_list, new_ds, split_decisions, y):
    """
    Result has
        loss - scalar
        splits - sub-result dicts
        decisions - for next boosting round
    """
    result = dict(splits=split_results)

    # -- calculate the decisions
    new_decisions = np.zeros_like(split_decisions)
    for fold_idx, rr in enumerate(split_results):
        new_d_train, new_d_test = new_ds[fold_idx]
        train_idxs, test_idxs = tt_idxs_list[fold_idx]
        new_decisions[fold_idx][train_idxs] = new_d_train
        new_decisions[fold_idx][test_idxs] = new_d_test
    result['decisions'] = [list(dd) for dd in new_decisions]

    # -- calculate the loss
    #    XXX: is it right to return test margin mean?
    def rr_loss(rr):
        rr['loss']
    def test_margin_mean(ii, rr):
        labels = y[tt_idxs_list[ii][1]]
        margin = np.asarray(rr['test_decisions']) * labels
        return 1 - np.minimum(margin, 1).mean()
    result['loss'] = np.mean([test_margin_mean(ii, rr)
        for ii, rr in enumerate(split_results)])

    return result


@lazy
def attach_svm(ctrl, svm, name='svm'):
    obj = dict(weights=svm.asgd_weights, bias=svm.asgd_bias)
    ctrl.attachments[name] = cPickle.dumps(obj)


@lazy
def run_all(*args):
    return args

def screening_prog(n_examples, n_folds, feat_spec, split_decisions,
        save_svms):
    """
    Build a genson execution graph representing the experiment.
    """

    if split_decisions is None:
        split_decisions = np.zeros((n_folds, n_examples))

    Xy = digits_xy.lazy(n_examples)
    ctrl = JSONFunction.KWARGS['ctrl']

    # -- build a graph with n_folds paths
    split_results = []
    new_ds = []
    tt_idxs_list = []
    for fold_idx in range(n_folds):
        split_result = {}
        decisions = np.asarray(split_decisions[fold_idx])

        train_idxs, test_idxs = random_train_test_idxs.lazy(
                n_examples, n_folds, fold_idx)

        train_Xyd = slice_Xyd.lazy(Xy, decisions, train_idxs)
        test_Xyd = slice_Xyd.lazy(Xy, decisions, test_idxs)

        train_Xyd_n, test_Xyd_n = normalize_Xcols.lazy(train_Xyd, test_Xyd)

        train_Xyd_f = features.lazy(train_Xyd_n, feat_spec)
        test_Xyd_f = features.lazy(test_Xyd_n, feat_spec)

        train_Xyd_fn, test_Xyd_fn = normalize_Xcols.lazy(
                train_Xyd_f, test_Xyd_f)

        svm = train_svm.lazy(train_Xyd_fn, l2_regularization=1e-3)

        new_d_train = svm_decisions.lazy(svm, train_Xyd_fn)
        new_d_test = svm_decisions.lazy(svm, test_Xyd_fn)

        split_result = result_binary_classifier_stats.lazy(
                train_Xyd,
                test_Xyd,
                new_d_train,
                new_d_test,
                result=split_result)

        new_ds.append((new_d_train, new_d_test))
        tt_idxs_list.append((train_idxs, test_idxs))

        # -- if we save weights, then do this:
        if save_svms:
            split_results.append(run_all.lazy(
                split_result,
                attach_svm.lazy(ctrl, svm, 'svm_%i' % fold_idx,
                ))[0])
        else:
            split_results.append(split_result)


    result = combine_results.lazy(
            split_results,
            tt_idxs_list,
            new_ds,
            split_decisions,
            Xy[1])

    return result


class BoostableDigits(BaseBandit):
    if 0:
        # this causes bug in genson
        param_gen = dict(
                n_examples=1795,
                n_folds=5,
                feat_spec=dict(
                    seed=choice([1, 2, 3, 4, 5]),
                    n_features=choice([1, 5, 10]),
                    scale=uniform(0, 5)
                    ),
                split_decisions=None
                )
    else:
        param_gen = {}

    def evaluate(self, config, ctrl):
        if 'split_decisions' in ctrl.attachments:
            config['split_decisions'] = ctrl.attachments['split_decisions']
        prog = screening_prog(**config)
        fn = JSONFunction(prog)
        rval = fn(ctrl=ctrl)
        return rval


