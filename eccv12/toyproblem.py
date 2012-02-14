import cPickle

import numpy as np

import hyperopt
from skdata.digits import Digits

import pyll
from pyll import scope

from asgd.auto_step_size import binary_fit

from .bandits import BaseBandit
from .classifier import get_result
from .margin_asgd import MarginBinaryASGD


@scope.define_info(o_len=2)
def digits_xy(begin, N):
    """
    Turn digits to binary classification: 0-4 vs. 5-9
    Returns, X, y as in classification task.
    """
    X, y = Digits().classification_task()
    np.random.RandomState(42).shuffle(X)
    np.random.RandomState(42).shuffle(y)
    y = (y < 5) * 2 - 1 # -- convert to +-1
    assert X.shape[1] == 64, X.shape[1]
    return X[begin:begin+N], y[begin:begin+N]


@scope.define_info(o_len=2)
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


@scope.define_info(o_len=3)
def slice_Xyd(Xy, decisions, idxs):
    """
    """
    X, y = Xy
    assert X.ndim == 2
    assert y.ndim == 1
    assert decisions.ndim == 1
    return X[idxs], y[idxs], decisions[idxs]


@scope.define_info(o_len=2)
def normalize_Xcols(train_Xyd, test_Xyd):
    X_train, y_train, d_train = train_Xyd
    X_test, y_test, d_test = test_Xyd

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    Xm = X_train.mean(axis=0)
    Xs = np.maximum(X_train.std(axis=0), 1e-8)

    # think and add tests before working in-place here
    X_train = (X_train - Xm) / Xs
    X_test = (X_test - Xm) / Xs

    return [
            (X_train, y_train, d_train),
            (X_test, y_test, d_test),
            ]


@scope.define
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


@scope.define
def train_svm(Xyd, l2_regularization, max_observations):
    """
    Return a sklearn-like classification model.
    """
    train_X, train_y, decisions = Xyd
    if train_X.ndim != 2:
        raise ValueError('train_X must be matrix')
    assert len(train_X) == len(train_y) == len(decisions)
    print "INFO: training binary classifier..."
    svm = MarginBinaryASGD(
        n_features=train_X.shape[1],
        l2_regularization=l2_regularization,
        dtype=train_X.dtype,
        rstate=np.random.RandomState(1234),
        max_observations=max_observations,
        )
    binary_fit(svm, (train_X, train_y, np.asarray(decisions)))
    print "INFO: fitting done!"
    return svm


@scope.define
def svm_decisions(svm, Xyd):
    X, y, d = Xyd
    inc = svm.decision_function(X)
    return d + inc


@scope.define
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


@scope.define
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

    # -- use last fold for held-out data... not use to choose the
    #    ensemble members
    result['loss'] = np.mean([test_margin_mean(ii, rr)
        for ii, rr in enumerate(split_results)])

    result['status'] = hyperopt.STATUS_OK

    return result


@scope.define
def attach_svm(ctrl, svm, name='svm'):
    obj = dict(weights=svm.asgd_weights, bias=svm.asgd_bias)
    ctrl.attachments[name] = cPickle.dumps(obj)
    return ctrl


@scope.define
def run_all(*args):
    return args


def screening_prog(
        n_examples_train,
        n_examples_test,
        n_folds,
        feat_spec,
        decisions,
        svm_l2_regularization,
        svm_max_observations,
        save_svms,
        ctrl):
    """
    Build a pyll graph representing the experiment.
    """
    split_decisions = decisions

    if split_decisions is None:
        split_decisions = np.zeros((n_folds, n_examples_train))
    else:
        # -- experiment may store this as list
        split_decisions = np.asarray(split_decisions)

    Xy = scope.digits_xy(0, n_examples_train)

    # -- build a graph with n_folds paths
    split_results = []
    new_ds = []
    tt_idxs_list = []
    for fold_idx in range(n_folds):
        split_result = {}
        decisions = np.asarray(split_decisions[fold_idx])

        train_idxs, test_idxs = scope.random_train_test_idxs(
                n_examples_train, n_folds, fold_idx)

        train_Xyd = scope.slice_Xyd(Xy, decisions, train_idxs)
        test_Xyd = scope.slice_Xyd(Xy, decisions, test_idxs)

        train_Xyd_n, test_Xyd_n = scope.normalize_Xcols(train_Xyd, test_Xyd)

        train_Xyd_f = scope.features(train_Xyd_n, feat_spec)
        test_Xyd_f = scope.features(test_Xyd_n, feat_spec)

        train_Xyd_fn, test_Xyd_fn = scope.normalize_Xcols(
                train_Xyd_f, test_Xyd_f)

        svm = scope.train_svm(train_Xyd_fn,
                l2_regularization=svm_l2_regularization,
                max_observations=svm_max_observations,
                )

        new_d_train = scope.svm_decisions(svm, train_Xyd_fn)
        new_d_test = scope.svm_decisions(svm, test_Xyd_fn)

        split_result = scope.result_binary_classifier_stats(
                train_Xyd,
                test_Xyd,
                new_d_train,
                new_d_test,
                result=split_result)

        new_ds.append((new_d_train, new_d_test))
        tt_idxs_list.append((train_idxs, test_idxs))

        # -- if we save weights, then do this:
        if save_svms:
            split_results.append(scope.run_all(
                split_result,
                scope.attach_svm(ctrl, svm, 'svm_%i' % fold_idx,
                ))[0])
        else:
            split_results.append(split_result)

    result = scope.combine_results(
            split_results,
            tt_idxs_list,
            new_ds,
            split_decisions,
            Xy[1])

    return result


class BoostableDigits(BaseBandit):
    param_gen = dict(
            n_examples_train=1250,
            n_examples_test=500,
            n_folds=5,
            feat_spec=dict(
                seed=scope.one_of(1, 2, 3, 4, 5),
                n_features=scope.one_of(2, 5, 10),
                scale=scope.uniform(0, 5)
                ),
            svm_l2_regularization=1e-3,
            decisions=None,
            svm_max_observations=20000,
            )

    # XXX: add the loss_variance to result dict
    def evaluate(self, config, ctrl):
        if 'split_decisions' in ctrl.attachments:
            config['split_decisions'] = ctrl.attachments['split_decisions']
        prog = screening_prog(save_svms=False, ctrl=ctrl, **config)
        rval = pyll.rec_eval(prog)
        return rval


    def score_mixture(self, trials, partial_svm):
        """
        partial_svm means fit an svm to each feature set as opposed to one
        joint training of svm.
        """
        n_examples_train = 1250
        n_examples_test = 500
        # -- load up the old training data
        Xy_train = digits_xy(begin=0, N=n_examples_train)
        # -- this data is really held-out, not used for model selection
        Xy_test = digits_xy(begin=n_examples_train, N=500)
        decisions_train = np.zeros(n_examples_train)
        decisions_test = np.zeros(n_examples_test)
        if not partial_svm:
            raise NotImplementedError()

        for ii, trial in enumerate(trials):
            assert trial['spec']['n_examples_train'] == n_examples_train
            train_Xyd_n, test_Xyd_n = normalize_Xcols(
                    (Xy_train[0], Xy_train[1], decisions_train),
                    (Xy_test[0], Xy_test[1], decisions_test))
            train_Xyd_f = features(train_Xyd_n,
                    trial['spec']['feat_spec'])
            test_Xyd_f = features(test_Xyd_n,
                    trial['spec']['feat_spec'])
            train_Xyd_fn, test_Xyd_fn = normalize_Xcols(
                    train_Xyd_f, test_Xyd_f)
            # XXX: match this
            svm = train_svm(train_Xyd_fn,
                    l2_regularization=trial['spec']['svm_l2_regularization'],
                    max_observations=trial['spec']['svm_max_observations'],
                    )

            decisions_train = svm_decisions(svm, train_Xyd_fn)
            decisions_test = svm_decisions(svm, test_Xyd_fn)

            train_err_ii = (np.sign(decisions_train) != Xy_train[1]).mean()
            test_err_ii = (np.sign(decisions_test) != Xy_test[1]).mean()

            print 'score_mixture', ii
            print 'train err:', train_err_ii
            print 'test err:', test_err_ii
        return test_err_ii


