import sys
import time

import numpy as np
import scipy as sp

try:
    import theano
except ImportError:
    pass

from asgd import NaiveBinaryASGD
from asgd.auto_step_size import binary_fit

from .utils import linear_kernel
import utils

#############
###scikits###
#############
try:
    ##use yamins81/feature/multiclass_kwargs branch of sklearn
    ##XXX need to include as git submodule
    from sklearn import svm as sklearn_svm
    from sklearn.multiclass import OneVsRestClassifier
except ImportError:
    print("Can't import scikits stuff")


def train_scikits(train_Xyd,
                  labelset,
                  model_type,
                  model_kwargs=None,
                  fit_kwargs=None,
                  normalization=True,
                  trace_normalize=False,
                  sample_weight_opts=None):

    """
    construct and train a scikits svm model

    model_type = which svm model to use (e.g. "svm.LinearSVC")
    model_kwargs = args sent to classifier initialization
    fit_kwargs = args sent to classifier fit
    use_decisions = whether to use "raw decisions" in weights or not
    normalization = do feature-wise train data normalization
    trace_normalize = do example-wise trace normalization
    """
    if model_kwargs is None:
        model_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    train_features, train_labels, train_decisions = train_Xyd
    assert labelset == [-1, 1] or labelset == range(len(labels)), labels
    assert set(train_labels) == set(labelset)

    #do normalization
    if normalization:
        train_features, train_mean, train_std, trace = normalize(
                            [train_features], trace_normalize=trace_normalize)
    else:
        train_mean = None
        train_std = None
        trace = None

    if sample_weight_opts is not None:
        #NB:  if sample_weight_opts is not None, the classifier better support
        #sample_weights fit argument, e.g. svm.SVC is fine but svm.LinearSVC
        #is NOT.
        assert 'sample_weight' not in fit_kwargs
        use_raw_decisions = sample_weight_opts['use_raw_decisions']
        alpha = sample_weight_opts['alpha']
        sample_weights = sample_weights_from_decisions(
                decisions=train_decisions,
                labels=train_labels,
                labelset=labelset,
                use_raw_decisions=use_raw_decisions,
                alpha=alpha)
        fit_kwargs['sample_weight'] = sample_weights

    model = train_scikits_core(train_features=train_features,
                               train_labels=train_labels,
                               model_type=model_type,
                               labelset=labelset,
                               model_kwargs=model_kwargs,
                               fit_kwargs=fit_kwargs)

    train_data = {'train_mean': train_mean,
                  'train_std': train_std,
                  'trace': trace}

    return model, train_data


def train_scikits_core(train_features,
                     train_labels,
                     model_type,
                     labelset,
                     model_kwargs,
                     fit_kwargs
                     ):
    """
    """
    if model_type.startswith('svm.'):
        ct = model_type.split('.')[-1]
        cls = getattr(sklearn_svm, ct)
    else:
        raise ValueError('Model type %s not recognized' % model_type)
    if labelset == [-1, 1]:
        clf = cls(**model_kwargs)
    else:
        clf = OneVsRestClassifier(cls(**model_kwargs))
    clf.fit(train_features, train_labels, **fit_kwargs)
    return clf


def sample_weights_from_decisions(decisions,
                                  labels,
                                  labelset,
                                  use_raw_decisions,
                                  alpha):

    assert labelset == [-1, 1] or labelset == range(len(labelset))
    assert decisions.shape[0] == labels.shape[0]

    if labelset == [-1, 1]:
        decisions = np.column_stack([-decisions, decisions]) / 2.
        labels = ((1 + labels) / 2).astype(np.int)

    if use_raw_decisions:
        actual = decisions[range(len(labels)), labels]
        decisions_c = decisions.copy()
        decisions_c[range(len(labels)), labels] = -np.inf
        max_c = decisions_c.max(1)
        margins = actual - max_c
    else:
        predictions = decisions.argmax(1)
        margins = 2 * (predictions == labels).astype(np.int) - 1
    weights = np.exp(-alpha * margins)
    weights = weights / weights.sum()

    return weights


#########
##stats##
#########

def get_regression_result(train_actual, test_actual,
        train_predicted, test_predicted):
    test_results = regression_stats(test_actual, test_predicted)
    train_results = regression_stats(train_actual, train_predicted,
            prefix='train')
    test_results.update(train_results)
    return test_results


def regression_stats(actual, predicted, prefix='test'):
    a_mean = actual.mean()
    ap = np.linalg.norm(actual - predicted)
    am = np.linalg.norm(actual - a_mean)
    test_rsquared = 1 - (ap / am) ** 2
    return {prefix + '_rsquared': test_rsquared}


def get_result(train_labels, test_labels,
        train_prediction, test_prediction, labels):
    assert all(l in labels for l in train_prediction), \
            np.unique([l for l in train_prediction if l not in labels])
    assert all(l in labels for l in test_prediction), \
            np.unique([l for l in test_prediction if l not in labels])
    result = {
            'train_errors': (train_labels != train_prediction).tolist(),
            'test_errors': (test_labels != test_prediction).tolist(),
            'train_prediction': train_prediction.tolist(),
            'test_prediction': test_prediction.tolist(),
            'label_set': list(labels),
     }
    stats = multiclass_stats(test_labels, test_prediction, train_labels,
            train_prediction, labels)
    result.update(stats)
    return result


def get_test_result(test_labels, test_prediction, labels, prefix='test'):
    result = {
     prefix + '_errors': (test_labels != test_prediction).tolist(),
     prefix + '_prediction': test_prediction.tolist(),
     'label_set': labels
     }
    stats = multiclass_test_stats(test_labels, test_prediction, labels,
            prefix=prefix)
    result.update(stats)
    return result


def multiclass_stats(test_actual, test_predicted,
        train_actual, train_predicted, labels):
    test_results = multiclass_test_stats(test_actual, test_predicted, labels)
    train_results = multiclass_test_stats(train_actual, train_predicted,
            labels, prefix='train')
    test_results.update(train_results)
    return test_results


def multiclass_test_stats(test_actual, test_predicted, labels, prefix='test'):
    test_accuracy = 100 * np.mean(test_predicted == test_actual)
    test_aps = []
    test_aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        test_prec, test_rec = precision_and_recall(test_actual,
                test_predicted, label)
        test_ap = ap_from_prec_and_rec(test_prec, test_rec)
        test_aps.append(test_ap)
        test_auc = auc_from_prec_and_rec(test_prec, test_rec)
        test_aucs.append(test_auc)
    test_ap = np.array(test_aps).mean()
    test_auc = np.array(test_aucs).mean()
    return {prefix + '_accuracy': float(test_accuracy),
            prefix + '_ap': float(test_ap),
            prefix + '_auc': float(test_auc)}


def precision_and_recall(actual, predicted, cls):
    c = (actual == cls)
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(predicted[si] == cls))
    fp = sp.cumsum(sp.single(predicted[si] != cls))
    rec = tp / sp.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec, rec


def ap_from_prec_and_rec(prec, rec):
    ap = 0
    rng = sp.arange(0, 1.1, 0.1)
    for th in rng:
        parray = prec[rec >= th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def auc_from_prec_and_rec(prec, rec):
    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc


#########
##utils##
#########

def normalize(feats, trace_normalize=False, data=None):
    """Performs normalizations before training on a list of feature array/label
    pairs. first feature array in list is taken by default to be training set
    and norms are computed relative to that one.
    """

    print >> sys.stderr, """"WARNING:, this trace normalization is not the same
    as is in sclas. sclas divides the gram matrix by the trace, whereas this
    normalization (utils.normalize) appears to scale by trace / n_examples
    """

    if data is None:
        train_f = np.asarray(feats[0])
        m = train_f.mean(axis=0)
        s = np.maximum(train_f.std(axis=0), 1e-8)
    else:
        m = data['train_mean']
        s = data['train_std']
    feats = [(np.asarray(f) - m) / s for f in feats]
    if trace_normalize:
        if data is None:
            train_f = feats[0]
            tr = np.maximum(np.sqrt((train_f ** 2).sum(axis=1)).mean(), 1e-8)
        else:
            tr = data['trace']
    else:
        tr = None
    if trace_normalize:
        feats = [f / tr for f in feats]
    feats = tuple(feats)
    return feats + (m, s, tr)


def split_center_normalize(X, y,
        validset_fraction=.2,
        validset_max_examples=5000,
        inplace=False,
        min_std=1e-4,
        batchsize=1):
    n_valid = int(min(
        validset_max_examples,
        validset_fraction * X.shape[0]))

    # -- increase n_valid to a multiple of batchsize
    while n_valid % batchsize:
        n_valid += 1

    n_train = X.shape[0] - n_valid

    # -- decrease n_train to a multiple of batchsize
    while n_train % batchsize:
        n_train -= 1

    if not inplace:
        X = X.copy()

    train_features = X[:n_train]
    valid_features = X[n_train:n_train + n_valid]
    train_labels = y[:n_train]
    valid_labels = y[n_train:n_train + n_valid]

    train_mean, train_std = utils.mean_and_std(X, min_std=min_std)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)

