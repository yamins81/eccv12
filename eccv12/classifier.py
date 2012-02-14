import numpy as np
import scipy as sp


#########
##stats##
#########

def get_regression_result(train_actual, test_actual, train_predicted, test_predicted):
    test_results = regression_stats(test_actual, test_predicted)
    train_results = regression_stats(train_actual, train_predicted, prefix='train')
    test_results.update(train_results)
    return test_results

def regression_stats(actual, predicted, prefix='test'):
    a_mean = actual.mean()
    test_rsquared = 1 - np.linalg.norm(actual - predicted)**2 / np.linalg.norm(actual-a_mean)**2
    return {prefix+'_rsquared' : test_rsquared}


def get_result(train_labels, test_labels, train_prediction, test_prediction, labels):
    assert all(l in labels for l in train_prediction), np.unique([l for l in train_prediction if l not in labels])
    assert all(l in labels for l in test_prediction), np.unique([l for l in test_prediction if l not in labels])
    result = {'train_errors': (train_labels != train_prediction).tolist(),
     'test_errors': (test_labels != test_prediction).tolist(),
     'train_prediction': train_prediction.tolist(),
     'test_prediction' : test_prediction.tolist(),
     'label_set' : list(labels),
     }
    stats = multiclass_stats(test_labels, test_prediction, train_labels, train_prediction, labels)
    result.update(stats)
    return result


def get_test_result(test_labels, test_prediction, labels, prefix='test'):
    result = {
     prefix + '_errors': (test_labels != test_prediction).tolist(),
     prefix + '_prediction' : test_prediction.tolist(),
     'label_set': labels
     }
    stats = multiclass_test_stats(test_labels, test_prediction, labels, prefix=prefix)
    result.update(stats)
    return result


def multiclass_stats(test_actual, test_predicted, train_actual, train_predicted,labels):
    test_results = multiclass_test_stats(test_actual, test_predicted, labels)
    train_results = multiclass_test_stats(train_actual, train_predicted, labels, prefix='train')
    test_results.update(train_results)
    return test_results


def multiclass_test_stats(test_actual, test_predicted, labels, prefix='test'):
    test_accuracy = float(100*(test_predicted == test_actual).sum() / float(len(test_predicted)))
    test_aps = []
    test_aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        test_prec,test_rec = precision_and_recall(test_actual,test_predicted,label)
        test_ap = ap_from_prec_and_rec(test_prec,test_rec)
        test_aps.append(test_ap)
        test_auc = auc_from_prec_and_rec(test_prec,test_rec)
        test_aucs.append(test_auc)
    test_ap = np.array(test_aps).mean()
    test_auc = np.array(test_aucs).mean()
    return {prefix+'_accuracy' : float(test_accuracy),
            prefix+'_ap' : float(test_ap),
            prefix+'_auc' : float(test_auc)}


def precision_and_recall(actual,predicted,cls):
    c = (actual == cls)
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(predicted[si] == cls))
    fp = sp.cumsum(sp.single(predicted[si] != cls))
    rec = tp /sp.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec,rec


def ap_from_prec_and_rec(prec,rec):
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def auc_from_prec_and_rec(prec,rec):
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

    if data is None:
        train_f = feats[0]
        m = train_f.mean(axis=0)
        s = np.maximum(train_f.std(axis=0), 1e-8)
    else:
        m = data['train_mean']
        s = data['train_std']
    feats = [(f - m) / s for f in feats]
    if trace_normalize:
        if data is None:
            train_f = feats[0]
            tr = np.maximum(np.sqrt((train_f**2).sum(axis=1)).mean(), 1e-8)
        else:
            tr = data['trace']
    else:
        tr = None
    if trace_normalize:
        feats = [f / tr for f in feats]
    feats = tuple(feats)
    return feats + (m, s, tr)


def mean_and_std(X, min_std):
    # -- this loop is more memory efficient than numpy
    #    but not as numerically accurate as possible
    m = np.zeros(X.shape[1], dtype='float64')
    msq = np.zeros(X.shape[1], dtype='float64')
    for i in xrange(X.shape[0]):
        alpha = 1.0 / (i + 1)
        v = X[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=X.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(X.dtype)
    return train_mean, train_std


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

    train_mean, train_std = mean_and_std(X, min_std=min_std)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)


