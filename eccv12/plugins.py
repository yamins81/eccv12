import cPickle


import sys
import os
#import hashlib

from thoreano.slm import SLMFunction

from skdata import larray
import skdata.utils
import skdata.lfw
import numpy as np

from .margin_asgd import MarginBinaryASGD
from .margin_asgd import binary_fit

from hyperopt import genson_helpers
from hyperopt import GensonBandit

#from .classifier import train_only_asgd
from .classifier import get_result

from .fson import register
from .fson import run_all
from .fson import fson_eval
import model_params
import comparisons
from .utils import ImgLoaderResizer

from bandits import BaseBandit

@register(call_w_scope=True)
def fetch_decisions(split, scope):
    """
    Load the accumulated decisions of models selected for the ensemble,
    for the verification examples of the given split.
    """
    return scope['decisions'][split]


@register()
def get_images(dtype='uint8'):
    """
    Return a lazy array whose elements are all the images in lfw.

    XXX: Should the images really be returned in greyscale?

    """
    all_paths = skdata.lfw.Aligned().raw_classification_task()[0]
    rval = larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  # lfw-specific
                    dtype=dtype),
                all_paths)
    return rval


def _verification_pairs_helper(all_paths, lpaths, rpaths):
    positions = np.argsort(all_paths)
    srtd_paths = np.asarray(all_paths)[positions]
    lidxs = positions[np.searchsorted(srtd_paths, lpaths)]
    ridxs = positions[np.searchsorted(srtd_paths, rpaths)]
    return lidxs, ridxs


@register()
def verification_pairs(split):
    """
    Return three integer arrays: lidxs, ridxs, match.

    lidxs: position in the given split of the left image
    ridxs: position in the given split of the right image
    match: 1 if they correspond to the same people, else -1
    """
    dataset = skdata.lfw.Aligned()
    all_paths = dataset.raw_classification_task()[0]
    lpaths, rpaths, matches = dataset.raw_verification_task(split=split)
    lidxs, ridxs = _verification_pairs_helper(all_paths, lpaths, rpaths)
    return lidxs, ridxs, (matches * 2 - 1)


@register()
def slm_memmap(desc, X, name):
    """
    Return a cache_memmap object representing the features of the entire
    set of images.
    """
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    rval = larray.cache_memmap(feat, name, basedir=os.getcwd())
    return rval


@register()
def delete_memmap(obj):
    """
    Delete the files associated with cache_memmap `obj`

    TODO: Think of some mechanism for registering this as a cleanup
    handler corresponding to successful calls to slm_memmap.
    """
    obj.delete_files()


class PairFeaturesFn(object):
    """
    larray.lmap-friendly implementation of the comparison function
    that maps tensor pairs (of image features) to a 1-D flattened
    feature vector.
    """
    __name__  = 'PairFeaturesFn'

    def __init__(self, X, fn_name):
        self.X = X
        self.fn = getattr(comparisons, fn_name)

    def rval_getattr(self, attr, objs):
        if attr == 'shape':
            return (np.prod(self.X.shape[1:]),)
        if attr == 'ndim':
            return 1
        if attr == 'dtype':
            return self.X.dtype
        raise AttributeError()

    def __call__(self, li, ri):
        lx = self.X[li]
        rx = self.X[ri]
        return self.fn(lx, rx)


@register()
def pairs_memmap(pair_labels, X, comparison_name, name):
    """
    pair_labels    - something like comes out of verification_pairs
    X              - feature vectors to be combined
    combination_fn - some lambda X[i], X[j]: features1D
    """
    lidxs, ridxs, matches = pair_labels
    pf = larray.lmap(
            PairFeaturesFn(X, comparison_name),
            lidxs,
            ridxs)
    pf_cache = larray.cache_memmap(pf, name, basedir=os.getcwd())
    return pf_cache, np.asarray(matches)

@register()
def pairs_cleanup(obj):
    """
    Pass in the rval from pairs_memmap to clean up the memmap
    """
    obj[0].delete_files()


@register()
def train_linear_svm_w_decisions(train_data, l2_regularization, decisions):
    """
    Return a sklearn-like classification model.
    """
    train_X, train_y = train_data
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


@register(call_w_scope=True)
def attach_object(obj, name, scope):
    scope['ctrl'].attachments[name] = cPickle.dumps(obj)


@register(call_w_scope=True)
def attach_svmasgd(svm, name, scope):
    obj = dict(weights=svm.asgd_weights, bias=svm.asgd_bias)
    scope['ctrl'].attachments[name] = cPickle.dumps(obj)


@register(call_w_scope=True)
def save_boosting_result(
        svm,
        train_data,
        test_data,
        previous_train_decisions,
        previous_test_decisions,
        scope):

    new_train_decisions = svm.decision_function(train_data[0])
    new_test_decisions = svm.decision_function(test_data[0])
    attach_object(
            previous_train_decisions + new_train_decisions,
            'post_train_decisions',
            scope)
    attach_object(
            previous_test_decisions + new_test_decisions,
            'post_test_decisions',
            scope)

@register(call_w_scope=True)
def result_binary_classifier_stats(
        train_data,
        test_data,
        train_decisions,
        test_decisions,
        scope):
    """
    Compute classification statistics and store them to scope['result']

    The train_decisions / test_decisions are the real-valued confidence
    score whose sign indicates the predicted class for binary
    classification.

    """
    result = scope.setdefault('result', {})
    stats = get_result(train_data[1],
                             test_data[1],
                             np.sign(train_decisions),
                             np.sign(test_decisions),
                             [-1, 1])
    result.update(stats)
    result['loss'] = float(1 - result['test_accuracy']/100.)
    result['train_decisions'] = list(train_decisions)
    result['test_decisions'] = list(test_decisions)


@register()
def svm_decisions(svm,
                  train_verification_dataset,
                  pre_decisions):
    base = pre_decisions
    inc = svm.decision_function(train_verification_dataset[0])
    return base + inc


def screening_program(slm_desc, comparison, namebase):
    image_features = slm_memmap.son(
                desc=slm_desc,
                X=get_images.son('float32'),
                name=namebase + '_img_feat')
    #XXX: check that float32 images lead to correct features

    # XXX: check that it's right that DevTrain has only 2200 verification
    # pairs!? How we supposed to train a good model?!

    # XXX: make sure namebase takes a different value for each process
    #      because it's important that the memmaps don't interfere on disk

    def pairs_dataset(split):
        return pairs_memmap.son(
            verification_pairs.son(split=split),
            image_features,
            comparison_name=comparison,
            name=namebase + '_pairs_' + split,
            )

    train_verification_dataset = pairs_dataset('DevTrain')
    test_verification_dataset = pairs_dataset('DevTest')

    pre_train_decisions = fetch_decisions.son('DevTrain')
    pre_test_decisions = fetch_decisions.son('DevTest')

    svm = train_linear_svm_w_decisions.son(
        train_verification_dataset,
        l2_regularization=1e-3,
        decisions=pre_train_decisions,
        )

    post_train_decisions = svm_decisions.son(
        svm,
        train_verification_dataset,
        pre_train_decisions)

    post_test_decisions = svm_decisions.son(
        svm,
        test_verification_dataset,
        pre_test_decisions)

    rval = run_all.son(
        attach_svmasgd.son(svm, 'svm_asgd'),
        result_binary_classifier_stats.son(
            train_verification_dataset,
            test_verification_dataset,
            post_train_decisions,
            post_test_decisions,
            ),
        pairs_cleanup.son(train_verification_dataset),
        pairs_cleanup.son(test_verification_dataset),
        delete_memmap.son(image_features),
        )
    return locals()


class Bandit(BaseBandit):
    param_gen = dict(
            slm=model_params.fg11_desc,
            comparison='mult',
            )
    def evaluate(self, config, ctrl):
        prog = screening_program(
                slm_desc=config['slm'],
                comparison=config['comparison'],
                namebase='memmap_')['rval']

        scope = dict(
                ctrl=ctrl,
                decisions={},
                )
        # XXX: hard-codes self.train_decisions to be DevTrain - what happens
        # in view 2?
        if self.train_decisions is None:
            scope['decisions']['DevTrain'] = np.zeros(2200)
        else:
            scope['decisions']['DevTrain'] = self.train_decisions

        if self.test_decisions is None:
            scope['decisions']['DevTest'] = np.zeros(1000)
        else:
            scope['decision']['DevTest'] = self.test_decisions

        fson_eval(prog, scope=scope)
        print scope
        return scope['result']

