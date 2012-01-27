import cPickle

#import sys
#import os
#import hashlib

from thoreano.slm import SLMFunction

import skdata.larray
import skdata.utils
import skdata.lfw
import numpy as np
#from thoreano.slm import (TheanoExtractedFeatures, use_memmap)
                          
#from .classifier import train_only_asgd
from .classifier import get_result

from .fson import register
from .fson import run_all

#import comparisons as comp_module

from .utils import ImgLoaderResizer

@register(call_w_scope=True)
def fetch_train_decisions(scope):
    ctrl = scope['ctrl']
    return cPickle.loads(ctrl.get_attachment('train_decisions'))


@register(call_w_scope=True)
def fetch_test_decisions(scope):
    ctrl = scope['ctrl']
    return cPickle.loads(ctrl.get_attachment('test_decisions'))


@register()
def get_images(dtype='uint8'):
    all_paths = skdata.lfw.Aligned().img_classification_task()[0]
    rval = skdata.larray.lmap(
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
    Return three numpy arrays: lidxs, ridxs, match.

    lidxs: position in the given split of the left image
    ridxs: position in the given split of the right image
    match: 0 if they correspond to different people, else 1
    """
    dataset = skdata.lfw.Aligned()
    all_paths = dataset.raw_classification_task()[0]
    lpaths, rpaths, matches = dataset.raw_verification_task(split=split)
    lidxs, ridxs = _verification_pairs_helper(all_paths, lpaths, rpaths)
    return lidxs, ridxs, matches


@register()
def slm_memmap(desc, X):
    feat_fn = SLMFunction(desc, X.shape[1:])
    rval = skdata.larray.lmap(feat_fn, X)
    return rval


@register()
def load_comparison(comparison):
    raise NotImplementedError()


@register()
def pairs_memmap(pairs, X, combination_fn):
    """
    pairs - something like comes out of verification_pairs
    X - feature vectors to be combined
    combination_fn - some lambda X[i], X[j]: features
    """
    raise NotImplementedError()


@register()
def train_linear_svm_w_margins(train_data, l2_regularization, margins):
    """
    Return a sklearn-like classification model.
    """
    raise NotImplementedError()


@register(call_w_scope=True)
def attach_object(obj, name, scope):
    scope['ctrl'].set_attachment(name, cPickle.dumps(obj))


@register(call_w_scope=True)
def attach_svmasgd(svm, name, scope):
    obj = dict(weights=svm.asgd_weights, bias=svm.asgd_bias)
    return attach_object(obj, name, scope) 


@register(call_w_scope=True)
def save_boosting_results(
        svm,
        train_data,
        test_data,
        previous_train_decisions,
        previous_test_decisions,
        scope):
    ctrl = scope['ctrl']

    new_train_decisions = svm.decision_function(train_data[0])
    new_test_decisions = svm.decision_function(test_data[0])
    ctrl.set_attachment('post_train_decisions',
                        cPickle.dumps(
                            previous_train_decisions + new_train_decisions))
    ctrl.set_attachment('post_test_decisions',
                        cPickle.dumps(
                            previous_test_decisions + new_test_decisions))

@register(call_w_scope=True)
def results_binary_classifier_stats(
        train_data, 
        test_data,
        train_decisions,
        test_decisions,
        scope):
    result = scope['result']
    stats = get_result(train_data[1],
                             test_data[1],
                             np.sign(train_decisions),
                             np.sign(test_decisions),
                             [-1, 1])
    result.update(stats)
    result['loss'] = float(1 - result['test_accuracy']/100.)
    return result


@register()
def svm_decisions(svm,
                  train_verification_dataset,
                  pre_decisions):
    raise NotImplementedError()


def screening_program(slm_desc, comparison):
    def pairs_dataset(split):
        return pairs_memmap.son(
            pairs=verification_pairs.son(split=split),
            X=slm_memmap.son(
                desc=slm_desc,
                X=get_images.son()),
            comparison_fn=load_comparison.son(comparison),
            )

    train_verification_dataset = pairs_dataset('DevTrain')
    test_verification_dataset = pairs_dataset('DevTest')

    pre_train_decisions = fetch_train_decisions.son()
    pre_test_decisions = fetch_test_decisions.son()

    svm = train_linear_svm_w_margins.son(
        train_verification_dataset,
        l2_regularization=1e-3,
        margins = pre_train_decisions,
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
        attach_object.son(post_train_decisions, 'post_train_decisions'),
        attach_object.son(post_test_decisions, 'post_test_decisions'),
        results_binary_classifier_stats.son(
            train_verification_dataset,
            test_verification_dataset,
            post_train_decisions,
            post_test_decisions,
            ),
        )

    return rval

