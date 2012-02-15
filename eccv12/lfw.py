import cPickle
import os

from thoreano.slm import SLMFunction

from skdata import larray
import skdata.utils
import skdata.lfw
import numpy as np

import genson

import model_params
import comparisons

from .bandits import BaseBandit
from .utils import ImgLoaderResizer
from .classifier import get_result

###TODO: move code from .toyproblem to "real" modules
### --> probably mostly to classifier.py
from .toyproblem import normalize_Xcols
from .toyproblem import train_svm
from .toyproblem import run_all


class FG11Bandit(BaseBandit):
    param_gen = dict(
            slm=model_params.fg11_desc,
            comparison=model_params.choice(['mult', 'sqrtabsdiff']),
            preproc={'global_normalize': 0,
                     'size': [200, 200],
                     'crop': [0, 0, 250, 250],
                    }, # XXX: THIS IS NOT TESTED SINCE CHANGING THE CROPPING
            )

    def performance_func(self, config, ctrl):
        slm = config['slm']
        preproc = config['preproc']
        comparison = config['comparison']
        decisions = config['decisions']
        return get_performance(slm, decisions, preproc, comparison, ctrl)


class MainBandit(BaseBandit):
    param_gen = dict(
            model=model_params.main_params,
            comparison=model_params.choice(['mult', 'sqrtabsdiff']),
            )

    def performance_func(self, config, ctrl):
        slm = config['model']['slm']
        preproc = config['model']['preproc']
        comparison = config['comparison']
        decisions = config['decisions']
        return get_performance(slm, decisions, preproc, comparison, ctrl)


class TestBandit(MainBandit):
        param_gen = dict(
                        model=model_params.test_params,
                        comparison=model_params.choice(['mult', 'sqrtabsdiff']),
                    )


@genson.lazy
def fetch_decisions(split, decisions):
    """
    Load the accumulated decisions of models selected for the ensemble,
    for the verification examples of the given split.
    """
    assert split in ['DevTrain', 'DevTest']
    if split == 'DevTrain':
        split_inds = np.arange(0, 2200) 
    else
        split_inds = np.arange(2200, 3200)
    return decisions[split_inds][:, 0]


@genson.lazy
def get_images(dtype, preproc):
    """
    Return a lazy array whose elements are all the images in lfw.

    XXX: Should the images really be returned in greyscale?

    preproc : a dictionary with keys:
        global_normalize - True / False
        size - (height, width)
        crop - (l, t, r, b)

    """

    all_paths = skdata.lfw.Aligned().raw_classification_task()[0]
    rval = larray.lmap(
                ImgLoaderResizer(
                    dtype=dtype,
                    shape=preproc['size'],
                    crop=preproc['crop'],
                    normalize=preproc['global_normalize']),
                all_paths)
    return rval


def _verification_pairs_helper(all_paths, lpaths, rpaths):
    positions = np.argsort(all_paths)
    srtd_paths = np.asarray(all_paths)[positions]
    lidxs = positions[np.searchsorted(srtd_paths, lpaths)]
    ridxs = positions[np.searchsorted(srtd_paths, rpaths)]
    return lidxs, ridxs


@genson.lazy
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


@genson.lazy
def slm_memmap(desc, X, name):
    """
    Return a cache_memmap object representing the features of the entire
    set of images.
    """
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    rval = larray.cache_memmap(feat, name, basedir=os.getcwd())
    return rval


@genson.lazy
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


@genson.lazyinfo(len=2)
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

@genson.lazy
def pairs_cleanup(obj):
    """
    Pass in the rval from pairs_memmap to clean up the memmap
    """
    obj[0].delete_files()


@genson.lazy
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
                             np.sign(train_decisions).astype(np.int),
                             np.sign(test_decisions).astype(np.int),
                             [-1, 1])
    result.update(stats)
    result['loss'] = float(1 - result['test_accuracy']/100.)
    dec = np.concatenate([train_decisions[1], test_decisions[1]])
    dec = dec.reshape((len(dec), 1))
    result['decisions'] = dec.tolist()
    result['labels'] = np.concatenate([train_data[1], test_data[1]]).tolist()
    return result


@genson.lazy
def svm_decisions(svm, Xyd):
    X, y, d = Xyd
    inc = svm.decision_function(X)
    return d + inc


def screening_program(slm_desc, decisions, comparison, preproc, namebase):
    image_features = slm_memmap.lazy(
                desc=slm_desc,
                X=get_images.lazy('float32', preproc=preproc),
                name=namebase + '_img_feat')
    # XXX: check that float32 images lead to correct features

    # XXX: check that it's right that DevTrain has only 2200 verification
    # pairs!? How we supposed to train a good model?!

    # XXX: make sure namebase takes a different value for each process
    #      because it's important that the memmaps don't interfere on disk

    def pairs_dataset(split):
        return pairs_memmap.lazy(
            verification_pairs.lazy(split=split),
            image_features,
            comparison_name=comparison,
            name=namebase + '_pairs_' + split,
            )

    result = {}

    train_X, train_y = pairs_dataset('DevTrain')
    test_X, test_y = pairs_dataset('DevTest')

    ctrl = genson.JSONFunction.KWARGS['ctrl']
    train_d = fetch_decisions.lazy('DevTrain', decisions)
    test_d = fetch_decisions.lazy('DevTest', decisions)

    train_Xyd_n, test_Xyd_n = normalize_Xcols.lazy(
        (train_X, train_y, train_d,),
        (test_X, test_y, test_d,))

    svm = train_svm.lazy(train_Xyd_n, l2_regularization=1e-3)

    new_d_train = svm_decisions.lazy(svm, train_Xyd_n)
    new_d_test = svm_decisions.lazy(svm, test_Xyd_n)

    result = result_binary_classifier_stats.lazy(
            train_Xyd_n,
            test_Xyd_n,
            new_d_train,
            new_d_test,
            result=result)

    result_w_cleanup = run_all.lazy(
        result,
        delete_memmap.lazy(train_X),
        delete_memmap.lazy(test_X),
        delete_memmap.lazy(image_features),
        )[0]

    return result_w_cleanup, locals()


def get_performance(slm, decisions, preproc, comparison, ctrl,
                    namebase=None, progkey='result_w_cleanup'):
    if decisions is None:
        decisions = np.zeros((3200, 1))
    else:
        decisions = np.asarray(decisions)
    assert decisions.shape == (3200, 1)
    if namebase is None:
        namebase = 'memmap_' + str(np.random.randint(1e8))
    prog = screening_program(
            slm_desc=slm,
            preproc=preproc,
            comparison=comparison,
            namebase=namebase,
            decisions=decisions)[1]
    
    prog_fn = genson.JSONFunction(prog[progkey])
    return prog_fn(ctrl=ctrl)

