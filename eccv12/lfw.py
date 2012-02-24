import copy
import cPickle
import os

from thoreano.slm import SLMFunction
import hyperopt

from skdata import larray
import skdata.utils
import skdata.lfw
import numpy as np

import pyll
from pyll import scope

import model_params
import comparisons

from .bandits import BaseBandit, validate_config, validate_result
from .utils import ImgLoaderResizer
from .classifier import get_result


class FG11Bandit(BaseBandit):
    param_gen = dict(
            slm=model_params.fg11_desc,
            comparison=model_params.choice(['mult', 'sqrtabsdiff']),
            preproc={'global_normalize': 0,
                     'size': [200, 200],
                     'crop': [0, 0, 250, 250],
                    },
            )

    def performance_func(self, config, ctrl):
        slm = config['slm']
        preproc = config['preproc']
        comparison = config['comparison']
        decisions = config.get('decisions')
        return get_performance(slm, decisions, preproc, comparison)


class MainBandit(BaseBandit):
    comparison = scope.one_of('mult', 'sqrtabsdiff')
    param_gen = dict(
            model=model_params.pyll_param_func(),
            comparison=comparison,
            decisions=None,
            # XXX SVM STUFF?
            )

    def performance_func(self, config, ctrl):
        slm = config['model']['slm']
        preproc = config['model']['preproc']
        comparison = config['comparison']
        decisions = config.get('decisions')
        return get_performance(slm, decisions, preproc, comparison)

class MultiBandit(hyperopt.Bandit):
    def __init__(self):
        hyperopt.Bandit.__init__(self, copy.deepcopy(MainBandit.param_gen))

    def evaluate(self, config, ctrl):
        validate_config(config)
        slm = config['model']['slm']
        preproc = config['model']['preproc']
        comparison = config['comparison']
        decisions = config.get('decisions')

        # -- hackity hack (move to hyperopt)
        #    This computes the key for the MainBandit.comparison choice
        #    in the idxs/vals for this Bandit.
        algo = hyperopt.Random(self)
        cloned_comp = algo.template_clone_memo[self.comparison]
        comp_node_id = algo.vh.node_id[cloned_comp]
        # -- now make sure we've got it right...
        compval = ctrl.current_trial['misc']['vals'][comp_node_id]
        # -- the each key indexes its position in the "one_of" in MainBandit
        val_of_comp = {'mult':0, 'sqrtabsdiff':1}
        assert compval == [val_of_comp[comparison]]

        cmp_results = get_performance(slm, decisions, preproc,
                                      comparison=None,
                                      return_multi=True)
        my_result = None
        # XXX This logic is tightly coupled to get_performance
        #     in order to not break tests and other code using get_performance
        #     however, before adding SVM parameters and other things to this
        #     loop, consider reorganizing to simplify.
        for comp, result in cmp_results:
            result.setdefault('status', hyperopt.STATUS_OK)
            if result['status'] == hyperopt.STATUS_OK:
                validate_result(result, config)
            if comp == comparison:
                # -- store this to the db the normal way
                my_result = result
            else:
                # -- inject this to the trials db directly
                my_trial = ctrl.current_trial
                misc = dict(idxs=copy.deepcopy(my_trial['misc']['idxs']),
                            vals=copy.deepcopy(my_trial['misc']['vals']))
                assert len(misc['vals'][comp_node_id][0]) == 0
                misc['vals'][comp_node_id][0] == val_of_comp[comp]
                ctrl.inject_results([config], [result], [misc])
        assert my_result is not None
        return my_result


class TestBandit(MainBandit):
        param_gen = dict(
                        model=model_params.test_params,
                        comparison=model_params.choice(['mult', 'sqrtabsdiff']),
                    )


#XXX currently the weird name (e.g. '_lfw') here is handle the namespace problem in pyll
@scope.define
def get_decisions_lfw(split, decisions):
    """
    Load the accumulated decisions of models selected for the ensemble,
    for the verification examples of the given split.
    """
    assert split in ['DevTrain', 'DevTest']
    if split == 'DevTrain':
        split_inds = np.arange(0, 2200) 
    else:
        split_inds = np.arange(2200, 3200)
    return decisions[0][split_inds]


@scope.define
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


@scope.define
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


@scope.define
def slm_memmap(desc, X, name):
    """
    Return a cache_memmap object representing the features of the entire
    set of images.
    """
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    rval = larray.cache_memmap(feat, name, basedir=os.getcwd())
    return rval


@scope.define
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


@scope.define_info(o_len=2)
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

@scope.define
def pairs_cleanup(obj):
    """
    Pass in the rval from pairs_memmap to clean up the memmap
    """
    obj[0].delete_files()


@scope.define
def result_binary_classifier_stats_lfw(
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
    dec = np.concatenate([train_decisions, test_decisions])
    dec = dec.reshape((1, len(dec)))
    result['decisions'] = dec.tolist()
    result['labels'] = np.concatenate([train_data[1], test_data[1]]).tolist()
    result['is_test'] = np.column_stack([np.zeros((1, 2200)), np.ones((1, 1000))]).tolist()
    
    return result


@scope.define
def svm_decisions_lfw(svm, Xyd):
    X, y, d = Xyd
    inc = svm.decision_function(X)
    return d + inc


def screening_program(slm_desc, decisions, comparison, preproc, namebase,
                      image_features=None):
    if image_features is None:
        image_features = scope.slm_memmap(
                desc=slm_desc,
                X=scope.get_images('float32', preproc=preproc),
                name=namebase + '_img_feat')
    # XXX: check that float32 images lead to correct features

    # XXX: make sure namebase takes a different value for each process
    #      because it's important that the memmaps don't interfere on disk

    def pairs_dataset(split):
        return scope.pairs_memmap(
            scope.verification_pairs(split=split),
            image_features,
            comparison_name=comparison,
            name=namebase + '_pairs_' + split,
            )

    result = {}

    train_X, train_y = pairs_dataset('DevTrain')
    test_X, test_y = pairs_dataset('DevTest')

    train_d = scope.get_decisions_lfw('DevTrain', decisions)
    test_d = scope.get_decisions_lfw('DevTest', decisions)

    train_Xyd_n, test_Xyd_n = scope.normalize_Xcols(
        (train_X, train_y, train_d,),
        (test_X, test_y, test_d,))

    svm = scope.train_svm(train_Xyd_n, l2_regularization=1e-3, max_observations=20000)

    new_d_train = scope.svm_decisions_lfw(svm, train_Xyd_n)
    new_d_test = scope.svm_decisions_lfw(svm, test_Xyd_n)

    result = scope.result_binary_classifier_stats_lfw(
            train_Xyd_n,
            test_Xyd_n,
            new_d_train,
            new_d_test,
            result=result)

    result_w_cleanup = scope.run_all(
        result,
        scope.delete_memmap(train_X),
        scope.delete_memmap(test_X),
        scope.delete_memmap(image_features),
        )[0]

    return result_w_cleanup, locals()


def get_performance(slm, decisions, preproc, comparison,
                    namebase=None, progkey='result_w_cleanup',
                    return_multi=False):
    if decisions is None:
        decisions = np.zeros((1, 3200))
    else:
        decisions = np.asarray(decisions)
    assert decisions.shape == (1, 3200)
    if namebase is None:
        namebase = 'memmap_' + str(np.random.randint(1e8))
    image_features = scope.slm_memmap(
            desc=slm,
            X=scope.get_images('float32', preproc=preproc),
            name=namebase + '_img_feat')
    if return_multi:
        comps = ['mult', 'sqrtabsdiff']
    else:
        comps = [comparison]
    cmp_progs = []
    for comp in comps:
        sresult = screening_program(
                    slm_desc=slm,
                    preproc=preproc,
                    comparison=comp,
                    namebase=namebase,
                    decisions=decisions,
                    image_features=image_features)[1][progkey]
        cmp_progs.append((comp, sresult))
    cmp_results = pyll.rec_eval(cmp_progs)
    if return_multi:
        return cmp_results
    else:
        return cmp_results[0][1]

