import copy
import cPickle
import logging
logger = logging.getLogger(__name__)
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
from .utils import ImgLoaderResizer, linear_kernel
from .classifier import get_result, train_scikits
from .classifier import normalize as dan_normalize

# -- register symbols in pyll.scope
import toyproblem


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
    def __init__(self, n_features=None):
        # if n-features is given, it will set the number of filters
        # in the top-most layer
        self.comparison = scope.one_of('mult', 'sqrtabsdiff')
        template = dict(
                model=model_params.pyll_param_func(n_features),
                comparison=self.comparison,
                decisions=None,
                # XXX SVM STUFF?
                )
        hyperopt.Bandit.__init__(self, template)

    def evaluate(self, config, ctrl):
        validate_config(config)
        slm = config['model']['slm']
        preproc = config['model']['preproc']
        comparison = config['comparison']
        decisions = config.get('decisions')

        # -- hackity hack (move to hyperopt)
        #    This computes the key for the self.comparison choice
        #    in the idxs/vals for this Bandit.
        algo = hyperopt.Random(self)
        cloned_comp = algo.template_clone_memo[self.comparison]
        cloned_comp_choice = algo.vh.choice_memo[cloned_comp]
        comp_node_id = algo.vh.node_id[cloned_comp_choice]
        # -- now make sure we've got it right...
        ##print algo.vh.node_id.values()
        ##print ctrl.current_trial['misc']['vals'].keys()
        # -- the each key indexes its position in the "one_of" in MainBandit
        val_of_comp = {'mult':0, 'sqrtabsdiff':1}
        if ctrl:
            compval = ctrl.current_trial['misc']['vals'][comp_node_id]
            assert compval == [val_of_comp[comparison]]

        cmp_results = get_performance(slm, decisions, preproc,
                                      comparison=None,
                                      return_multi=True,
                                      ctrl=ctrl)
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
                new_tid, = ctrl.trials.new_trial_ids(1)
                my_trial = ctrl.current_trial
                new_misc = dict(tid=new_tid,
                            idxs=copy.deepcopy(my_trial['misc']['idxs']),
                            vals=copy.deepcopy(my_trial['misc']['vals']))
                for nid, nid_idxs in new_misc['idxs'].items():
                    assert len(nid_idxs) <= 1
                    if nid_idxs:
                        assert nid_idxs[0] == my_trial['tid']
                        nid_idxs[0] = new_tid
                new_config = copy.deepcopy(config)
                new_config['comparison'] = comp

                # -- modify the vals corresponding to the comparison function
                assert len(new_misc['vals'][comp_node_id]) == 1
                assert new_misc['vals'][comp_node_id][0] == val_of_comp[comparison]
                new_misc['vals'][comp_node_id][0] = val_of_comp[comp]
                logger.info('injecting %i from %i (exp_key=%s)' % (
                    new_tid, my_trial['tid'], ctrl.trials._exp_key))
                ctrl.inject_results([new_config], [result], [new_misc],
                                    new_tids=[new_tid])
        assert my_result is not None
        return my_result


class MultiBanditL3(MultiBandit):
    def __init__(self, n_features=None):
        # if n-features is given, it will set the number of filters
        # in the top-most layer
        self.comparison = scope.one_of('mult', 'sqrtabsdiff')
        template = dict(
                model=model_params.pyll_param_func_l3(n_features),
                comparison=self.comparison,
                decisions=None,
                )
        hyperopt.Bandit.__init__(self, template)


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
def verification_pairs(split, subset=None):
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
    if subset is None:
        return lidxs, ridxs, (matches * 2 - 1)
    elif isinstance(subset, int):
        return lidxs[:subset], ridxs[:subset],  (matches[:subset] * 2 - 1)
    else:
        assert all([isinstance(_t, int) for _t in subset])
        return lidxs[subset], ridxs[subset], (matches[subset] * 2 - 1)


@scope.define
def slm_memmap(desc, X, name, basedir=None):
    """
    Return a cache_memmap object representing the features of the entire
    set of images.
    """
    if basedir is None:
        basedir = os.getcwd()
    feat_fn = SLMFunction(desc, X.shape[1:])
    feat = larray.lmap(feat_fn, X)
    rval = larray.cache_memmap(feat, name, basedir=basedir)
    return rval


def get_model_shape(model):
   X = get_images('float32', preproc=model['preproc']) 
   feat_fn = SLMFunction(model['slm'], X.shape[1:])
   return feat_fn.slm.pythor_out_shape
   

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

    # -- mimic how function objects are named
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
        rval = self.fn(lx, rx)
        return rval


@scope.define_info(o_len=2)
def pairs_memmap(pair_labels, X, comparison_name, name, basedir=None):
    """
    pair_labels    - something like comes out of verification_pairs
    X              - feature vectors to be combined
    combination_fn - some lambda X[i], X[j]: features1D
    """
    if basedir is None:
        basedir = os.getcwd()
    lidxs, ridxs, matches = pair_labels
    pf = larray.lmap(
            PairFeaturesFn(X, comparison_name),
            lidxs,
            ridxs)
    pf_cache = larray.cache_memmap(pf, name, basedir=basedir)
    return pf_cache, np.asarray(matches)

@scope.define
def pairs_cleanup(obj):
    """
    Pass in the rval from pairs_memmap to clean up the memmap
    """
    obj[0].delete_files()


def lfw_result_margin(result):
    # -- this function is implemented this way so that
    #    it can be called on results saved before the 'margin' field.
    test_mask = np.asarray(result['is_test'])
    all_labels = np.asarray(result['labels'])
    all_decisions = np.asarray(result['decisions'])
    N, = all_labels.shape
    assert all_decisions.shape == (1, N)
    assert test_mask.shape == (1, N)
    margins = all_labels * all_decisions
    hinges = 1 - np.minimum(margins, 1)
    # -- Compute the mean over test_mask==1 elements
    return (hinges * test_mask).sum() / test_mask.sum()


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
    # -- Note that the margin is not computed here
    result['loss'] = float(1 - result['test_accuracy']/100.)
    dec = np.concatenate([train_decisions, test_decisions])
    dec = dec.reshape((1, len(dec)))
    result['decisions'] = dec.tolist()
    result['labels'] = np.concatenate([train_data[1], test_data[1]]).tolist()
    result['is_test'] = np.column_stack([np.zeros((1, 2200)), np.ones((1, 1000))]).tolist()

    result['margin'] = lfw_result_margin(result)
    
    return result


@scope.define
def svm_decisions_lfw(svm, Xyd):
    X, y, d = Xyd
    inc = svm.decision_function(X)
    rval = d + inc
    return rval


@scope.define
def attach_feature_kernels(train_Xyd, test_Xyd, ctrl, comp):
    X, y, d = train_Xyd
    XX = np.dot(X, X.T)
    packed = []
    for i, xx_i in enumerate(XX):
        packed.extend(xx_i[i:])
    blob = cPickle.dumps(np.asarray(packed), -1)
    key = 'packed_normalized_DevTrain_kernel_%s' % comp
    assert key not in ctrl.attachments
    ctrl.attachments[key] = blob

    K2 = np.dot(X, test_Xyd[0].T)
    blob = cPickle.dumps(K2, -1)
    ctrl.attachments['normalized_DevTrainTest_kernel_%s' % comp] = blob

    return train_Xyd


def screening_program(slm_desc, decisions, comparison, preproc, namebase,
                      image_features=None, ctrl=None):
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

    if 0 and ctrl is not None:
        print >> sys.stderr, "SKIPPING FEATURE KERNEL"
        train_Xyd_n = scope.attach_feature_kernels(train_Xyd_n, test_Xyd_n,
                ctrl, comparison)

    ### TODO: put consts in config, possibly loop over them in MultiBandit
    svm = scope.train_svm(train_Xyd_n,
            l2_regularization=1e-3,
            max_observations=20000)

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
                    return_multi=False, ctrl=None):
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
                    image_features=image_features,
                    ctrl=ctrl)[1][progkey]
        cmp_progs.append([comp, sresult])
    cmp_results = pyll.rec_eval(cmp_progs)
    if return_multi:
        return cmp_results
    else:
        return cmp_results[0][1]


def view2_filename(namebase, split_num):
    return namebase + '_pairs_view2_fold_%d' % split_num


def get_view2_features(slm_desc, preproc, comparison, namebase, basedir,
                       test=None):
    image_features = slm_memmap(
            desc=slm_desc,
            X=get_images('float32', preproc=preproc),
            name=namebase + '_img_feat_view2',
            basedir=basedir)

    # list coersion: `comparison` -> `comp_l`
    if isinstance(comparison, (list, tuple)):
        comp_l = comparison
    else:
        comp_l = [comparison]
    del comparison

    pfs_by_comp = {}
    for split_num in range(10):
        print ('extracting fold %d' % split_num)
        pair_labels = verification_pairs('fold_%d' % split_num, subset=test)
        lidxs, ridxs, matches = pair_labels
        for comparison in comp_l:
            pf = larray.lmap(
                    PairFeaturesFn(image_features, comparison),
                    lidxs,
                    ridxs)
            pf[:] # -- this little guy here computes all the features
            pfs_by_comp.setdefault(comparison, []).append(pf)
    return image_features, pfs_by_comp


def predictions_from_decisions(decisions):
    return np.sign(decisions)


def train_view2(namebases, basedirs, test=None, use_libsvm=False,
                trace_normalize=False, model_kwargs=None):
    """To use use precomputed kernels with libsvm, do
    use_libsvm = {'kernel': 'precomputed'}
    otherwise, use_libsvm = True will use 'linear'
    """
    pair_features = [[larray.cache_memmap(None,
                                   name=view2_filename(nb, snum),
                                   basedir=bdir) for snum in range(10)]
                      for nb, bdir in zip(namebases, basedirs)]

    split_data = [verification_pairs('fold_%d' % split_num, subset=test) for split_num in range(10)]

    train_errs = []
    test_errs = []
    if model_kwargs is None:
        model_kwargs = {}

    for ind in range(10):
        train_inds = [_ind for _ind in range(10) if _ind != ind]
        print ('Constructing stuff for split %d ...' % ind)
        test_X = [pf[ind][:] for pf in pair_features]

        test_y = split_data[ind][2]
        train_X = [np.vstack([pf[_ind][:] for _ind in train_inds])
                             for pf in pair_features]
        train_y = np.concatenate([split_data[_ind][2] for _ind in train_inds])
        train_decisions = np.zeros(len(train_y))
        test_decisions = np.zeros(len(test_y))
        
        #train_Xyd_n, test_Xyd_n = toyproblem.normalize_Xcols(
        #    (np.hstack(train_X), train_y, train_decisions,),
        #    (np.hstack(test_X), test_y, test_decisions,))
        
        normalized = [dan_normalize((t0, t1),
                       trace_normalize=trace_normalize,
                       data=None) for t0, t1 in zip(train_X, test_X)]
        train_X = np.hstack([n[0] for n in normalized])
        test_X = np.hstack([n[1] for n in normalized])
        
        train_Xyd_n = (train_X, train_y, train_decisions)
        test_Xyd_n = (test_X, test_y, test_decisions)
        
        print ('Training split %d ...' % ind)
        if use_libsvm:
            if hasattr(use_libsvm, 'keys'):
                kernel = use_libsvm.get('kernel', 'linear')
            else:
                kernel = 'linear'
            if kernel == 'precomputed':
                (_Xtrain, _ytrain, _dtrain) = train_Xyd_n
                print ('Computing training kernel ...')
                Ktrain = linear_kernel(_Xtrain, _Xtrain, use_theano=True)
                print ('... computed training kernel of shape', Ktrain.shape)
                train_Xyd_n = (Ktrain, _ytrain, _dtrain)
                train_data = (Ktrain, _ytrain, _dtrain)
                print ('Computing testtrain kernel ...')
                (_Xtest, _ytest, _dtest) = test_Xyd_n
                Ktest = linear_kernel(_Xtest, _Xtrain, use_theano=True)
                print ('... computed testtrain kernel of shape', Ktest.shape)
                test_Xyd_n = (Ktest, _ytest, _dtest)

            model_kwargs['kernel'] = kernel
            svm, _ = train_scikits(train_Xyd_n,
                                labelset=[-1, 1],
                                model_type='svm.SVC',
                                model_kwargs=model_kwargs,
                                normalization=False
                                )
        else:
            svm = toyproblem.train_svm(train_Xyd_n,
                l2_regularization=1e-3,
                max_observations=20000)

        #train_decisions = svm_decisions_lfw(svm, train_Xyd_n)
        #test_decisions = svm_decisions_lfw(svm, test_Xyd_n)
        
        #train_predictions = predictions_from_decisions(train_decisions)
        #test_predictions = predictions_from_decisions(test_decisions)

        train_predictions = svm.predict(train_Xyd_n[0])
        test_predictions = svm.predict(test_Xyd_n[0])
        train_err = (train_predictions != train_y).mean()
        test_err = (test_predictions != test_y).mean()

        print 'split %d train err %f' % (ind, train_err)
        print 'split %d test err %f' % (ind, test_err)
        
        train_errs.append(train_err)
        test_errs.append(test_err)

    train_err_mean = np.mean(train_errs)
    print 'train err mean', train_err_mean
    test_err_mean = np.mean(test_errs)
    print 'test err mean', test_err_mean

    return train_err_mean, test_err_mean

