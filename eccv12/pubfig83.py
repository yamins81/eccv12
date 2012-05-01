import copy
import cPickle
import logging

logger = logging.getLogger(__name__)
import os
import os.path as path

import pyll
from pyll import scope

import hyperopt
import hyperopt.base as base
from hyperopt.utils import fast_isin 
from hyperopt.pyll_utils import hp_choice

from skdata import larray
import skdata.utils
import skdata.pubfig83
import numpy as np

from .utils import ImgLoaderResizer, linear_kernel
from .classifier import get_result, train_scikits, normalize

import pyll_slm

pyll.scope.import_(globals(),
    'partial',
    'callpipe1',
    'asarray',
    'pyll_theano_batched_lmap',
    )
    

@scope.define
def pubfig83_dataset(ntrain, nvalidate, ntest, nfolds):
    return skdata.pubfig83.PubFig83(
                                   ntrain=ntrain, 
                                   nvalidate=nvalidate,
                                   ntest=ntest,
                                   nfolds=nfolds)

@scope.define_info(o_len=2)                                 
def get_dataset_splits(dataset):
    splits = dataset.classification_splits
    can_use = list(set(list(itertools.chain(*[s[k] for s in splits if 
                                  s.startswith(('Train', 'Validate'))]))))
    can_use.sort()
    return splits, can_use

#####pre and post process######
@scope.define_info(o_len=2)
def get_decisions(ctrl, use_decisions):
    """puts decisions back into config.
    NB: This function modifies 'config' argument
    """
    continues = ctrl.current_job['misc']['continues']
    if use_decisions:
        if continues is not None:
            trials = ctrl.trials
            trials.refresh()
            trial = [_t for _t in trials if _t['tid'] == continues]
            assert len(trial) == 1
            trial = trial[0]
            trial_attachments = trials.trial_attachments(trial=trial)
            decisions = trial_attachments['decisions']
            decisions = cPickle.loads(decisions)
        else:
            decisions = None
    else:
        assert 'predictions' not in config
        if continues is not None:
            trials = ctrl.trials
            trials.refresh()
            trial = [_t for _t in trials if _t['tid'] == continues]
            assert len(trial) == 1
            trial = trial[0]
            trial_attachments = trials.trial_attachments(trial=trial)
            predictions = trial_attachments['predictions']
            decisions = cPickle.loads(predictions)
        else:
            decisions = None
    
    return decisions, predictions


@scope.define
def validate_pubfig83_decisions(decisions, predictions, ndim, use_decisions, 
                ntrain, nvalidate, nfolds):
    """
    Checks that config has the right format
    """
    if use_decisions:
        if decisions is not None:
            decisions = np.asarray(decisions)
            assert decisions.ndim == ndim
    else:
        if predictions is not None:
            predictions = np.asarray(predictions)
            assert predictions.ndim == 2
            
    if decisions is not None:
        decisions = np.asarray(decisions)
        assert decisions.shape == (nfolds, 83 * (ntrain + nvalidate), 83)
    if predictions is not None:
        predictions = np.asarrray(predictions)
        assert predictions.shape == (nfolds, 83 * (ntrain + nvalidate))
    

@scope.define
def validate_pubfig83_result(result, use_decisions, ndim, orig_decisions, orig_predictions):
    """this checks that the result has the right format for being used by both
    the boosting bandit algo as well as the adaboost mixture 
    decisions are a matrix of decisions of shape (num_splits, num_examples)
    labels is a 1-d array of binary lables
    """
    if use_decisions:
        decisions = np.asarray(result['decisions'])
        assert decisions.ndim == ndim
        labels = np.asarray(result['labels'])
        assert labels.ndim == 1
        assert decisions.shape[1] == len(labels)
        is_test = np.asarray(result['is_test'])
        assert is_test.shape == decisions.shape[:2]
        if orig_decisions is not None:
            assert decisions.shape == np.array(orig_deicisions).shape
    else:
        predictions = np.asarray(result['predictions'])
        assert predictions.ndim == 1
        labels = np.asarray(result['labels'])
        assert labels.ndim == 1
        assert predictions.shape[1] == labels.shape
        is_test = np.asarray(result['is_test'])
        assert is_test.shape == predictions.shape
        if orig_predictions is not None:
            assert predictions.shape == np.array(orig_predictions).shape


@scope.define
def postprocess_pubfig83_result(result, config):
    """removes big stuff from the attachments and puts them in the filesystem
    NB:  This function modifies 'result' argument
    """
    attachments = result['attachments'] = {}
    if config['use_decisions']:
        attachments['decisions'] = cPickle.dumps(result.pop('decisions'))
    else:
        attachmets['predictions'] = cPickle.dumps(result.pop('predictions'))
    attachments['labels'] = cPickle.dumps(result.pop('labels'))
    attachments['is_test'] = cPickle.dumps(result.pop('is_test'))


############
@scope.define
def get_pubfig83_images(dataset, dtype, preproc):
    """
    Return a lazy array whose elements are all the images in lfw.

    XXX: Should the images really be returned in greyscale?

    preproc : a dictionary with keys:
        global_normalize - True / False
        size - (height, width)
        crop - (l, t, r, b)

    """
    
    all_paths = dataset.raw_classification_task()[0]
    rval = larray.lmap(
                ImgLoaderResizer(
                    inshape=(100, 100),
                    dtype=dtype,
                    shape=preproc['size'],
                    crop=preproc['crop'],
                    normalize=preproc['global_normalize']),
                all_paths)
    return rval


def slice_Xydp(X, y, d, p, idxs):
    """
    """
    X = X[idxs]
    y = y[idxs]
    
    xshp = X.shape
    X = X.reshape((xshp[0], xshp[1]*xshp[2]*xshp[3]))
    
    assert X.ndim == 2, X.shape
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    if d is not None:
        d = d[idxs]
        assert d.ndim == 2, d.shape
        assert y.shape[0] == d.shape[0]
    if p is not None:
        p = d[idxs]
        assert d.ndim == 1, p.shape
        assert y.shape == p.shape
    return X, y, d, p

   
def decision_function(svm, X):
    if hasattr(svm, 'coef_'):
        Y = np.dot(X, svm.coef_.T) + self.intercept_.T
    else:
        estimators = svm.estimators_
        Y = np.array([e.decision_function(X) for e in estimators])
        Y = Y[:, :, 0].T
    return Y
   

def svm_decisions(svm, Xydp):
    X, y, d, p = Xyd
    inc = decision_function(svm, X)
    if d is not None:
        return d + inc
    else:
        return inc


def predictions_from_decisions(X, labelset):
    assert X.ndim == 2 
    assert X.shape[1] == len(labelset)
    return X.argmax(1)
   

def result_classifier_stats(
        train_data,
        test_data,
        train_decisions,
        test_decisions,
        labelset):
    """
    The train_decisions / test_decisions are the real-valued confidence
    score whose sign indicates the predicted class for binary
    classification.

    """
    result = {}
    labelset = list(labelset)
    train_predictions = predictions_from_decisions(train_decisions, labelset)
    test_predictions = predictions_from_decisions(test_decisions, labelset)
    stats = get_result(train_data[1],
                         test_data[1],
                         train_predictions,
                         test_predictions,
                         labelset)
    result.update(stats)
    result['loss'] = float(1 - result['test_accuracy']/100.)
    return result


def combine_results(split_results, tt_idxs_list, new_ds, 
                    decisions, y, labelset, dataset, use_decisions):
    """
    Result has
        loss - scalar
        splits - sub-result dicts
        decisions - for next boosting round
    """
    result = dict(splits=split_results)

    # -- calculate the decisions
    if decisions is not None:
        new_decisions = np.zeros_like(decisions)
    else:
        nfolds = dataset.nfolds
        ntrain = dataset.ntrain
        nvalidate = dataset.nvalidate
        new_decisions = np.zeros((nfolds, 83 * (ntrain + nvalidate), 83))
    is_test = []
    for fold_idx, rr in enumerate(split_results):
        new_d_train, new_d_test = new_ds[fold_idx]
        train_idxs, test_idxs = tt_idxs_list[fold_idx]
        new_decisions[fold_idx][train_idxs] = new_d_train
        new_decisions[fold_idx][test_idxs] = new_d_test
        test = np.zeros((len(new_decisions[fold_idx]),))
        test[test_idxs] = 1
        is_test.append(test)
    
    result['is_test'] = np.array(is_test).tolist()  
    if use_decisions:
        result['decisions'] = [dd.tolist() for dd in new_decisions]
    else:
        result['predictions'] = [predictions_from_decisions(dd, labelset).tolist() for dd in new_decisions]
    result['labels'] = y.tolist()
    result['loss'] = float(np.mean([r['loss'] for r in split_results])) 
    return result

              
@base.as_bandit()
def pubfig83_bandit(ntrain,
                    nvalidate,
                    ntest,
                    nfolds,
                    use_decisions,
                    use_raw_decisions,
                    npatches,
                    n_imgs_for_patches,
                    max_n_features,
                    max_layer_sizes,
                    pipeline_timeout,
                    batchsize,
                    memmap_name,
                    namebase=None):
    ctrl = hyperopt.Bandit.pyll_ctrl
    decision_dims = 3                                       
    decisions, predictions = scope.get_decisions(ctrl)    
    scope.validate_pubfig83_decisions(decisions, predictions, 
            decision_dims, use_decisions, ntrain, nvalidate, nfolds)

    if namebase is None:
        namebase = 'memmap_' + str(np.random.randint(1e8))    
        
    dataset = scope.pubfig83_dataset(ntrain, nvalidate, ntest, nfolds)
    splits, can_use = scope.get_dataset_splits(dataset)
    
    images = scope.get_pubfig83_images(dataset, dtype='float32',
                                       preproc=hp_choice('preproc',
                                    [
                                        {
                                        'global_normalize': 0,
                                        'size': [1, 200, 200],
                                        'crop': [0, 0, 250, 250],
                                        },
                                    ]))
                                         
    pipeline = choose_pipeline(
            Xcm=scope.asarray(images[can_use[:n_imgs_for_patches]]),
            n_patches=n_patches,
            batchsize=batchsize,
            max_n_features=max_n_features,
            max_layer_sizes=max_layer_sizes,
            time_limit=pipeline_timeout,
            )
                
    image_features = scope.larray_cache_memmap(
            pyll_theano_batched_lmap(
                partial(callpipe1, pipeline['pipe']),
                images,
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=max_n_features,
                ), memmap_name)                
                
    result = scope.traintest_stuff(decisions, predictions, splits, dataset,
                   image_features, use_raw_decisions)
    
    scope.validate_pubfig83_result(result, use_decisions, decision_dims, 
                                   decisions, predictions)
    scope.postprocess_pubfig83_result(result) 
    
    return result


@scope.define
def traintest_stuff(decisions, predictions, splits, dataset, image_features, 
                    use_raw_decisions):


    _, all_labels, _1 = dataset.raw_classification_task()
    labelset = np.unique(all_labels).tolist()
    
    split_results = []
    new_ds = []
    tt_idxs_list = []
    idxs0 = np.concatenate([splits['Train0'], splits['Validate0']])
    idxs0.sort()
    
    image_features = image_features[idxs0]
    all_labels = all_labels[idxs0]
    
    nfolds = len(splits)
    for fold_idx in range(nfolds):
        if decisions is not None:
            split_decisions = np.asarray(decisions[fold_idx])
        else:
            split_decisions = None
        if predictions is not None:
            split_predictions = np.assay(predictions[fold_idx])
        else:
            split_predictions = None
          
        train_idxs = splits['Train%d' % fold_idx] 
        test_idxs = splits['Validate%d' % fold_idx] 
        idxs = np.concatenate([train_idxs, test_idxs])
        idxs.sort()
        assert (idxs0 == idxs).all()
        
        train_idxs0 = np.searchsorted(idxs0, train_idxs)
        test_idxs0 = np.searchsorted(idxs0, test_idxs)
        
        train_X, train_y, train_d, train_p = slice_Xydp(image_features,
                                              all_labels, 
                                              split_decisions,
                                              split_predictions,
                                              train_idxs0)
        test_X, test_y, test_d, test_p = slice_Xydp(image_features,
                                           all_labels, 
                                           split_decisions,
                                           split_predictions,
                                           test_idxs0)                                            
        print ('Finished slicing fold %d' % fold_idx)
        train_X, test_X, _m, _s, _tr = normalize((train_X, test_X))
        
        print('Computing kernels fold %d' % fold_idx)
        train_K = linear_kernel(train_X, train_X, use_theano=True)
        print('Computed fold %d train kernel of size' % fold_idx, train_K.shape)
        test_K = linear_kernel(test_X, train_X, use_theano=True)
        print('Computed fold %d test kernel of size' % fold_idx, test_K.shape)

        train_Kydp = (train_K, train_y, train_d, train_p)
        test_Kydp = (test_K, test_y, test_d, test_p)

        print('Training fold %d' % fold_idx)
        if test_d is not None:
            C = len(labelset)
            er = (test_d.argmax(1) != test_y).astype(int).sum() / float(len(test_y))
            alpha = 0.5 * np.log((1 - er) / er * (C - 1))
        elif test_p is not None:
            C = len(labelset)
            er = (test_p != test_y).astype(int).sum() / float(len(test_y))
            alpha = 0.5 * np.log((1 - er) / er * (C - 1))
        else:
            alpha = None
        svm, train_data = train_scikits(train_Kydp,
                                labelset=labelset,
                                model_type='svm.SVC',
                                model_kwargs={'kernel': 'precomputed'},
                                normalization=False,
                                trace_normalize=False,
                                sample_weight_opts={'use_raw_decisions': use_raw_decisions,
                                                    'alpha': alpha }
                                ) 
        #XXX might be useful to save train_data in attachment

        new_d_train = svm_decisions(svm, train_Kydp)
        new_d_test = svm_decisions(svm, test_Kydp)

        split_result = result_classifier_stats(
                train_Kydp,
                test_Kydp,
                new_d_train,
                new_d_test,
                labelset=labelset)

        new_ds.append((new_d_train, new_d_test))
        tt_idxs_list.append((train_idxs0, test_idxs0))

        split_results.append(split_result)

    result = combine_results(
            split_results,
            tt_idxs_list,
            new_ds,
            decisions,
            all_labels,
            labelset,
            dataset,
            use_decisions
            )

    s = image_features.shape
    result['num_features'] = int(np.prod(s[1:]))
    result['feature_shape'] = list(s)

    result.setdefault('status', hyperopt.STATUS_OK)
    return result


####
####
####view 2 stuff

def Ktrain_name(basedir, namebase, fold):
    return os.path.join(basedir, namebase + '_view2_fold_%i_Ktrain.npy' % fold)

def Ktest_name(basedir, namebase, fold):
    return os.path.join(basedir, namebase + '_view2_fold_%i_Ktest.npy' % fold)


def get_view2_splits(dataset, ntrain, ntest, nfolds):
    rng = np.random.RandomState(0)
    labels = np.unique(dataset.names)
    train_splits = dataset.classification_splits
    splits = {}    
    for label in labels:
        to_consider = (dataset.names == label).astype(int)
        all_train = np.concatenate([train_splits['Train0'], train_splits['Validate0']])
        to_consider[all_train] == 0
        to_consider = to_consider.nonzero()[0]
        assert len(to_consider) >= ntrain + ntest
        p = rng.permutation(len(to_consider))
        to_consider = to_consider[p[:ntrain + ntest]]
        for ind in range(nfolds):
            p = rng.permutation(len(to_consider))
            if 'Train%d' % ind not in splits:
                splits['Train%d' % ind] = []
            splits['Train%d' % ind].extend(to_consider[p[:ntrain]])
            if 'Validate%d' % ind not in splits:
                splits['Validate%d' % ind] = []         
            splits['Validate%d' % ind].extend(to_consider[p[ntrain:ntrain+ntest]])
        
    return splits
                


def get_view2_kernels(dataset, 
                      slm_desc,
                      preproc,
                      namebase,
                      outputdir):
    
    all_paths, all_labels, _ = dataset.raw_classification_task()
    labelset = np.unique(all_labels).tolist()

    if path.exists('/scratch_local/dyamins'):
        basedir = '/scratch_local/dyamins'
    else:
        basedir = None
    Images = get_images(dataset=dataset, 
                        dtype='float32',
                        preproc=preproc)
    image_features = slm_memmap(
            desc=slm_desc,
            X=Images,
            name=namebase + '_view2_img_feat',
            basedir=basedir)

    
    
    splits = dataset.splits
    
    idxs0 = np.concatenate([splits['Train0'], splits['Validate0'], splits['Test']])
    idxs0.sort()
    image_features = image_features[idxs0]
    all_labels = all_labels[idxs0]
    
    fold_idx = 0
    train_file_name = Ktrain_name(outputdir, namebase, fold_idx)
    test_file_name = Ktest_name(outputdir, namebase, fold_idx)
    if not (path.exists(train_file_name) and path.exists(test_file_name)):
        print ('Creating %s %s' % (train_file_name, test_file_name))
        train_idxs = np.concatenate(splits['Train%d' % fold_idx],
                                    splits['Validate%d' % fold_idx] )
        train_idsx.sort()
        test_idxs = splits['Test'] 
        idxs = np.concatenate([train_idxs, test_idxs])
        idxs.sort()
        assert (idxs0 == idxs).all()
        
        train_idxs0 = np.searchsorted(idxs0, train_idxs)
        test_idxs0 = np.searchsorted(idxs0, test_idxs)

        train_X, train_y, train_d = slice_Xydp(image_features,
                                          all_labels, 
                                          None,
                                          None,
                                          train_idxs0)
        test_X, test_y, test_d = slice_Xydp(image_features,
                                       all_labels, 
                                       None,
                                       None,
                                       test_idxs0)                                            
        print ('Finished slicing fold %d' % fold_idx)
        train_X, test_X, _m, _s, _tr = normalize((train_X, test_X))
    
        print('Computing kernels fold %d' % fold_idx)
        train_K = linear_kernel(train_X, train_X, use_theano=True)
        print('Computed fold %d train kernel of size' % fold_idx, train_K.shape)
        test_K = linear_kernel(test_X, train_X, use_theano=True)
        print('Computed fold %d test kernel of size' % fold_idx, test_K.shape)            
        np.save(train_file_name, train_K)
        np.save(test_file_name, test_K)
    else:
        print ('Already exists: %s %s' % (train_file_name, test_file_name))


def train_view2_models_incremental(N, basedir, namebases, 
                                   dataset, outdir, dryrun=True):
    # allocate the gram matrix for each fold
    # This will be incremented as we loop over the models
    Ktrains = [np.zeros((ntrain * 83, ntrain * 83), dtype='float')
               for i in range(1)]
    Ktests  = [np.zeros((ntest * 83, ntrain * 83), dtype='float32')
               for i in range(1)]

    test_errs = {}
    # loop over top N `_id`s incrementing train_X by Ktrain
    
    splits = dataset.splits
    all_paths, all_labels, _ = dataset.raw_classification_task()
    test_y =  all_labels[splits['Test']]
    train_idxs = np.concatenate([splits['Train0'], splits['Validate0']])
    train_idxs.sort()
    train_y = all_labels[train_idxs]
    
    for (n_ind, namebase) in enumerate(namebases[:N]):
        print 'Working through model', namebase

        fold_idx = 0

        if dryrun:
            try:
                open(Ktrain_name(basedir, namebase, fold_idx)).close()
                open(Ktest_name(basedir, namebase, fold_idx)).close()
            except IOError:
                print '---> Missing', namebase, fold_idx
            continue

        Ktrain_n_fold = np.load(Ktrain_name(basedir, namebase, fold_idx))
        Ktest_n_fold = np.load(Ktest_name(basedir, namebase, fold_idx))
        Ktrains[fold_idx] += Ktrain_n_fold
        Ktests[fold_idx] += Ktest_n_fold
    
        svm, _ = train_scikits(
                (Ktrains[fold_idx], train_y, None),
                labelset=range(83),
                model_type='svm.SVC',
                model_kwargs={'kernel': 'precomputed', 'C': 100},
                normalization=False
                )
        test_predictions = svm.predict(Ktests[fold_idx])
        test_err = (test_predictions != test_y).mean()
        print namebase, fold_idx, test_err
        test_errs[(namebase, fold_idx)] = test_err

        outfile = os.path.join(outdir, 'up_to_%d.pkl' % n_ind)
        if not dryrun:
            print 'Mean', namebase, np.mean(
                [test_errs[(namebase, ii)] for ii in range(nfolds)])
            cPickle.dump(test_errs,
                         open(outfile,'w'))
        else:
            print "Would have written to ", outfile
