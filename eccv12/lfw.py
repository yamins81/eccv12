import sys
import os
import hashlib

import skdata.larray
import skdata.utils
import skdata.lfw
import numpy as np
from thoreano.slm import (TheanoExtractedFeatures,
                          use_memmap)
                          
from thoreano.classifier import (train_only_asgd, 
                                 get_result)
import comparisons as comp_module

from .utils import ImgLoaderResizer

DEFAULT_COMPARISON = 'mult'

##################################
########lfw task evaluation
def test_splits():
    # Does this  function belong in skdata?
    T = ['fold_' + str(i) for i in range(10)]
    splits = []
    for i in range(10):
        inds = range(10)
        inds.remove(i)
        v_ind = (i+1) % 10
        inds.remove(v_ind)
        test = T[i]
        validate = T[v_ind]
        train = [T[ind] for ind in inds]
        splits.append({'train': train,
                       'validate': validate,
                       'test': test})
    return splits


def get_test_performance(outfile, config, flip_lr=False, comparison=DEFAULT_COMPARISON):
    """adapter to construct split notation for 10-fold split and call
    get_performance on it (e.g. this is like "test a config on View 2")
    """
    splits = test_splits()
    return get_performance(outfile, config, train_test_splits=splits,
                           use_theano=use_theano, flip_lr=flip_lr, tlimit=None,
                           comparisons=comparisons)


def get_performance(config,
                    train_decisions,
                    test_decisions,
                    comparison=DEFAULT_COMPARISON):
    """
    config - a Pythor3-compatible dictionary
    """
    config_hash = get_config_string(config)
    assert hasattr(comp_module, comparison)
    dataset = skdata.lfw.Aligned()
    X, y, Xr = get_relevant_images(dataset, splits = ['DevTrain', 'DevTest'],
                                   dtype='float32')
    batchsize = 4
    feature_file_name = 'features_' + config_hash + '.dat'
    train_pairs_filename = 'train_pairs_' + config_hash + '.dat'
    test_pairs_filename = 'test_pairs_' + config_hash + '.dat'
    comparison_obj = getattr(comp_module,comparison)
    TEF_args = X, batchsize, [config], [feature_file_name]
    with TheanoExtractedFeatures(*TEF_args) as features_fps:
        feature_shps = [features_fp.shape for features_fp in features_fps]
        print('Doing comparison %s' % comparison)
        n_features = sum([comparison_obj.get_num_features(f_shp)
                          for f_shp in feature_shps])
        f_info = {'feature_shapes': feature_shps, 'n_features': n_features}
        #TODO: figure out whoat to do here  and ensure training_errors,
        #      predictions, labels are returned 
        #      correctly
        with PairFeatures(dataset, 'DevTrain', Xr,
                          n_features, features_fps, comparison_obj, 
                          train_pairs_filename) as train_Xy:
            train_features, train_labels = train_Xy
            train_margins = train_labels * train_decisions       
            model, training_data = train_only_asgd(train_Xy,
                                                   margin_biases=train_margins)
            new_train_decisions = model.decision_function(train_features)
        with PairFeatures(dataset, 'DevTest', Xr,
                          n_features, features_fps, comparison_obj, 
                          test_pairs_filename) as test_Xy:
            test_features, test_labels = test_Xy
            new_test_decisions = model.decision_function(test_features)
     
    train_decisions += new_train_decisions
    test_decisions += new_test_decisions
    train_predictions = np.sign(train_decisions)
    test_predictions = np.sign(test_decisions)
 
    result = {} 
    result.update(f_info)
    result['weights'] = model.asgd_weights
    result['bias'] = model.asgd_bias
    result['train_decisions'] = train_decisions.tolist() 
    result['test_decisions'] = test_decisions.tolist() 
    stats = get_result(train_labels,
                             test_labels,
                             train_predictions,
                             test_predictions,
                             [-1, 1])
    result.update(stats)
    result['loss'] = float(1 - result['test_accuracy']/100.)
    return result


def get_relevant_images(dataset, splits=None, dtype='uint8'):
    """
    Return lazy image array, individuals by id, and image path array
    """
    # load & resize logic is LFW Aligned -specific
    assert 'Aligned' in str(dataset.__class__)

    # fetch the raw paths of lfw
    # Xr (X raw) is the image paths
    # yr is the individuals (by number)
    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)

    if splits is not None:
        splits = unroll(splits)

    if splits is not None:
        all_images = []
        for s in splits:
            A, B, c = dataset.raw_verification_task(split=s)
            all_images.extend([A,B])
        all_images = np.unique(np.concatenate(all_images))

        inds = np.searchsorted(Xr, all_images)
        Xr = Xr[inds]
        yr = yr[inds]

    X = skdata.larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  # lfw-specific
                    dtype=dtype),
                Xr)

    Xr = np.array([os.path.split(x)[-1] for x in Xr])

    return X, yr, Xr


class PairFeatures(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def work(self, dset, split, X, n_features,
             features_fps, comparison_obj, filename, flip_lr=False):
        if isinstance(split, str):
            split = [split]
        A = []
        B = []
        labels = []
        for s in split:
            if s.startswith('re'):
                s = s[2:]
                A0, B0, labels0 = dset.raw_verification_task_resplit(split=s)
            else:
                A0, B0, labels0 = dset.raw_verification_task(split=s)
            A.extend(A0)
            B.extend(B0)
            labels.extend(labels0)
        Ar = np.array([os.path.split(ar)[-1] for ar in A])
        Br = np.array([os.path.split(br)[-1] for br in B])
        labels = np.array(labels)
        if set(labels)  == set([0, 1]):
            labels = 2*labels - 1
        Aind = np.searchsorted(X, Ar)
        Bind = np.searchsorted(X, Br)
        assert len(Aind) == len(Bind)
        pair_shp = (len(labels), n_features)

        if flip_lr:
            pair_shp = (4 * pair_shp[0], pair_shp[1])

        size = 4 * np.prod(pair_shp)
        print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))
        memmap = filename is not None and use_memmap(size)
        if memmap:
            print('get_pair_fp memmap %s for features of shape %s' % (
                                                    filename, str(pair_shp)))
            feature_pairs_fp = np.memmap(filename,
                                    dtype='float32',
                                    mode='w+',
                                    shape=pair_shp)
        else:
            print('using memory for features of shape %s' % str(pair_shp))
            feature_pairs_fp = np.empty(pair_shp, dtype='float32')
        feature_labels = []

        for (ind,(ai, bi)) in enumerate(zip(Aind, Bind)):
            # -- this flattens 3D features to 1D features
            if flip_lr:
                feature_pairs_fp[4 * ind + 0] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, :, :],
                            fp[bi, :, :, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 1] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, ::-1, :],
                            fp[bi, :, :, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 2] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, :, :],
                            fp[bi, :, ::-1, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 3] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, ::-1, :],
                            fp[bi, :, ::-1, :])
                            for fp in features_fps])

                feature_labels.extend([labels[ind]] * 4)
            else:
                feats = [comparison_obj(fp[ai],fp[bi])
                        for fp in features_fps]
                feature_pairs_fp[ind] = np.concatenate(feats)
                feature_labels.append(labels[ind])
            if ind % 100 == 0:
                print('get_pair_fp  %i / %i' % (ind, len(Aind)))

        if memmap:
            print ('flushing memmap')
            sys.stdout.flush()
            del feature_pairs_fp
            self.filename = filename
            self.features = np.memmap(filename,
                    dtype='float32',
                    mode='r',
                    shape=pair_shp)
        else:
            self.features = feature_pairs_fp
            self.filename = ''

        self.labels = np.array(feature_labels)


    def __enter__(self):
        self.work(*self.args, **self.kwargs)
        return (self.features, self.labels)

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)


def unroll(X):
    Y = []
    for x in X:
        if isinstance(x,str):
            Y.append(x)
        else:
            Y.extend(x)
    return np.unique(Y)


def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def random_id():
    return hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()
