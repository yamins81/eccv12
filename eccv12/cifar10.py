import sys
import time
import numpy as np

import pyll
from pyll import scope
from pyll.base import Lambda
from pyll import as_apply

import hyperopt

from skdata.cifar10 import CIFAR10
CF10 = CIFAR10()

import pyll_slm  # adds the symbols to pyll.scope
from .slm import choose_pipeline

pyll.scope.import_(globals(),
    # -- from pyll
    'partial',
    'callpipe1',
    'asarray',
    'sqrt',
    'switch',
    #
    # -- misc. from ./pyll_slm.py
    'pyll_theano_batched_lmap',
    'fit_linear_svm',
    'model_predict',
    'model_decisions',
    'error_rate',
    'mean_and_std',
    'flatten_elems',
    'np_transpose',
    'np_RandomState',
    'print_ndarray_summary',
    'pickle_dumps',
    )


@pyll.scope.define_info(o_len=2)
def cifar10_img_classification_task(dtype, n_train, n_valid, n_test,
        label_set=range(10), shuffle_seed=None):
    images, labels = CF10.img_classification_task(dtype=dtype)
    n_classes = 10
    assert n_train + n_valid <= 50000
    assert n_test <= 10000

    # -- divide up the dataset as it was meant: train / test
    trn_images = images[:50000]
    trn_labels = labels[:50000]
    tst_images = images[50000:]
    tst_labels = labels[50000:]

    # -- now carve it up so that we have balanced classes for fitting and
    #    validation

    train = {}
    test = {}
    print 're-indexing dataset'
    for label in label_set:
        print 'pulling out label', label
        train[label] = trn_images[trn_labels == label]
        test[label] = tst_images[tst_labels == label]
        assert len(train[label]) == len(trn_labels) / n_classes
        assert len(test[label]) == len(tst_labels) / n_classes

    if np.any(np.asarray([n_train, n_valid, n_test]) % len(label_set)):
        raise NotImplementedError()
    else:
        trn_K = n_train // len(label_set)
        val_K = n_valid // len(label_set)
        tst_K = n_test // len(label_set)
        trn_images = np.concatenate([train[label][:trn_K]
            for label in label_set])
        trn_labels = np.concatenate([[label] * trn_K
            for label in label_set])

        assert len(trn_images) == len(trn_labels)
        assert trn_images.shape == (n_train, 32, 32, 3)
        assert trn_labels.shape == (n_train,)

        val_images = np.concatenate([train[label][trn_K:trn_K + val_K]
            for label in label_set])
        val_labels = np.concatenate([[label] * val_K
            for label in label_set])

        assert len(val_images) == len(val_labels)
        assert val_images.shape == (n_valid, 32, 32, 3)
        assert val_labels.shape == (n_valid,)

        tst_images = np.concatenate([test[label][:tst_K]
            for label in label_set])
        tst_labels = np.concatenate([[label] * tst_K
            for label in label_set])

        assert len(tst_images) == len(tst_labels)
        assert tst_images.shape == (n_test, 32, 32, 3)
        assert tst_labels.shape == (n_test,)

    print 'done re-indexing dataset'
    def shuffle(X, s):
        if shuffle_seed:
            np.random.RandomState(shuffle_seed + s).shuffle(X)
        return X

    return {
            'trn_images': shuffle(trn_images, 0),
            'trn_labels': shuffle(trn_labels, 0),
            'val_images': shuffle(val_images, 1),
            'val_labels': shuffle(val_labels, 1),
            'tst_images': shuffle(tst_images, 2),
            'tst_labels': shuffle(tst_labels, 2),
            }


@hyperopt.as_bandit(
        exceptions=[
            (
                lambda e: (
                    isinstance(e, pyll_slm.InvalidDescription)
                    or isinstance(e, ZeroDivisionError)
                    or isinstance(e, MemoryError)
                    or (isinstance(e, ValueError)
                        and 'rowlen' in str(e)
                        and 'exceeds limit' in str(e))
                    or (isinstance(e, ValueError)
                        and 'dimension mis-match' in str(e)
                        and '= 0' in str(e))
                    or (isinstance(e, ValueError)
                        and 'had size 0' in str(e))
                    or (isinstance(e, ValueError)
                        and 'size on that axis is 0' in str(e))
                    or (isinstance(e, ValueError)
                        and 'low >= high' in str(e))
                    or (isinstance(e, RuntimeError)
                        and 'taking too long' in str(e))
                    or (isinstance(e, RuntimeError)
                        and 'allocate memory' in str(e))
                    or (isinstance(e, RuntimeError)
                        and 'kernel_reduce_sum' in str(e)
                        and 'block: 0 x' in str(e))
                    or (isinstance(e, RuntimeError)
                        and 'CudaNdarray has dim 0' in str(e))
                ),
                lambda e: {
                    'loss': float(1.0),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }
            ),
        ],
        )
def cifar10bandit(
        n_train=40000,
        n_valid=10000,
        n_test=10000,
        batchsize=20,
        n_imgs_for_patches=10000,
        # -- maximum n. features per example coming out of the pipeline
        max_n_features=16000,
        n_patches=50000,
        # -- seconds allocated to pipeline creation
        #    (This includes processing time for computing patches)
        pipeline_timeout=90.0,
        # -- max n. filterbank elements going into another layer
        max_layer_sizes=[64, 128],
        svm_solver=('asgd.SubsampledTheanoOVA', {
            'dtype': 'float32',
            'verbose': 1,
            }),
        ):

    data = scope.cifar10_img_classification_task(
            dtype='uint8',
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            shuffle_seed=5)

    pipeline = choose_pipeline(
            Xcm=np_transpose(
                data['trn_images'][:n_imgs_for_patches],
                (0, 3, 1, 2)),
            n_patches=n_patches,
            batchsize=batchsize,
            max_n_features=max_n_features,
            max_layer_sizes=max_layer_sizes,
            time_limit=pipeline_timeout,
            )
    #print pipeline

    features = {}
    for split in 'trn', 'val', 'tst':
        features[split] = pyll_theano_batched_lmap(
                partial(callpipe1, pipeline['pipe']),
                np_transpose(data['%s_images' % split], (0, 3, 1, 2)),
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=max_n_features,
                )

    # load full training set into memory
    cache_train = flatten_elems(features['trn'][:])

    xmean, xstd = mean_and_std(cache_train, remove_std0=pipeline['remove_std0'])
    xmean = print_ndarray_summary('Xmean', xmean)
    xstd = print_ndarray_summary('Xstd', xstd)

    xstd = sqrt(xstd ** 2 + pipeline['varthresh'])

    trn_xy=((cache_train - xmean) / xstd, data['trn_labels'])
    val_xy = ((flatten_elems(features['val'][:]) - xmean) / xstd,
            data['val_labels'])
    tst_xy = ((flatten_elems(features['tst'][:]) - xmean) / xstd,
            data['tst_labels'])

    svm = fit_linear_svm(trn_xy,
            l2_regularization=pipeline['l2_reg'],
            verbose=True,
            solver=svm_solver,
            )
    val_erate = error_rate(model_predict(svm, val_xy[0]), val_xy[1])
    result = {
            # -- criterion to optimize
            'loss': val_erate,
            # -- other error rates
            'trn_erate': error_rate(model_predict(svm, trn_xy[0]), trn_xy[1]),
            'val_erate': val_erate,
            'tst_erate': error_rate(model_predict(svm, tst_xy[0]), tst_xy[1]),
            # -- larger stats to save
            'attachments': {
                'val_decisions.npy.pkl': pickle_dumps(
                    asarray(
                        model_decisions(svm, val_xy[0]),
                        dtype='float32'),
                    protocol=-1)
                }
            }
    return result

