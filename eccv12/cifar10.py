import time
import numpy as np

import pyll
from pyll import scope
from pyll.base import Lambda
from pyll import as_apply

import hyperopt
from hyperopt.pyll_utils import hp_normal
from hyperopt.pyll_utils import hp_choice
from hyperopt.pyll_utils import hp_uniform
from hyperopt.pyll_utils import hp_quniform
from hyperopt.pyll_utils import hp_loguniform
from hyperopt.pyll_utils import hp_qloguniform

from skdata.cifar10 import CIFAR10
CF10 = CIFAR10()

import pyll_slm  # adds the symbols to pyll.scope

pyll.scope.import_(globals(),
    # -- from pyll
    'partial',
    'callpipe1',
    'asarray',
    'sqrt',
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
    #
    # -- filterbank allocators  (./pyll.slm.py)
    'random_patches',
    'alloc_filterbank',
    'patch_whitening_filterbank_X',
    'fb_whitened_patches',
    'fb_whitened_projections',
    'slm_uniform_M_FB',
    #
    # -- pipeline elements  (./pyll.slm.py)
    'slm_lnorm',
    'slm_lpool',
    'slm_lpool_smallgrid',
    'slm_gnorm',
    'slm_fbcorr',
    'slm_wnorm_fbcorr',
    'slm_alpha_quantize',
    'slm_quantize_gridpool',
    'slm_flatten',
    #
    # -- renamed symbols
    **{
    # NEW NAME:         ORIG NAME
    's_int':           'int',
    'pyll_len':        'len',
    'pyll_map':        'map',
    })


def rfilter_size(label, smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(hp_quniform(label, smin - q / 2.0 + 1e-5, smax + q / 2.0, q))


def logu_range(label, lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return hp_loguniform(label, np.log(lower), np.log(upper))


# move to hyperopt.pyll_util
def hp_TF(label):
    return hp_choice(label, [0, 1])


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
    if shuffle_seed:
        # -- my idea here was to use RandomState.shuffle to mix up each set
        #    so that the classes aren't all contiguous.  SGD in particular
        #    benefits from that.
        raise NotImplementedError()

    return {
            'trn_images': trn_images,
            'trn_labels': trn_labels,
            'val_images': val_images,
            'val_labels': val_labels,
            'tst_images': tst_images,
            'tst_labels': tst_labels,
            }


@hyperopt.as_bandit(
        exceptions=[
            (
                # -- this is raised when the n. of features > max_n_features
                lambda e: isinstance(e, ValueError),
                lambda e: {
                    'loss': float('inf'),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }
            )
        ])
def cifar10bandit(n_train=40000, n_valid=10000, n_test=10000, batchsize=20,
        max_n_features=13000):

    data = scope.cifar10_img_classification_task(
            dtype='uint8',
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test)

    fb0_size = rfilter_size('fb0_size', 2, 8)

    # duplicate the variables that depend critically on the grid-resolution
    # so that the output nfilters can be tuned separately for each grid res.

    # XXX - modify pyll's choice fn to inspect the o_len of each option. If
    # they all match, then set its own o_len to that too... this would permit
    # unpacking: grid_res, nfilters, use_mid = hp_choice(...)
    out_shape_info = hp_choice('grid_res',
            [
                [
                    2,
                    s_int(
                        hp_qloguniform(
                            'fbf0_nfilters2',
                            np.log(16 / 2.0) + 1e-5,
                            np.log(1600),
                            q=16)),
                    hp_TF('qp_use_mid2'),
                ],
                [
                    3,
                    s_int(
                        hp_qloguniform(
                            'fbf0_nfilters3',
                            np.log(16 / 2.0) + 1e-5,
                            np.log(1600),
                            q=16)),
                    hp_TF('qp_use_mid3'),
                ]
            ])
    qp_grid_res = out_shape_info[0]
    fb0_nfilters = out_shape_info[1]
    qp_use_mid = out_shape_info[2]

    fb0_remove_mean = hp_TF('fb0_remove_mean')
    fb0_beta = logu_range('fb0_beta', 1., 1e4)
    fb0_hard_beta = hp_TF('fb0_hard')

    patches = random_patches(data['trn_images'], 50000, fb0_size, fb0_size,
            rng=np_RandomState(3214))

    M0_FB0 = hp_choice('fb0_algo', [
        # -- Pinto's SLM filterbank initialization (random in pixel space)
        #    (verified to match Pythor3)
        slm_uniform_M_FB(
            nfilters=fb0_nfilters,
            size=fb0_size,
            rseed=hp_choice('fb0_pr_rseed', range(1, 6)),
            normalize=hp_TF('fb0_pr_normalize'),
            ),
        # -- Coates et al. ICML2011 random projections initialization
        #    (XXX: unverified)
        fb_whitened_projections(patches,
            patch_whitening_filterbank_X(patches,
                gamma=logu_range('fb0_wr_gamma', 1e-4, 1.0),
                o_ndim=2,
                remove_mean=fb0_remove_mean,  # match wfb0 partial below
                beta=fb0_beta,                # ...
                hard_beta=fb0_hard_beta,      # ...
                ),
            n_filters=fb0_nfilters,
            rseed=hp_choice('fb0_wr_rseed', range(6, 11)),
            ),
        # -- Coates et al. ICML2011 patch-based initialization (imprinting)
        #    (verified to match matlab demo code)
        fb_whitened_patches(patches,
            patch_whitening_filterbank_X(patches,
                gamma=logu_range('fb0_wp_gamma', 1e-4, 1.0),
                o_ndim=2,
                remove_mean=fb0_remove_mean,  # match wfb0 partial below
                beta=fb0_beta,                # ...
                hard_beta=fb0_hard_beta,      # ...
                ),
            n_filters=fb0_nfilters,
            rseed=hp_choice('fb0_wp_rseed', range(6, 11)),
            ),
        # --> MORE FB LEARNING ALGOS HERE <--
        # TODO: V1-like filterbank (with whitening matrix)
        # TODO: random matrix multiplied by whitening matrix
        # TODO: RBM/sDAA/ssRBM
        ])

    wfb0 = partial(slm_wnorm_fbcorr,
            w_means=np_transpose(
                asarray(M0_FB0[0], 'float32'),
                (2, 0, 1)),
            w_fb=np_transpose(
                asarray(M0_FB0[1], 'float32'),
                (0, 3, 1, 2)),
            remove_mean=fb0_remove_mean,
            beta=fb0_beta,
            hard_beta=fb0_hard_beta)

    qp = partial(slm_quantize_gridpool,
            alpha=hp_normal('qp_alpha', 0.0, 1.0),
            use_mid=qp_use_mid,
            grid_res=qp_grid_res,
            order=hp_choice('qp_order', [
                1.0, 2.0, logu_range('qp_order_free', .1, 10.)]))
    pipeline = [wfb0, qp]

    #print pipeline
    #
    # TODO:
    #   pipeline = hp_choice('pipe_len',
    #         [pipeline1, pipeline2, pipeline3])
    pipeline = pyll.as_apply(pipeline)
    features = {}
    for split in 'trn', 'val', 'tst':
        features[split] = pyll_theano_batched_lmap(
                partial(callpipe1, pipeline),
                np_transpose(data['%s_images' % split], (0, 3, 1, 2)),
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=max_n_features,
                )

    # load full training set into memory
    cache_train = flatten_elems(features['trn'][:])

    xmean, xstd = mean_and_std(cache_train, remove_std0=hp_TF('remove_std0'))
    xmean = print_ndarray_summary('Xmean', xmean)
    xstd = print_ndarray_summary('Xstd', xstd)

    xstd_inc = logu_range('classif_squash_lowvar', 1e-10, 1.)
    xstd = sqrt(xstd ** 2 + xstd_inc)

    trn_xy=((cache_train - xmean) / xstd, data['trn_labels'])
    val_xy = ((features['val'] - xmean) / xstd, data['val_labels'])
    tst_xy = ((features['tst'] - xmean) / xstd, data['tst_labels'])

    svm = fit_linear_svm(trn_xy,
            l2_regularization=logu_range('l2_reg', 1e-6, 1e-1),
            verbose=True,
            solver=('asgd.SubsampledTheanoOVA', {
                'dtype': 'float32',
                'verbose': 1,
                })
            )
    val_erate = error_rate(model_predict(svm, val_xy[0]), val_xy[1]),
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

