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
    'str_join',
    'switch',
    #
    # -- misc. from ./pyll_slm.py
    'pyll_theano_batched_lmap',
    'fit_linear_svm',
    'model_predict',
    'error_rate',
    'mean_and_std',
    'flatten_elems',
    'np_transpose',
    'np_RandomState',
    'print_ndarray_summary',
    #
    # -- filterbank allocators  (./pyll.slm.py)
    'random_patches',
    'alloc_filterbank',
    'patch_whitening_filterbank_X',
    'fb_whitened_patches',
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
def cifar10_img_classification_task(dtype):
    imgs, labels = CF10.img_classification_task(dtype=dtype)
    return imgs, labels

@hyperopt.as_bandit(
        exceptions=[
            (
                lambda e: isinstance(e, ValueError),
                lambda e: {
                    'loss': float('inf'),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }
            )
        ])
def cifar10bandit(n_train=40000, n_valid=10000, n_test=10000, batchsize=20):

    all_imgs, all_labels = scope.cifar10_img_classification_task(dtype='uint8')

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

    patches = random_patches(all_imgs[:n_train], 50000, fb0_size, fb0_size,
            rng=np_RandomState(3214))

    M0_FB0 = hp_choice('fb0_algo_i', [
        # -- Pinto's SLM filterbank initialization
        slm_uniform_M_FB(
            nfilters=fb0_nfilters,
            size=fb0_size,
            rseed=hp_choice('fb0_af_rseed', range(1, 6)),
            normalize=hp_TF('fb0_af_normalize')),
        # -- simply return whitened pixel values
        patch_whitening_filterbank_X(patches,
            gamma=logu_range('fb0_pwf_gamma', 1e-4, 1.0),
            o_ndim=4,
            remove_mean=fb0_remove_mean,
            beta=fb0_beta,
            hard_beta=fb0_hard_beta,
            )[:2],
        # -- Coates et al. ICML2011 patch-based initialization
        fb_whitened_patches(patches,
            patch_whitening_filterbank_X(patches,
                gamma=logu_range('fb0_wp_gamma', 1e-4, 1.0),
                o_ndim=2,
                remove_mean=fb0_remove_mean,
                beta=fb0_beta,
                hard_beta=fb0_hard_beta,
                ),
            n_filters=fb0_nfilters,
            rseed=hp_choice('fb0_wp_rseed', range(6, 11))),
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

    assert n_train + n_valid <= 50000
    assert n_test <= 10000

    # -- map cifar10 through the pipeline
    all_imgs_cmajor = np_transpose(all_imgs, (0, 3, 1, 2))

    screen_features = pyll_theano_batched_lmap(
            partial(callpipe1, pipeline),
            all_imgs_cmajor[:n_train + n_valid],
            batchsize=batchsize,
            print_progress=100,
            abort_on_rows_larger_than=13000,
            )

    test_features = pyll_theano_batched_lmap(
            partial(callpipe1, pipeline),
            all_imgs_cmajor[50000:50000 + n_test],
            batchsize=batchsize,
            print_progress=100,
            abort_on_rows_larger_than=13000,
            )

    cache_train = flatten_elems(screen_features[:n_train])

    xmean, xstd = mean_and_std(
            cache_train,
            remove_std0=True)
    xmean = print_ndarray_summary('Xmean', xmean)
    xstd = print_ndarray_summary('Xstd', xstd)

    xstd_inc = logu_range('classif_squash_lowvar', 1e-6, 1e-1)
    xstd = sqrt(xstd ** 2 + xstd_inc)

    trn_xy=(
        (cache_train - xmean) / xstd,
        all_labels[:n_train])

    val_xy = (
        (screen_features[n_train:n_train + n_valid]
            - xmean) / xstd,
        all_labels[n_train:n_train + n_valid])

    tst_xy = (
        (test_features[:] - xmean) / xstd,
        all_labels[50000:50000 + n_test])

    svm = fit_linear_svm(trn_xy,
            l2_regularization=logu_range('l2_reg', 1e-6, 1e-1),
            verbose=True,
            solver=('asgd.SubsampledTheanoOVA', {
                'dtype': 'float32',
                'verbose': 1,
                })
            )

    outputs = []
    trn_erate = error_rate(model_predict(svm, trn_xy[0]), trn_xy[1])

    val_pred = model_predict(svm, val_xy[0])
    val_erate = error_rate(val_pred, val_xy[1])
    tst_erate = error_rate(model_predict(svm, tst_xy[0]), tst_xy[1])
    result = {
            # -- criterion to optimize
            'loss': val_erate,
            # -- other error rates
            'trn_erate': trn_erate,
            'val_erate': val_erate,
            'tst_erate': tst_erate,
            # -- other stats to save
            'val_pred': str_join('', pyll_map(str, val_pred)),
            }
    return result

