import time
import numpy as np

import hyperopt
import pyll
from pyll import scope
from pyll.base import Lambda
from pyll import as_apply

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
    # -- /begin distributions that hyperopt can tune
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'normal',
    'one_of',
    'choice',
    # -- /end distributions that hyperopt can tune
    #
    # -- renamed symbols
    **{
    # NEW NAME:         ORIG NAME
    's_int':           'int',
    'HP':              'hyperopt_param',
    'HR':              'hyperopt_result',
    })


def rfilter_size(smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(quniform(smin - q / 2.0 + 1e-5, smax + q / 2.0, q))


def logu_range(lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return loguniform(np.log(lower), np.log(upper))


def maybe():
    return one_of(0, 1)


@pyll.scope.define_info(o_len=2)
def cifar10_img_classification_task(dtype):
    imgs, labels = CF10.img_classification_task(dtype=dtype)
    return imgs, labels


class Cifar10Bandit1(pyll_slm.HPBandit):
    exceptions = [
            (lambda e: isinstance(e, ValueError),
                lambda e: (None, {
                    'loss': float('inf'),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }))
            ]
    def __init__(self, n_train=40000, n_valid=10000, n_test=10000,
            batchsize=20,
            ):
        all_imgs, all_labels = scope.cifar10_img_classification_task(dtype='uint8')

        #TODO: make the nfb0_size depend on the grid_res used in the grouping
        #      at the end,
        #      otherwise the search space is obviously multi-modal
        nfb0_size = HP('nfb0_size', rfilter_size(2, 8))
        nfb0_remove_mean = HP('nfb0_remove_mean', maybe())
        nfb0_beta = HP('nfb0_beta', logu_range(1, 1e4))
        nfb0_hard_beta = HP('nfb0_hard', maybe())
        nfb0_nfilters=s_int(
                        HP('nfbf0_nfilters', qloguniform(
                            np.log(16 / 2.0) + 1e-5,
                            np.log(1600),   # LAYER BY LAYER DIFFERENT
                            q=16)))
        patches = random_patches(all_imgs[:n_train], 50000, nfb0_size, nfb0_size,
                rng=np_RandomState(3214))

        M0_FB0_options = as_apply([
            # -- Pinto's SLM filterbank initialization
            [
                np.asarray(0).reshape((1, 1, 1)),
                alloc_filterbank(
                    nfb0_nfilters, nfb0_size, nfb0_size, 3, 'float32',
                    method_name='random:uniform',
                    method_kwargs={
                        'rseed': HP('nfb0_af_rseed',
                                one_of(1, 2, 3, 4, 5)),
                            },
                    normalize=HP('nfb0_af_normalize', maybe()))
            ],
            # -- simply return whitened pixel values
            patch_whitening_filterbank_X(patches,
                gamma=HP('nfb0_pwf_gamma', logu_range(1e-4, 1.0)),
                o_ndim=4,
                remove_mean=nfb0_remove_mean,
                beta=nfb0_beta,
                hard_beta=nfb0_hard_beta,
                )[:2],
            # -- Coates et al. ICML2011 patch-based initialization
            fb_whitened_patches(
                patches,
                patch_whitening_filterbank_X(patches,
                    gamma=HP('nfb0_wp_gamma', logu_range(1e-4, 1.0)),
                    o_ndim=2,
                    remove_mean=nfb0_remove_mean,
                    beta=nfb0_beta,
                    hard_beta=nfb0_hard_beta,
                    ),
                n_filters=nfb0_nfilters,
                rseed=HP('nfb0_wp_rseed', one_of(1, 2, 3, 4, 5))),
            # --> MORE FB LEARNING ALGOS HERE <--
            # TODO: V1-like filterbank (with whitening matrix)
            # TODO: random matrix multiplied by whitening matrix
            # TODO: RBM/sDAA/ssRBM
            ])


        #TODO: make the parameters of each fb algo conditioned on the *choice*
        #      of algo
        M0_FB0_i = HP('nfb0_algo_i',
                one_of(*range(len(M0_FB0_options.pos_args))))
        M0_FB0 = M0_FB0_options[M0_FB0_i]

        wfb0 = partial(slm_wnorm_fbcorr,
                w_means=np_transpose(
                    asarray(M0_FB0[0], 'float32'),
                    (2, 0, 1)),
                w_fb=np_transpose(
                    asarray(M0_FB0[1], 'float32'),
                    (0, 3, 1, 2)),
                remove_mean=nfb0_remove_mean,
                beta=nfb0_beta,
                hard_beta=nfb0_hard_beta)

        qp = partial(slm_quantize_gridpool,
                alpha=HP('qp_alpha', normal(0.0, 1.0)),
                use_mid=HP('qp_use_mid', maybe()),
                grid_res=HP('qp_grid_res', one_of(2, 3)),
                order=HP('qp_order', one_of(1.0, 2.0, logu_range(.1, 10.))),
                )
        pipeline = [wfb0, qp]

        #print pipeline
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
                abort_on_rows_larger_than=15000,
                )

        test_features = pyll_theano_batched_lmap(
                partial(callpipe1, pipeline),
                all_imgs_cmajor[50000:50000 + n_test],
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=15000,
                )

        cache_train = flatten_elems(screen_features[:n_train])

        xmean, xstd = mean_and_std(
                cache_train,
                remove_std0=True)
        xmean = print_ndarray_summary('Xmean', xmean)
        xstd = print_ndarray_summary('Xstd', xstd)

        xstd_inc = HP('classif_squash_lowvar', logu_range(1e-6, 1e-1))
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
                l2_regularization=HP('l2_reg', logu_range(1e-6, 1e-1)),
                verbose=True,
                solver=('asgd.SubsampledTheanoOVA', {
                    'dtype': 'float32',
                    'verbose': 1,
                    })
                )

        outputs = []
        for name, xy in ('trn', trn_xy), ('val', val_xy), ('tst', tst_xy):
            erate = error_rate(model_predict(svm, xy[0]), xy[1])
            if name == 'val':
                erate = HR('loss', erate)
            outputs.append(HR(name + "_erate", erate))

        pyll_slm.HPBandit.__init__(self, pyll.as_apply(outputs))


if 0:
    pipeline = []
    for ii, nfu in enumerate(nfilt_ubounds):
        ll = ii + 1
        # this is an interior layer, pool like normal SLM
        fbcorr1 = partial(fbcorr,
                ker_size=HP('f%i_size' % ll, rfilter_size(2, 6)),
                foo=foo,
                generate=('random:uniform',
                    {'rseed': HP('f%i_seed' % ll,
                        choice(range(10, 15)))}))
        lpool1 = partial(lpool,
                ker_size=HP('p%i_size' % ll, rfilter_size(2, 5)),
                order=HP('p%i_order' % ll,
                    loguniform(np.log(1), np.log(10))),
                stride=HP('p%i_stride' % ll, 1))

        lnorm1 = partial(lnorm,
                ker_size = HP('n%i_size' % ll, rfilter_size(2, 5)),
                remove_mean=HP('n%i_nomean' % ll, one_of(0, 1)),
                stretch=HP('n%i_stretch' % ll, logu_range(.1/3, 10.*3)),
                threshold=HP('n%i_thresh' % ll, logu_range(.1/3, 10.*3)))

        pipeline.extend([fbcorr1, lpool1, lnorm1])


