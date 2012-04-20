import time
import numpy as np

import pyll
from pyll import scope
from pyll.base import Lambda
from pyll import as_apply

import hyperopt
from hyperopt.pyll_utils import hp_choice
from hyperopt.pyll_utils import hp_uniform
from hyperopt.pyll_utils import hp_quniform
from hyperopt.pyll_utils import hp_loguniform
from hyperopt.pyll_utils import hp_qloguniform
from hyperopt.pyll_utils import hp_normal
from hyperopt.pyll_utils import hp_lognormal

import pyll_slm  # adds the symbols to pyll.scope

pyll.scope.import_(globals(),
    # -- from pyll
    'partial',
    'callpipe1',
    'switch',
    #
    # -- misc. from ./pyll_slm.py
    'pyll_theano_batched_lmap',
    'np_RandomState',
    #
    # -- filterbank allocators  (./pyll.slm.py)
    'random_patches',
    'alloc_random_uniform_filterbank',
    'patch_whitening_filterbank_X',
    'fb_whitened_patches',
    'fb_whitened_projections',
    'slm_uniform_M_FB',
    #
    # -- pipeline elements  (./pyll.slm.py)
    'slm_lpool',
    'slm_lpool_alpha',
    'slm_fbncc_chmaj',
    'slm_quantize_gridpool',
    #
    # -- renamed symbols
    **{
    # NEW NAME:         ORIG NAME
    's_int':           'int',
    'pyll_getattr':    'getattr',
    })


def hp_TF(label):
    return hp_choice(label, [0, 1])


def rfilter_size(label, smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(hp_quniform(label, smin - q / 2.0 + 1e-5, smax + q / 2.0, q))


def logu_range(label, lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return hp_loguniform(label, np.log(lower), np.log(upper))


def new_fbncc_layer(prefix, Xcm, n_patches, n_filters, size):
    def lab(msg):
        return '%s_fbncc_%s' % (prefix, msg)

    patches = random_patches(Xcm, n_patches, size, size,
            rng=np_RandomState(hash(prefix)), channel_major=True)

    remove_mean = hp_TF(lab('remove_mean'))
    beta = hp_lognormal(lab('beta'), np.log(100), np.log(100))
    hard_beta = hp_TF(lab('hard'))

    # TODO: use different nfilters, beta etc. for each algo

    # -- random projections filterbank allocation
    random_projections = partial(slm_fbncc_chmaj,
        m_fb=slm_uniform_M_FB(
            nfilters=n_filters,
            size=size,
            channels=pyll_getattr(Xcm, 'shape')[1],
            rseed=hp_choice(lab('r_rseed'), range(1, 6)),
            normalize=hp_TF(lab('r_normalize')),
            dtype='float32',
            ret_cmajor=True,
            ),
        remove_mean=remove_mean,
        beta=beta,
        hard_beta=hard_beta)

    # -- random whitened projections filterbank allocation
    random_whitened_projections = partial(slm_fbncc_chmaj,
            m_fb=fb_whitened_projections(patches,
                patch_whitening_filterbank_X(patches,
                    gamma=hp_lognormal(lab('wr_gamma'),
                                       np.log(1e-2), np.log(100)),
                    o_ndim=2,
                    remove_mean=remove_mean,
                    beta=beta,
                    hard_beta=hard_beta,
                    ),
                n_filters=n_filters,
                rseed=hp_choice(lab('wr_rseed'), range(6, 11)),
                dtype='float32',
                ),
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # -- whitened patches filterbank allocation
    whitened_patches = partial(slm_fbncc_chmaj,
            m_fb=fb_whitened_patches(patches,
                patch_whitening_filterbank_X(patches,
                    gamma=hp_lognormal(lab('wp_gamma'),
                                       np.log(1e-2), np.log(100)),
                    o_ndim=2,
                    remove_mean=remove_mean,
                    beta=beta,
                    hard_beta=hard_beta,
                    ),
                n_filters=n_filters,
                rseed=hp_choice(lab('wp_rseed'), range(6, 11)),
                dtype='float32',
                ),
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # --> MORE FB LEARNING ALGOS HERE <--
    # TODO: V1-like filterbank (incl. with whitening matrix)
    # TODO: sparse coding
    # TODO: OMP from Coates 2011
    # TODO: K-means
    # TODO: RBM/sDAA/ssRBM
    rchoice = hp_choice(lab('algo'), [
        random_projections,
        random_whitened_projections,
        whitened_patches,
        ])
    return rchoice


def pipeline_extension(prefix, X, n_patches, max_filters):
    assert max_filters > 16
    f_layer = new_fbncc_layer(prefix, X, n_patches,
            n_filters=s_int(
                hp_qloguniform('%sfb_nfilters' % prefix,
                    np.log(8.01), np.log(max_filters), q=16)),
            size=rfilter_size('%sfb_size' % prefix, 3, 8),
            )

    p_layer = partial(slm_lpool,
            stride=hp_choice('%sp_stride' % prefix, [1, 2]),
            order=hp_choice('%sp_order' % prefix,
                [1, 2, hp_lognormal('%sp_order_real' % prefix,
                    mu=np.log(1), sigma=np.log(3))]),
            ker_size=rfilter_size('%sp_size' % prefix, 2, 8))

    return [f_layer, p_layer]


def pipeline_exits(pipeline, layer_num, Xcm, n_patches, max_n_features):
    def lab(msg):
        return 'l%i_out_%s' % (layer_num, msg)

    rval = []

    fsize = rfilter_size(lab('fsize'), 3, 8)

    grid_res = hp_choice(lab('grid_res'), [2, 3])
    grid_features_per_filter = 2 * (grid_res ** 2)
    grid_nfilters = max_n_features // grid_features_per_filter

    grid_filtering = new_fbncc_layer(
            prefix='l%ieg' % layer_num,
            Xcm=Xcm,
            n_patches=n_patches,
            n_filters=grid_nfilters,
            size=fsize,
            )

    grid_pooling = partial(slm_quantize_gridpool,
            alpha=hp_normal(lab('grid_alpha'), 0.0, 1.0),
            use_mid=False,
            grid_res=grid_res,
            order=hp_choice(lab('grid_order'), [
                1.0, 2.0, logu_range(lab('grid_order_real'), .1, 10.)]))

    rval.append({
        'pipe': pipeline + [grid_filtering, grid_pooling],
        'remove_std0': hp_TF(lab('grid_classif_remove_std0')),
        'varthresh': hp_lognormal(lab('grid_classif_varthresh'),
            np.log(1e-4), np.log(1000)),
        'l2_reg': hp_lognormal(lab('grid_classif_l2_reg'),
            np.log(1e-3), np.log(100)),
        })

    #
    # -- now set up the lpool_alpha option
    filtering_res = pyll_getattr(Xcm, 'shape')[2] - fsize + 1
    # -- N.B. Xrows depends on other params, so we can't use it to set the
    #         upper bound on lpsize. We can only sample independently, and
    #         then fail below with non-positive number of features.
    lpool_size = rfilter_size(lab('lpsize'), 1, 5)
    lpool_res = scope.max(filtering_res - lpool_size + 1, 0)
    if 0:
        # XXX: This is a smarter way to pick the n_filters, but it triggers
        # a bug in hyperopt.vectorize_helper.  The build_idxs_vals function
        # there needs to be smarter -- to recognize when wanted_idxs is a
        # necessarily subset of the all_idxs, and then not to append
        # wanted_idxs to the union defining all_idxs... because that creates a
        # cycle.  The trouble is specifically that lpool_res is used in the
        # switch statement below both in the condition and the response.
        lpool_nfilters = switch(lpool_res > 0,
            max_n_features // (2 * (lpool_res ** 2)),
            scope.Raise(ValueError, 'Non-positive number of features'))
    else:
        # this is less good because it risks dividing by zero,
        # and forces the bandit to catch weirder errors from new_fbncc_layer
        # caused by negative nfilters
        lpool_nfilters = max_n_features // (2 * (lpool_res ** 2))

    local_filtering = new_fbncc_layer(
            prefix='l%iel' % layer_num,
            Xcm=Xcm,
            n_patches=n_patches,
            n_filters=lpool_nfilters,
            size=fsize,
            )

    local_pooling = partial(slm_lpool_alpha,
            ker_size=lpool_size,
            alpha=hp_normal(lab('local_alpha'), 0.0, 1.0),
            order=hp_choice(lab('local_order'), [
                1.0, 2.0, logu_range(lab('local_order_real'), .1, 10.)]))

    rval.append({
        'pipe': pipeline + [local_filtering, local_pooling],
        'remove_std0': hp_TF(lab('local_classif_remove_std0')),
        'varthresh': hp_lognormal(lab('local_classif_varthresh'),
            np.log(1e-4), np.log(1000)),
        'l2_reg': hp_lognormal(lab('local_classif_l2_reg'),
            np.log(1e-3), np.log(100)),
        })

    print 'EXITS RVAL DFS LEN', len(pyll.toposort(as_apply(rval)))
    return rval


def choose_pipeline(Xcm, n_patches, batchsize,
        max_n_features, max_layer_sizes, time_limit,
        memlimit=1000 * (1024 ** 2)):
    """
    This function works by creating a linear pipeline, with multiple exit
    points that could be the feature representation for classification.

    Xcm - channel-major images from which patches are to be extracted.

    The function returns a switch among all of these exit points.
    """
    start_time = time.time()

    pipeline = []
    exits = pipeline_exits(
            pipeline,
            layer_num=0,
            Xcm=Xcm,
            n_patches=n_patches,
            max_n_features=max_n_features)
    for layer_i, max_layer_size in enumerate(max_layer_sizes):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise RuntimeError('pipeline creation is taking too long')
        extension = pipeline_extension(
                'l%i' % layer_i, Xcm, n_patches, max_layer_size)

        pipeline.extend(extension)
        Xcm = pyll_theano_batched_lmap(
                partial(callpipe1, extension),
                scope.print_ndarray_summary('Xcm %i' % layer_i, Xcm),
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=memlimit / scope.len(Xcm)
                )[:]  # -- indexing computes all the values (during rec_eval)

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise RuntimeError('pipeline creation is taking too long')
        exits.extend(
                pipeline_exits(
                    pipeline=pipeline,
                    layer_num=layer_i + 1,
                    Xcm=Xcm,
                    n_patches=n_patches,
                    max_n_features=max_n_features))

    return hp_choice("feature_algo", exits)
