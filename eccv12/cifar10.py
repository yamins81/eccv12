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
    def shuffle(X, s):
        if shuffle_seed:
            np.random.RandomState(shuffle_seed + s).shuffle(X)
        else:
            return X

    return {
            'trn_images': shuffle(trn_images, 0),
            'trn_labels': shuffle(trn_labels, 0),
            'val_images': shuffle(val_images, 1),
            'val_labels': shuffle(val_labels, 1),
            'tst_images': shuffle(tst_images, 2),
            'tst_labels': shuffle(tst_labels, 2),
            }


def new_fbncc_layer(layer_num, X, n_patches, n_filters, size):
    def lab(msg):
        return 'l%i_fbncc_%s' % (layer_num, msg)

    patches = random_patches(X, n_patches, size, size,
            rng=np_RandomState(123 + layer_num))
    assert max_filters > 16

    remove_mean = hp_TF(lab('remove_mean'))
    beta = hp_lognormal(lab('beta'), np.log(100), np.log(100))
    hard_beta = hp_TF(lab('hard'))

    # TODO: use different nfilters, beta etc. for each algo

    # -- random projections filterbank allocation
    random_projections = partial(slm_fbncc,
        m_fb=slm_uniform_M_FB(
            nfilters=fb0_nfilters,
            size=size,
            rseed=hp_choice(lab('r_rseed'), range(1, 6)),
            normalize=hp_TF(lab('r_normalize')),
            ),
        remove_mean=remove_mean,
        beta=beta,
        hard_beta=hard_beta)

    # -- random whitened projections filterbank allocation
    random_whitened_projections = partial(slm_fbncc,
            m_fb=fb_whitened_projections(patches,
                patch_whitening_filterbank_X(patches,
                    gamma=logu_range(lab('rw_gamma'), 1e-4, 1.0),
                    o_ndim=2,
                    remove_mean=remove_mean,
                    beta=beta,
                    hard_beta=hard_beta,
                    ),
                n_filters=nfilters,
                rseed=hp_choice(lab('rw_rseed'), range(6, 11)),
                ),
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # -- whitened patches filterbank allocation
    whitened_patches = partial(slm_fbncc,
            m_fb=fb_whitened_patches(patches,
                patch_whitening_filterbank_X(patches,
                    gamma=logu_range(lab('wp_gamma'), 1e-4, 1.0),
                    o_ndim=2,
                    remove_mean=remove_mean,
                    beta=beta,
                    hard_beta=hard_beta,
                    ),
                n_filters=nfilters,
                rseed=hp_choice(lab('wp_rseed'), range(6, 11)),
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


def pipeline_extension(layer_num, X, n_patches, max_filters):
    f_layer = new_fbncc_layer(layer_num, X, n_patches,
            n_filters=s_int(
                hp_qloguniform(f('nfilters'),
                    np.log(8.01), np.log(max_filters), q=16)),
            size=rfilter_size('l%ifb_size' % layer_num, 3, 8),
            )

    p_layer = partial(slm_lpool,
            stride=hp_choice('l%ip_stride' % layer_num, [1, 2]),
            order=hp_choice('l%ip_order' % layer_num,
                [1, 2, hp_lognormal('l%ip_order_real' % layer_num,
                    mu=np.log(1), sigma=np.log(3))]),
            ker_size=rfilter_size('l%ip_size', 2, 8),
            ))

    return [f_layer, p_layer]


def pipeline_exits(pipeline, layer_num, X, n_patches, max_n_features):
    def lab(msg):
        return 'l%i_out_%s' % (layer_num, msg)

    rval = []

    fsize = rfilter_size(lab('fsize'), 3, 8)

    grid_res = hp_choice(lab('grid_res'), [2, 3])
    grid_features_per_filter = 2 * (grid_res ** 2)
    grid_nfilters = max_features // grid_features_per_filter

    grid_filtering = new_fbncc_layer(layer_num,
            X=X,
            n_patches=n_patches,
            nfilters=grid_nfilters,
            size=fsize,
            )

    grid_pooling = partial(slm_quantize_gridpool,
            alpha=hp_normal(lab('grid_alpha'), 0.0, 1.0),
            use_mid=False,
            grid_res=grid_res,
            order=hp_choice(lab('grid_order'), [
                1.0, 2.0, logu_range(lab('grid_order_real'), .1, 10.)]))

    rval.append(pipeline + [grid_filtering, grid_pooling])

    # -- now set up the lpool_alpha option
    filtering_res = Xrows - fsize + 1
    # -- N.B. Xrows depends on other params, so we can't use it to set the
    #         upper bound on lpsize. We can only sample independently, and
    #         then fail below with non-positive number of features.
    lpool_size = rfilter_size(lab('lpsize'), 1, 5)
    lpool_res = scope.max(filtering_res - lpool_size + 1, 0)
    lpool_nfilters = switch(lpool_res > 0,
            max_features // (2 * (lpool_res ** 2)),
            scope.Raise(ValueError, 'Non-positive number of features'))

    local_filtering = new_fbncc_layer(layer_num,
            X=X,
            n_patches=n_patches,
            nfilters=local_nfilters,
            size=fsize,
            )

    local_pooling = partial(slm_lpool_alpha,
            ker_size=lpool_size,
            alpha=hp_normal(lab('local_alpha'), 0.0, 1.0),
            grid_res=grid_res,
            order=hp_choice(lab('local_order'), [
                1.0, 2.0, logu_range(lab('local_order_real'), .1, 10.)]))

    rval.append(pipeline + [local_filtering, local_pooling])

    return rval


def choose_pipeline(X, n_patches, batchsize,
        max_n_features, max_layer_sizes, time_limit,
        memlimit=1000 * (1024 ** 2)):
    """
    This function works by creating a linear pipeline, with multiple exit
    points that could be the feature representation for classification.

    The function returns a switch among all of these exit points.
    """
    start_time = time.time()

    pipeline = []
    exits = pipeline_exits(
            pipeline,
            layer_num=0,
            X=X,
            n_patches=n_patches,
            max_n_features=max_n_features)
    for layer_i, max_layer_size in enumerate(max_layer_sizes):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise RuntimeError('pipeline creation is taking too long')
        extension = pipeline_extension(layer_i, X, n_patches, max_layer_size)

        pipeline.extend(extension)
        X = pyll_theano_batched_lmap(
                partial(callpipe1, extension),
                X,
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=memlimit / scope.len(X)
                ))[:]  # -- indexing computes all the values (during rec_eval)

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise RuntimeError('pipeline creation is taking too long')
        exits.extend(pipeline_exits(layer_i + 1, pipeline, X, max_n_features))

    return hp_switch("feature_algo", exits)


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
            ),
            (
                # -- this often doesn't work because Python can get screwed
                #    up by memory errors, but it's worth a try.
                lambda e: isinstance(e, MemoryError),
                lambda e: {
                    'loss': float('inf'),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }
            ),
            (
                # -- this is raised when computations are taking too long
                lambda e: (isinstance(e, RuntimeError)
                    and 'taking too long' in str(e)),
                lambda e: {
                    'loss': float('inf'),
                    'status': hyperopt.STATUS_FAIL,
                    'failure': repr(e)
                }
            ),
        ])
def cifar10bandit(
        n_train=40000,
        n_valid=10000,
        n_test=10000,
        batchsize=20,
        n_imgs_for_patches=10000,
        # -- maximum n. features per example coming out of the pipeline
        max_n_features=16000,
        # -- seconds allocated to pipeline creation
        #    (This includes processing time for computing patches)
        pipeline_timeout=60.0,
        # -- max n. filterbank elements going into another layer
        max_layer_sizes=[64, 128, 256],
        ):

    data = scope.cifar10_img_classification_task(
            dtype='uint8',
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            shuffle_seed=5)

    pipeline = choose_pipeline(
            X=data['trn_images'][:n_imgs_for_patches],
            n_patches=50000,
            max_out_features=max_n_features,
            )
    #print pipeline

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

    xstd_inc = hp_lognormal('classif_squash_lowvar', np.log(1e-4), np.log(1000))
    xstd = sqrt(xstd ** 2 + xstd_inc)

    trn_xy=((cache_train - xmean) / xstd, data['trn_labels'])
    val_xy = ((features['val'] - xmean) / xstd, data['val_labels'])
    tst_xy = ((features['tst'] - xmean) / xstd, data['tst_labels'])

    svm = fit_linear_svm(trn_xy,
            l2_regularization=hp_lognormal('l2_reg', np.log(1e-3), np.log(100)),
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

