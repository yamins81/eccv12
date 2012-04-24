import cPickle
import sys
import time

import numpy as np

import skdata.lfw
from skdata import larray

import pyll
from pyll import scope

import hyperopt
from hyperopt.pyll_utils import hp_choice

import pyll_slm
import comparisons
from .utils import ImgLoaderResizer
from .slm import choose_pipeline

# -- pyll.scope import
pyll.scope.import_(globals(),
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
    'np_RandomState',
    'print_ndarray_summary',
    'pickle_dumps',
    )

_Aligned = None
def Aligned():
    global _Aligned
    if _Aligned is None:
        _Aligned = skdata.lfw.Aligned()
    return _Aligned



@scope.define
def lfw_aligned_images(dtype, preproc):
    """
    Return a lazy array whose elements are all the images in lfw.

    XXX: Should the images really be returned in greyscale?

    preproc : a dictionary with keys:
        global_normalize - True / False
        size - (height, width)
        crop - (l, t, r, b)

    """

    all_paths = Aligned().raw_classification_task()[0]
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


@scope.define_info(o_len=3)
def lfw_verification_pairs(split, subset=None, interleaved=False):
    """
    Return three integer arrays: lidxs, ridxs, match.

    lidxs: position in the given split of the left image
    ridxs: position in the given split of the right image
    match: 1 if they correspond to the same people, else -1
    """
    if not interleaved:
        import sys
        print >> sys.stderr, "WARNING: not interleaved verification pairs"
    dataset = Aligned()
    all_paths = dataset.raw_classification_task()[0]
    lpaths, rpaths, matches = dataset.raw_verification_task(split=split)
    if interleaved and split.startswith('fold'):
        lpaths = np.vstack([lpaths[:300], lpaths[300:]]).T.flatten()
        rpaths = np.vstack([rpaths[:300], rpaths[300:]]).T.flatten()
        matches = np.vstack([matches[:300], matches[300:]]).T.flatten()
    elif interleaved and split == 'DevTrain':
        assert len(lpaths) == 2200
        lpaths = np.vstack([lpaths[:1100], lpaths[1100:]]).T.flatten()
        rpaths = np.vstack([rpaths[:1100], rpaths[1100:]]).T.flatten()
        matches = np.vstack([matches[:1100], matches[1100:]]).T.flatten()
    elif interleaved and split == 'DevTest':
        assert len(lpaths) == 1000
        lpaths = np.vstack([lpaths[:500], lpaths[500:]]).T.flatten()
        rpaths = np.vstack([rpaths[:500], rpaths[500:]]).T.flatten()
        matches = np.vstack([matches[:500], matches[500:]]).T.flatten()
    elif interleaved:
        raise ValueError(split)

    lidxs, ridxs = _verification_pairs_helper(all_paths, lpaths, rpaths)
    if subset is None:
        return lidxs, ridxs, (matches * 2 - 1)
    elif isinstance(subset, int):
        return lidxs[:subset], ridxs[:subset],  (matches[:subset] * 2 - 1)
    else:
        assert all([isinstance(_t, int) for _t in subset])
        return lidxs[subset], ridxs[subset], (matches[subset] * 2 - 1)


class PairFeaturesFn(object):
    """
    larray.lmap-friendly implementation of the comparison function
    that maps tensor pairs (of image features) to a 1-D flattened
    feature vector.
    """

    # -- mimic how function objects are named
    __name__  = 'PairFeaturesFn'

    def __init__(self, X, fn_name, timelimit):
        self.X = X
        self.fn = getattr(comparisons, fn_name)
        self.n_calls = 0
        self.t_total = 0.0
        self.fn_name = fn_name
        self.timelimit = timelimit

    def rval_getattr(self, attr, objs):
        if attr == 'shape':
            return (np.prod(self.X.shape[1:]),)
        if attr == 'ndim':
            return 1
        if attr == 'dtype':
            return self.X.dtype
        raise AttributeError()

    def __call__(self, li, ri):
        t0 = time.time()
        self.n_calls += 1
        if 0 == (self.n_calls % 100):
            print ('PairFeatureFn{%s} %i calls in %f seconds' % (self.fn_name,
                self.n_calls, self.t_total))
        lx = self.X[li]
        rx = self.X[ri]
        rval = self.fn(lx, rx)
        t1 = time.time()
        self.t_total += t1 - t0
        if self.timelimit:
            if (self.n_calls > 100
                    and (self.t_total / self.n_calls > self.timelimit)):
                raise RuntimeError("pair features taking too long",
                        (self.t_total, self.n_calls, self.timelimit))
        return rval


@scope.define_info(o_len=2)
def cache_feature_pairs(pair_labels, X, comparison_name, pair_timelimit):
    """
    pair_labels    - something like comes out of verification_pairs
    X              - feature vectors to be combined
    combination_fn - some lambda X[i], X[j]: features1D
    """
    lidxs, ridxs, matches = pair_labels
    pf = larray.lmap(
            PairFeaturesFn(X, comparison_name, timelimit=pair_timelimit),
            lidxs,
            ridxs)
    pf_cache = larray.cache_memory(pf)
    return pf_cache, np.asarray(matches)


@scope.define_info(o_len=2)
def get_decisions(ctrl):
    # TODO
    print >> sys.stderr, "TODO: retrieve decisions from database"
    return 0, 0

@scope.define_info(o_len=2)
def worth_calculating_view2(ctrl, loss, thresh_rank=5):
    """
    Return True iff `loss` might be in the stop `thresh_rank` trials at the
    end of the experiment.
    """
    if ctrl is None:
        # -- this is the debugging scenario
        return True
    else:
        if loss > .4:
            # -- terrible performance, just bail.
            return False
        trials = ctrl.trials
        query = {
                'result.status': hyperopt.STATUS_OK,
                'exp_key': trials._exp_key,
                }
        docs = list(trials.handle.jobs.find(query,
            {'result.loss': 1}))
        losses = [d['result']['loss'] for d in docs]
        losses.sort()
        rank = np.searchsorted(losses, loss)
        print 'worth_calculating_view2: loss ranked', rank, 'among', len(losses)
        if len(losses) == 0 or rank < thresh_rank:
            return True
        return False


@scope.define
def lfw_view2_results(data, pipeline, result, solver):
    """ """

    cmps = list(data[0].keys())
    cmps.sort()

    train_errs = []
    test_errs = []
    attachments = {}

    for ind in range(10):
        train_inds = range(10)
        del train_inds[ind]
        print ('Constructing stuff for split %d ...' % ind)
        test_y = data[ind][cmps[0]][1]
        train_y = np.concatenate([data[ii][cmps[0]][1] for ii in train_inds])

        test_X = np.hstack([data[ind][cc][0][:] for cc in cmps])
        train_X = np.hstack([
            np.vstack([data[ii][cc][0][:] for ii in train_inds])
                for cc in cmps])

        xmean, xstd = pyll_slm.mean_and_std(train_X,
                remove_std0=pipeline['remove_std0'])
        pyll_slm.print_ndarray_summary('Xmean', xmean)
        pyll_slm.print_ndarray_summary('Xstd', xstd)
        xstd = np.sqrt(xstd ** 2 + pipeline['varthresh'])

        train_X -= xmean
        train_X /= xstd
        test_X -= xmean
        test_X /= xstd

        print ('Training svm %d ...' % ind)
        svm = pyll_slm.fit_linear_svm(
                [train_X, train_y],
                verbose=True,
                l2_regularization=pipeline['l2_reg'],
                solver=solver,
                )

        train_predictions = svm.predict(train_X)
        test_predictions = svm.predict(test_X)
        train_err = (train_predictions != train_y).mean()
        test_err = (test_predictions != test_y).mean()

        print 'split %d train err %f' % (ind, train_err)
        print 'split %d test err %f' % (ind, test_err)
        train_errs.append(train_err)
        test_errs.append(test_err)

        attachments['view2_trn_%i.npy.pkl' % ind] = cPickle.dumps(
                    np.asarray(train_predictions, dtype='float32'),
                    protocol=-1)
        attachments['view2_tst_%i.npy.pkl' % ind] = cPickle.dumps(
                    np.asarray(test_predictions, dtype='float32'),
                    protocol=-1)
        del svm

    train_err_mean = np.mean(train_errs)
    print 'train err mean', train_err_mean
    test_err_mean = np.mean(test_errs)
    print 'test err mean', test_err_mean

    assert 'view2' not in result
    result['view2'] = {
            'train_err_mean': train_err_mean,
            'test_err_mean': test_err_mean,
            'train_errs': train_errs,
            'test_errs': test_errs,
            }
    result['true_loss'] = test_err_mean
    result['attachments'].update(attachments)

    return result

@scope.define
def do_view2_if_promising(result, ctrl, devtst_erate, view2_xy, pipeline,
        solver):
    if worth_calculating_view2(ctrl, devtst_erate):
        # write the loss to the database now before embarking on view2, so
        # that the optimizer can use that information.
        if ctrl:
            temp_result = dict(result,
                    loss=devtst_erate,
                    status=hyperopt.STATUS_OK)
            del temp_result['attachments']
            ctrl.checkpoint(temp_result)
        return lfw_view2_results(view2_xy, pipeline, result, solver)
    else:
        return result


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
def lfw_bandit(
        n_train=2200,
        n_test=1000,
        n_view2_per_split=600,
        batchsize=1,
        n_patches=50000,
        n_imgs_for_patches=1000,
        max_n_features=16000,
        max_layer_sizes=[64, 128],
        pipeline_timeout=90.0,
        pair_timelimit=20 * 60 / 3200.0, # -- seconds per image pair
        svm_solver=('asgd.BinarySubsampledTheanoOVA', {
                'dtype': 'float32',
                'verbose': 0,  # -1 is silent, 0 is terse, 1 is verbose
                'bfgs_factr': 1e8,
                'feature_bytes': 500 * (1024 ** 2),  # unit: bytes
                }),
        screen_comparison='sqrtabsdiff',
        ):
    """
    High-throughput screening setup for classification of Aligned LFW images.
    """
    ctrl = hyperopt.Bandit.pyll_ctrl
    decisions_devtrn, decisions_devtst = scope.get_decisions(ctrl)

    # -- add the decisions_devtrn to the svm_solver kwargs
    svm_solver_name, svm_solver_kwargs = svm_solver
    svm_solver_kwargs = scope.dict(svm_solver_kwargs, decisions=decisions_devtrn)
    svm_solver = pyll.as_apply([svm_solver_name, svm_solver_kwargs])

    images = scope.lfw_aligned_images(
            dtype='float32',
            preproc=hp_choice('preproc',
                [
                    {
                    'global_normalize': 0,
                    'size': [1, 200, 200],
                    'crop': [0, 0, 250, 250],
                    },
                    # XXX: verify cropping logic in sclas, add crops here.
                ]),
            )

    trn_l, trn_r, trn_m = scope.lfw_verification_pairs('DevTrain',
            subset=n_train,
            interleaved=True)

    pipeline = choose_pipeline(
            Xcm=scope.asarray(images[trn_l[:n_imgs_for_patches]]),
            n_patches=n_patches,
            batchsize=batchsize,
            max_n_features=max_n_features,
            max_layer_sizes=max_layer_sizes,
            time_limit=pipeline_timeout,
            )
    #print pipeline

    ### XXX Why are pair features sorted? if they are interlevaed?

    image_features = scope.larray_cache_memory(
            pyll_theano_batched_lmap(
                partial(callpipe1, pipeline['pipe']),
                images,
                batchsize=batchsize,
                print_progress=100,
                abort_on_rows_larger_than=max_n_features,
                ))

    def pairs_dataset(split, comparison_name, subset=None):
        return scope.cache_feature_pairs(
            scope.lfw_verification_pairs(split=split,
                subset=subset,
                interleaved=True),
            image_features,
            comparison_name=comparison_name,
            pair_timelimit=pair_timelimit)

    result = {}

    devtrn_x, devtrn_y = pairs_dataset('DevTrain', screen_comparison, n_train)
    devtst_x, devtst_y = pairs_dataset('DevTest', screen_comparison, n_test)
    devtrn_x = devtrn_x[:]  # compute all

    xmean, xstd = mean_and_std(devtrn_x, remove_std0=pipeline['remove_std0'])
    xmean = print_ndarray_summary('Xmean', xmean)
    xstd = print_ndarray_summary('Xstd', xstd)
    xstd = sqrt(xstd ** 2 + pipeline['varthresh'])

    devtrn_xy = ((devtrn_x - xmean) / xstd, devtrn_y)
    devtst_xy = ((devtst_x - xmean) / xstd, devtst_y)

    svm = scope.fit_linear_svm(
            devtrn_xy,
            verbose=True,
            l2_regularization=pipeline['l2_reg'],
            solver=svm_solver,
            )

    devtrn_d = model_decisions(svm, devtrn_xy[0]) + decisions_devtrn
    devtst_d = model_decisions(svm, devtst_xy[0]) + decisions_devtst

    devtrn_pred = model_predict(svm, devtrn_xy[0])
    devtst_pred = model_predict(svm, devtst_xy[0])

    devtrn_erate = error_rate(devtrn_pred, devtrn_xy[1])
    devtst_erate = error_rate(devtst_pred, devtst_xy[1])

    result = {
            'status': hyperopt.STATUS_OK,
            # -- criterion to optimize
            'loss': devtst_erate,
            # -- other error rates
            'trn_erate': devtrn_erate,
            'tst_erate': devtst_erate,
            # -- larger stats to save
            'attachments': {
                'devtrn_d.npy.pkl': scope.pickle_dumps(
                    scope.asarray(devtrn_d, dtype='float32'),
                    protocol=-1),
                'devtst_d.npy.pkl': scope.pickle_dumps(
                    scope.asarray(devtst_d, dtype='float32'),
                    protocol=-1),
                }
            }

    # -- VIEW2 stuff
    view2_xy = {}
    for fold in range(10):
        fold_vp = scope.lfw_verification_pairs(split='fold_%i' % fold,
            subset=n_view2_per_split,
            interleaved=True)
        for comparison in ['mult', 'sqdiff', 'sqrtabsdiff', 'absdiff']:
            # if we could get view1 extracted in time, then assume it's fast
            # enough here. Don't want view2 results cancelled by a delay that
            # puts one feature calculation over the limit.
            pd = scope.cache_feature_pairs(fold_vp, image_features,
                comparison_name=comparison, pair_timelimit=None)
            view2_xy.setdefault(fold, {}).setdefault(comparison, pd)

    if 1:
        # -- this is done without using switch to work around the
        # cycle-creating bug
        result = scope.do_view2_if_promising(result,
                                         ctrl, devtst_erate, view2_xy,
                                         pipeline, svm_solver)
    else:
        # --  the switch here triggers cycle-creation in
        # hyperopt.vectorize_helper.
        result = scope.switch(
            scope.worth_calculating_view2(ctrl, devtst_erate),
            result,
            scope.lfw_view2_results(view2_xy, pipeline, result))

    return result
