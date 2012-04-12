"""
Some helpful commands from npinto March 16/2012 showing how to reproduce scores using the sclas project:


[10:27:59 AM EDT] Nicolas Pinto: % for i in `seq -w 01 10`; do python svm_ova_fromfilenames.py /share/datasets/LFW_FG11/lfw_view2_split_${i}.csv.
kernel.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.{sqrtabs_diff,mul,abs_diff,sq_diff}.mat -C 1e5 | 
grep accuracy; done | awk '{print $0; sum+=$7} END {print sum/NR}'                                              
Classification accuracy on test data (%): 83.3333333333
Classification accuracy on test data (%): 82.8333333333
Classification accuracy on test data (%): 84.0
B^[[AClassification accuracy on test data (%): 84.1666666667
Classification accuracy on test data (%): 85.8333333333
Classification accuracy on test data (%): 84.3333333333
Classification accuracy on test data (%): 83.1666666667
Classification accuracy on test data (%): 83.3333333333Classification accuracy on test data (%): 84.1666666667
Classification accuracy on test data (%): 85.5
84.0667
[10:32:36 AM EDT] Nicolas Pinto: for all models:
[10:32:37 AM EDT] Nicolas Pinto: for m in $(< /share/datasets/LFW_FG11/HT1_1_LFW_Models/ht_l3_top5_6917.txt); do echo model $m; for i in `seq -
w 01 10`; do python svm_ova_fromfilenames.py /share/datasets/LFW_FG11/lfw_view2_split_${i}.csv.kernel.ht1_1_l3__
${m}_gray.{sqrtabs_diff,mul,abs_diff,sq_diff}.mat -C 1e5 | grep accuracy; done | awk '{print $0; sum+=$7} END {p
rint "average:" sum/NR}'; done;
[10:39:09 AM EDT] Nicolas Pinto: ===
[10:39:10 AM EDT] Nicolas Pinto: % for i in `seq -w 01 10`; do python svm_ova_fromfilenames.py /share/datasets/LFW_FG11/lfw_view2_split_${i}.csv.kernel.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.{sqrtabs_diff,mul}.mat -C 1e5 | grep accuracy; done | awk '{print $0; sum+=$7} END {print sum/NR}'
Classification accuracy on test data (%): 83.1666666667
Classification accuracy on test data (%): 79.5
Classification accuracy on test data (%): 81.1666666667
Classification accuracy on test data (%): 83.1666666667
Classification accuracy on test data (%): 82.6666666667
Classification accuracy on test data (%): 81.3333333333
Classification accuracy on test data (%): 80.6666666667
Classification accuracy on test data (%): 77.8333333333
Classification accuracy on test data (%): 82.3333333333
Classification accuracy on test data (%): 80.6666666667
81.25
[10:41:33 AM EDT] Nicolas Pinto: sclas / kernel_generate_fromcsv.py
[10:43:19 AM EDT] Nicolas Pinto: % for m in $(< /share/datasets/LFW_FG11/HT1_1_LFW_Models/ht_l3_top5_6917.txt); do echo model $m; for i in `seq -
w 01 10`; do python svm_ova_fromfilenames.py /share/datasets/LFW_FG11/lfw_view2_split_${i}.csv.kernel.ht1_1_l3__
${m}_gray.{sqrtabs_diff,mul,abs_diff,sq_diff}.mat -C 1e5 | grep accuracy; done | awk '{print $0; sum+=$7} END {p
rint "average:" sum/NR}'; done;
model 150fd767e9d5d6822e414b6ae20d7da6ce9469fa
Classification accuracy on test data (%): 83.3333333333Classification accuracy on test data (%): 82.8333333333
Classification accuracy on test data (%): 84.0
Classification accuracy on test data (%): 84.1666666667
Classification accuracy on test data (%): 85.8333333333
Classification accuracy on test data (%): 84.3333333333
Classification accuracy on test data (%): 83.1666666667
Classification accuracy on test data (%): 83.3333333333
Classification accuracy on test data (%): 84.1666666667
Classification accuracy on test data (%): 85.5
average:84.0667
model d87123face6a91282d28a845dffb2e3e7328a669
Classification accuracy on test data (%): 83.8333333333
Classification accuracy on test data (%): 83.6666666667
Classification accuracy on test data (%): 83.5
Classification accuracy on test data (%): 84.6666666667
Classification accuracy on test data (%): 84.6666666667
Classification accuracy on test data (%): 81.1666666667
Classification accuracy on test data (%): 84.6666666667
Classification accuracy on test data (%): 84.3333333333
Classification accuracy on test data (%): 85.0Classification accuracy on test data (%): 83.3333333333
average:83.8833
model de2b6b2be72cb9e6f9af06f04c7a073544d5c9e3
Classification accuracy on test data (%): 83.5
Classification accuracy on test data (%): 85.0
Classification accuracy on test data (%): 82.8333333333
Classification accuracy on test data (%): 82.3333333333Classification accuracy on test data (%): 85.3333333333
Classification accuracy on test data (%): 80.6666666667
Classification accuracy on test data (%): 83.1666666667
Classification accuracy on test data (%): 83.5
Classification accuracy on test data (%): 84.0
Classification accuracy on test data (%): 83.1666666667
average:83.35
model 53976e98ffa38aaea63b26f6aba132a486606236
Classification accuracy on test data (%): 83.8333333333
Classification accuracy on test data (%): 82.5
Classification accuracy on test data (%): 83.8333333333
Classification accuracy on test data (%): 83.1666666667
Classification accuracy on test data (%): 81.5
Classification accuracy on test data (%): 81.8333333333
Classification accuracy on test data (%): 81.8333333333
Classification accuracy on test data (%): 81.5
Classification accuracy on test data (%): 82.0
Classification accuracy on test data (%): 81.0
average:82.3
model 0f1aff3b5033e9442244accc9acbe1db6327dfa3
Classification accuracy on test data (%): 81.8333333333
Classification accuracy on test data (%): 79.6666666667
Classification accuracy on test data (%): 86.0
Classification accuracy on test data (%): 83.6666666667
Classification accuracy on test data (%): 85.1666666667
Classification accuracy on test data (%): 80.0
Classification accuracy on test data (%): 82.1666666667
Classification accuracy on test data (%): 82.3333333333
Classification accuracy on test data (%): 84.1666666667
Classification accuracy on test data (%): 82.5
average:82.75
[10:44:08 AM EDT] Nicolas Pinto: /share/datasets/LFW_FG11/*csv
[10:44:23 AM EDT] Nicolas Pinto: /share/datasets/LFW_FG11/lfw_view2_*csv
[10:51:30 AM EDT] Nicolas Pinto: python kernel_generate_fromcsv.py /path/to/csv .ht1_1_l3_BLABLABLA.mat kernel.mat
[10:52:21 AM EDT] Nicolas Pinto: python kernel_generate_fromcsv.py
[10:56:19 AM EDT] Nicolas Pinto: cd pythor3/pythor3/model && nosetests -s
[10:57:41 AM EDT] Nicolas Pinto: ls ~/.pythor3/test_data_cache/model/slm/
[11:17:55 AM EDT] Nicolas Pinto: re
[11:18:05 AM EDT] Nicolas Pinto: found some fbs
[11:18:12 AM EDT] James Bergstra: awesome
[11:18:16 AM EDT] Nicolas Pinto: in ~/.pythor3/test_data_cache/model/slm/
[11:18:24 AM EDT] Nicolas Pinto: fn = '06b9e1a1b22f0b6ba9e361e37abbe3282c280803.pkl'
[11:18:33 AM EDT] Nicolas Pinto: then something like:
[11:18:34 AM EDT] Nicolas Pinto: In [35]: retina_size = pkl.load(open(fn))['init_args'][0]

In [36]: model_desc = pkl.load(open(fn))['init_args'][1]

In [37]: layer = 1

In [38]: operation = 0  # i.e. fbcorr

In [39]: fb = model_desc[layer][operation][1]['initialize']

In [40]: fb.shape
Out[40]: (64, 3, 3)
[11:19:03 AM EDT] Nicolas Pinto: In [44]: model_desc[layer][operation][1]['kwargs']
Out[44]: {'max_out': None, 'min_out': 0}
[11:19:18 AM EDT] Nicolas Pinto: In [45]: model_desc[layer][operation][0]
Out[45]: 'fbcorr'
[11:19:33 AM EDT] Nicolas Pinto: similar to the l3 model_desc you are used to
[11:19:44 AM EDT] Nicolas Pinto: except that 'initialize' now contains the actual filterbank


"""

import cPickle
import os
import sys
import time

from nose import SkipTest
import numpy as np
import scipy.io

import pyll
from skdata import larray
import sklearn.svm

from eccv12.classifier import train_scikits
from eccv12.lfw import Aligned
from eccv12.lfw import screening_program
from eccv12.lfw import verification_pairs
from eccv12.lfw import view2_fold_kernels_by_spec
from eccv12.lfw import get_view2_features
from eccv12.utils import mean_and_std, linear_kernel
from eccv12.utils import dot

feature_root = '/share/datasets/LFW_FG11/lfw2'


def only_on_honeybadger(f):
    def wrapper(*args, **kwargs):
        if os.path.exists('/share/datasets/LFW_FG11'):
            return f(*args, **kwargs)
        else:
            raise SkipTest()
    wrapper.__name__ = f.__name__
    return wrapper


def load_official_view2_dump(test_fold, comparison):
    sclas_comparison = {
        'mult': 'mul',
        'sqrtabsdiff': 'sqrtabs_diff',
        'absdiff': 'abs_diff',
        'sqdiff': 'sq_diff',
    }[comparison]

    fname = '/home/jbergstra/tmp/lfw_view2_split_%02i.csv.kernel.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.%s.mat.pkl' % (
        10 - test_fold,  # N.B. official splits are backward (!!)
        sclas_comparison)

    print 'LOADING OFFICIAL SAVED KERNEL:', fname

    rval = cPickle.load(open(fname))
    return rval


def closeish(msg, A, B, atol=.25):
    AmB = abs(A - B)
    print msg, AmB.max(), (AmB / (abs(A) + abs(B))).max(), AmB.mean(), (AmB / (abs(A) + abs(B))).mean()
    #print A[:3, :3]
    #print B[:3, :3]
    # -- these numbers are typically pretty big, but a few are close to zero,
    #    so relative error is as much as .5 but I think it's OK.
    if atol:
        assert AmB.max() < atol


@pyll.scope.define
def get_fg11_features(suffix, expected_shape):
    dataset = Aligned()
    paths, identities = dataset.raw_classification_task()
    def load_path(path):
        basename = os.path.basename(path)
        name = basename[:-9]  # cut off the digits and the .jpg
        # -- touch the jpg to make sure it's there
        new_path = os.path.join(
            feature_root,
            name,
            basename)
        feature_path = new_path + suffix
        print 'loading', feature_path
        data = scipy.io.loadmat(feature_path)['data']
        assert data.shape == expected_shape
        return np.asarray(data, dtype='float32')
    # -- apply decorator manually here in nested scope
    load_path = larray.lmap_info(
        shape=expected_shape,
        dtype='float32')(load_path)

    rval = larray.lmap(load_path, paths)
    rval = larray.cache_memmap(rval, 'fcache_' + suffix, basedir=os.getcwd())
    return rval


@only_on_honeybadger
def test_load_fg11_features():
    feat = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))
    print 'FEAT[0] =', feat[0]
    assert feat[0].shape == (100, 256)


@only_on_honeybadger
def test_fg11_view1_from_saved_features():
    # -- this tests the screening program from image features -> loss
    fg11_features = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))
    prog = screening_program({},
                             decisions=np.zeros((1, 3200)),
                             comparison='sqrtabsdiff',
                             preproc=None,
                             image_features=fg11_features,
                             namebase='asdf_l3_150fd_sqrtabsdiff')[1]

    result = pyll.rec_eval(prog['result']) # -- N.B. NO CLEANUP

    print result
    assert result['test_accuracy'] > 81.0  # -- I just saw it score 82.3 (Feb 2012)
    # March 16 2012 -- current codebase scores it 81.7  (loss .183)

@only_on_honeybadger
def test_sclas():
    foo = cPickle.load(open('/home/jbergstra/cvs/sclas/ofoo.mat.pkl'))
    for ii in range(10):
        foo2 = load_official_view2_dump(ii, 'absdiff')
        K1 = foo['kernel_traintrain']
        K2 = foo2['kernel_traintrain']
        closeish('ofoo dump matches official data', K1, K2, atol=None)
        print 'tr: %e %s' % (K1.trace(), K2.trace())
        print 'sum: %e %s' % (K1.sum(), K2.sum())
        print 'max: %e %s' % (K1.max(), K2.max())
        print 'min: %e %s' % (K1.min(), K2.min())
        print 'sum_abs: %e %s' % (abs(K1).sum(), abs(K2).sum())


    foo2 = load_official_view2_dump(9, 'absdiff')
    K1 = foo['kernel_traintrain']
    K2 = foo2['kernel_traintrain']
    print 'tr: %e %s' % (K1.trace(), K2.trace())
    print 'sum: %e %s' % (K1.sum(), K2.sum())

    train_y = [-1 if l == 'same' else 1 for l in foo['train_labels']]
    test_y = [-1 if l == 'same' else 1 for l in foo['test_labels']]

    assert np.all(foo['train_labels'] == foo2['train_labels'])
    assert np.all(foo['test_labels'] == foo2['test_labels'])

    svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-3, C=1e5)
    svm.fit(K1, train_y)
    print (svm.predict(foo['kernel_traintest'].T) != test_y).mean()
    print (svm.predict(foo2['kernel_traintest'].T) != test_y).mean()

    svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-3, C=1e5)
    svm.fit(K2, train_y)
    print (svm.predict(foo['kernel_traintest'].T) != test_y).mean()
    print (svm.predict(foo2['kernel_traintest'].T) != test_y).mean()

    closeish('ofoo dump matches official data', K1, K2)


@only_on_honeybadger
def test_view2_features():
    """
    Follow the computation of a feature kernel from start to finish
    """
    #foo = cPickle.load(open('/home/jbergstra/cvs/sclas/ofoo.mat.pkl'))
    foo = cPickle.load(open('/home/jbergstra/cvs/sclas/ofoo_nowhiten.pkl.mat.pkl'))
    ktrn_ref = foo['kernel_traintrain']

    rows_refA = cPickle.load(open('/home/jbergstra/cvs/sclas/ofoo.mat.A.mat'))
    rows_refB = cPickle.load(open('/home/jbergstra/cvs/sclas/ofoo.mat.B.mat'))

    feat = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))
    assert feat[0].shape == (100, 256)
    #dbname = 'fakedbFG11'
    #_id = 'from_saved'

    test_fold = 9
    split_data = [verification_pairs('fold_%d' % fold, subset=None, interleaved=True)
            for fold in range(10)]
    train_y = np.concatenate([split_data[_ind][2]
            for _ind in range(10) if _ind != test_fold])
    test_y = split_data[test_fold][2]

    image_features, pair_features_by_comp = get_view2_features(
            slm_desc=None,
            preproc=None,
            namebase=None,
            comparison=['absdiff'],
            basedir=os.getcwd(),
            image_features=feat,
            )
    pf_list = pair_features_by_comp['absdiff']


    train_X = np.vstack([pf_list[ii][:]
                         for ii in range(10) if ii != test_fold])
    assert np.allclose(train_X, rows_refA)
    fmean, fstd = mean_and_std(train_X)
    shift = -fmean
    scale = 1.0 / fstd
    
    train_X = (train_X + shift) * scale
    assert np.allclose(train_X, rows_refB)

    ktrn = linear_kernel(train_X, train_X, use_theano=True)
    ktrn2 = dot(train_X, train_X.T)
    closeish('ktrn ktrn2', ktrn, ktrn2)
    closeish('ktrn ktrn_ref', ktrn, ktrn_ref)
    closeish('ktrn2 ktrn_ref', ktrn2, ktrn_ref)


@only_on_honeybadger
def test_view2_saved_kernels():
    """
    Test that the saved kernels match the ones from sclas
    """
    ofoo = cPickle.load(open('/home/jbergstra/cvs/sclas/ofoo.mat.pkl'))
    for comparison in ['absdiff', 'mult', 'sqrtabsdiff', 'sqdiff']:
        train_names = ['fg11_top_ktrain%i_%s.npy' % (i, comparison)
                       for i in range(10)]
        test_names = ['fg11_top_ktest%i_%s.npy' % (i, comparison)
                      for i in range(10)]
        for test_fold in range(10):
            ktrn = np.load(train_names[test_fold])
            ktst = np.load(test_names[test_fold])

            foo = load_official_view2_dump(test_fold, comparison)
            closeish('%s %s train' % (comparison, test_fold), ktrn, foo['kernel_traintrain'])
            closeish('%s %s test' % (comparison, test_fold), ktst, foo['kernel_traintest'].T)
            closeish('%s %s train (ofoo)' % (comparison, test_fold), ktrn, ofoo['kernel_traintrain'])
            closeish('%s %s test (ofoo)' % (comparison, test_fold), ktst, ofoo['kernel_traintest'].T)



@only_on_honeybadger
def test_view2_from_saved_features():
    # -- this tests the evaluation program from image features -> loss
    fg11_features = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))

    dbname = 'fakedbFG11'
    _id = 'from_saved'

    # -- used for labels
    split_data = [verification_pairs('fold_%d' % fold, subset=None, interleaved=True)
            for fold in range(10)]

    #comparisons=['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff'],
    for comparison in ['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff']:
        train_names = ['fg11_top_ktrain%i_%s.npy' % (i, comparison)
                       for i in range(10)]
        test_names = ['fg11_top_ktest%i_%s.npy' % (i, comparison)
                      for i in range(10)]
        view2_fold_kernels_by_spec(None, dbname, _id,
                comparisons=[comparison],
                image_features=fg11_features,
                Ktrain_names = train_names,
                Ktest_names = test_names,
                force_recompute_kernel=True)

        test_errs = []

        for test_fold in range(10):
            Ktrain = np.load(train_names[test_fold])
            Ktest = np.load(test_names[test_fold])

            trace = Ktrain.trace()
            Ktrain /= trace
            Ktest /= trace

            train_y = np.concatenate([split_data[_ind][2]
                    for _ind in range(10) if _ind != test_fold])
            test_y = split_data[test_fold][2]

            svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-3, C=1e5)
            svm.fit(Ktrain, train_y)
            test_predictions = svm.predict(Ktest)
            test_err = (test_predictions != test_y).mean()
            print comparison, test_fold, 'accuracy', (1.0 - test_err) * 100
            test_errs.append(test_err)

        print comparison, 'MEAN:', (1.0 - np.mean(test_errs)) * 100


@only_on_honeybadger
def classify_eccv12_blend():
    # -- used for labels
    split_data = [verification_pairs('fold_%d' % fold, subset=None, interleaved=True)
            for fold in range(10)]

    test_errs = []
    for test_fold in range(10):
        ktrn_blend = ktst_blend = None
        for comparison in ['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff']:
            if 1:
                train_name = '/home/jbergstra/tmp/fg11_top_ktrain%i_%s.npy' % (test_fold, comparison)
                test_name = '/home/jbergstra/tmp/fg11_top_ktest%i_%s.npy' % (test_fold, comparison)
                Ktrain = np.load(train_name)
                Ktest = np.load(test_name)
            else:
                foo = load_official_view2_dump(test_fold, comparison)
                Ktrain = foo['kernel_traintrain']
                Ktest = foo['kernel_traintest'].T

            trace = Ktrain.trace()
            Ktrain /= trace
            Ktest /= trace

            if ktrn_blend is None:
                ktrn_blend = Ktrain
                ktst_blend = Ktest
            else:
                ktrn_blend += Ktrain
                ktst_blend += Ktest
        trace = ktrn_blend.trace()
        ktrn_blend /= trace
        ktst_blend /= trace

        train_y = np.concatenate([split_data[_ind][2]
                for _ind in range(10) if _ind != test_fold])
        test_y = split_data[test_fold][2]

        svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-3, C=1e5)
        svm.fit(ktrn_blend, train_y)
        test_predictions = svm.predict(ktst_blend)
        test_err = (test_predictions != test_y).mean()
        print test_fold, 'accuracy', (1.0 - test_err) * 100
        test_errs.append(test_err)

    print comparison, 'MEAN:', (1.0 - np.mean(test_errs)) * 100


@only_on_honeybadger
def test_abs_diff_kernel_01():
    dump_path = '%s/lfw_view2_split_01.csv.kernel.%s.sqrtabs_diff.mat.pkl' % (
        '/home/jbergstra/tmp',
        'ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray')
    kernel_mat = cPickle.load(open(dump_path))

    fg11_features = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))

    comparison = 'sqrtabsdiff'
    test_fold = 9 # csv splits are backward for some reason

    dbname = 'fakedbFG11'
    _id = 'from_saved'
    train_names = ['fg11_top_ktrain%i_%s.npy' % (test_fold, comparison)]
    test_names = ['fg11_top_ktest%i_%s.npy' % (test_fold, comparison)]
    view2_fold_kernels_by_spec(None, dbname, _id,
            comparisons=[comparison],
            image_features = fg11_features,
            Ktrain_names = train_names,
            Ktest_names = test_names,
            force_recompute_kernel=True,
            test_folds = [test_fold])

    ktrn = np.load(train_names[0])
    ktst = np.load(test_names[0])

    ktrn_ref = kernel_mat['kernel_traintrain']
    ktst_ref = kernel_mat['kernel_traintest'].T

    print ktrn[:3, :3]
    print ktrn_ref[:3, :3]
    #print ktst[:3, :3]
    #print ktst_ref[:3, :3]

    assert ktrn.shape == ktrn_ref.shape
    assert ktst.shape == ktst_ref.shape


@only_on_honeybadger
def test_splits():
    """
    This tests that the view2 splits are equivalent to the ones used by sclas.
    """
    dataset = Aligned()
    all_paths = np.array(dataset.raw_classification_task()[0])
    skd = []
    for j in range(10):
        lidxs, ridxs, matches = verification_pairs('fold_%i' % j, interleaved=True)
        skd_Ltest = map(os.path.basename, all_paths[lidxs])
        skd_Rtest = map(os.path.basename, all_paths[ridxs])
        # skd puts all matches first, then all mis-matches
        # csv has them match, mismatch, match, mismatch
        #csvlike_Ltest = np.vstack([skd_Ltest[:300], skd_Ltest[300:]]).T.flatten()
        #csvlike_Rtest = np.vstack([skd_Rtest[:300], skd_Rtest[300:]]).T.flatten()
        skd.append((skd_Ltest, skd_Rtest))

    for i in range(1, 11):
        csv_path = '/share/datasets/LFW_FG11/lfw_view2_split_%02i.csv' % i
        csv_lines = np.asarray([l.split(',') for l in open(csv_path)])
        assert len(csv_lines) == 6000, len(csv_lines)
        assert np.all(csv_lines[-600:, 3] == 'test\r\n')
        assert np.all(csv_lines[:-600, 3] == 'train\r\n')
        csv_Ltrain = map(os.path.basename, csv_lines[:-600, 0])
        csv_Rtrain = map(os.path.basename, csv_lines[:-600, 1])
        assert np.all(csv_Ltrain == np.concatenate([l
                    for j, (l, r) in enumerate(skd) if j != (10 - i)]))
        assert np.all(csv_Rtrain == np.concatenate([r
                    for j, (l, r) in enumerate(skd) if j != (10 - i)]))

        csv_Ltest = map(os.path.basename, csv_lines[-600:, 0])
        csv_Rtest = map(os.path.basename, csv_lines[-600:, 1])

        ll, rr = skd[-i]  # curiously, the order is backward..
        print i, (np.all(csv_Ltest == ll) and np.all(csv_Rtest == rr))
        assert (np.all(csv_Ltest == ll) and np.all(csv_Rtest == rr))


def classify_sclas_kernels(input_filenames):
    """
    To run this program I had to hack the sclas file for kernel
    classification to dump matlab files to pickle format, because
    scipy.io complained about matlab version mismatches and refused to
    load the original files.

    The weird setup of the procedure makes this not really a unit or
    regression test but the result was that the same numbers came out as
    came out of the sclas program. (March 16 2012)
    
    """

    blend_train = blend_test = None
    for fname in input_filenames:
        print 'LOADING', fname
        kernel_mat = cPickle.load(open(fname))
        ktrn = kernel_mat['kernel_traintrain']
        ktst = kernel_mat['kernel_traintest']
        ktrn_trace = ktrn.trace()
        print 'KTRN TRACE', ktrn_trace
        ktrn /= ktrn_trace
        ktst /= ktrn_trace

        if blend_train is None:
            blend_train = ktrn
            blend_test = ktst
        else:
            blend_train += ktrn
            blend_test += ktst

    trc = blend_train.trace()
    blend_train /= trc
    blend_test /= trc

    train_labels = [str(elt) for elt in kernel_mat["train_labels"]]
    test_labels = [str(elt) for elt in kernel_mat["test_labels"]]
    train_y = [1 if l == 'same' else -1 for l in train_labels]
    test_y = [1 if l == 'same' else -1 for l in test_labels]

    print 'TRAIN_KERNEL VAR',
    print 'shape', blend_train.shape, blend_test.shape,
    print 'var', blend_train.var()

    C = 1e5
    svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-3, C=C)
    svm.fit(blend_train, train_y)
    test_predictions = svm.predict(blend_test.T)
    test_err = (test_predictions != test_y).mean()
    print 'Accuracy', (1.0 - test_err) * 100


if __name__ == '__main__':
    sys.exit(test_asdf(sys.argv[1:]))

