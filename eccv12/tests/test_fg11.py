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

feature_root = '/share/datasets/LFW_FG11/lfw2'


def only_on_honeybadger(f):
    def wrapper(*args, **kwargs):
        if os.path.exists('/share/datasets/LFW_FG11'):
            return f(*args, **kwargs)
        else:
            raise SkipTest()
    wrapper.__name__ = f.__name__
    return wrapper


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
def test_fg11_view2_from_saved_features():
    # -- this tests the evaluation program from image features -> loss
    fg11_features = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))


    config = {}
    config['decisions'] = None
    config['comparison'] = None
    config['preproc'] = None
    config['slm'] = None

    dbname = 'fakedbFG11'
    _id = 'from_saved'

    # -- used for labels
    split_data = [verification_pairs('fold_%d' % fold, subset=None)
            for fold in range(10)]

    #comparisons=['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff'],
    for comparison in ['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff']:
        train_names = ['fg11_top_ktrain%i_%s.npy' % (i, comparison)
                       for i in range(10)]
        test_names = ['fg11_top_ktest%i_%s.npy' % (i, comparison)
                      for i in range(10)]
        view2_fold_kernels_by_spec(config, dbname, _id,
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

            svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-4, C=1e5)
            svm.fit(Ktrain, train_y)
            test_predictions = svm.predict(Ktest)
            test_err = (test_predictions != test_y).mean()
            print test_fold, 'accuracy', (1.0 - test_err) * 100
            test_errs.append(test_err)

        print comparison, 'MEAN:', (1.0 - np.mean(test_errs)) * 100


@only_on_honeybadger
def test_fg11_view2_blend_from_saved_features():
    # -- used for labels
    split_data = [verification_pairs('fold_%d' % fold, subset=None)
            for fold in range(10)]

    comparisons = ['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff']
    #comparisons = ['sqrtabsdiff']
    test_errs = []
    for test_fold in range(10):

        blend_train = blend_test = None
        for comparison in comparisons:
            train_names = ['fg11_top_ktrain%i_%s.npy' % (i, comparison)
                           for i in range(10)]
            test_names = ['fg11_top_ktest%i_%s.npy' % (i, comparison)
                          for i in range(10)]

            Ktrain = np.load(train_names[test_fold]).astype('float64')
            Ktest = np.load(test_names[test_fold]).astype('float64')

            train_trace = Ktrain.trace()
            Ktrain /= train_trace
            Ktest /= train_trace

            if blend_train is None:
                blend_train = Ktrain
                blend_test = Ktest
            else:
                blend_train += Ktrain
                blend_test += Ktest
            del Ktrain, Ktest

        trc = blend_train.trace()
        blend_train /= trc
        blend_test /= trc

        train_y = np.concatenate([split_data[_ind][2]
                for _ind in range(10) if _ind != test_fold])
        test_y = split_data[test_fold][2]

        print 'TRAIN_KERNEL VAR', blend_train.var()

        for log10C in 5,:
            C = 10 ** log10C
            svm = sklearn.svm.SVC(kernel='precomputed', scale_C=False, tol=1e-3, C=C)
            svm.fit(blend_train, train_y)
            test_predictions = svm.predict(blend_test)
            test_err = (test_predictions != test_y).mean()
            print test_fold, C, 'accuracy', (1.0 - test_err) * 100
            test_errs.append(test_err)

        print C, 'MEAN:', (1.0 - np.mean(test_errs)) * 100, comparisons


def test_splits():
    import os
    dataset = Aligned()
    all_paths = np.array(dataset.raw_classification_task()[0])
    skd = []
    for j in range(10):
        lidxs, ridxs, matches = verification_pairs('fold_%i' % j)
        skd_Ltest = map(os.path.basename, all_paths[lidxs])
        skd_Rtest = map(os.path.basename, all_paths[ridxs])
        # skd puts all matches first, then all mis-matches
        # csv has them match, mismatch, match, mismatch
        csvlike_Ltest = np.vstack([skd_Ltest[:300], skd_Ltest[300:]]).T.flatten()
        csvlike_Rtest = np.vstack([skd_Rtest[:300], skd_Rtest[300:]]).T.flatten()
        skd.append((csvlike_Ltest, csvlike_Rtest))

    csv = []
    for i in range(1, 11):
        csv_path = '/share/datasets/LFW_FG11/lfw_view2_split_%02i.csv' % i
        csv_lines = np.asarray([l.split(',') for l in open(csv_path)])
        assert len(csv_lines) == 6000, len(csv_lines)
        assert np.all(csv_lines[-600:, 3] == 'test\r\n')
        assert np.all(csv_lines[:-600, 3] == 'train\r\n')
        csv_Ltest = map(os.path.basename, csv_lines[-600:, 0])
        csv_Rtest = map(os.path.basename, csv_lines[-600:, 1])
        csv.append((csv_Ltest, csv_Rtest))

    for i, (l, r) in enumerate(skd):
        ll, rr = csv[-i - 1]
        print i, (np.all(l == ll) and np.all(r == rr))


def test_can_classify_saved_kernels(input_filenames):

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

