import copy
import os

from nose import SkipTest
import numpy as np
import scipy.io

from thoreano.slm import SLMFunction
import genson
from genson import JSONFunction
import hyperopt
import skdata
from skdata import larray

from eccv12.plugins import screening_program
from eccv12 import model_params, plugins

feature_root = '/share/datasets/LFW_FG11/lfw2'


@genson.lazy
def get_fg11_features(suffix, expected_shape):
    dataset = skdata.lfw.Aligned()
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



def test_fg11_features():
    feat = get_fg11_features(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))
    print 'FEAT[0] =', feat[0]
    assert feat[0].shape == (100, 256)


def test_classifier_from_fg11_saved_features():
    if not os.path.isdir(feature_root):
        raise SkipTest('no lfw2 features - skipping FG11 regression test')
    prog = screening_program({},
                             comparison='sqrtabsdiff',
                             preproc=None,
                             namebase='asdf_l3_150fd_sqrtabsdiff')[1]

    image_features = prog['image_features']

    fg11_features = get_fg11_features.lazy(
        '.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray.mat',
        expected_shape=(100, 256))

    fg11_prog = copy.deepcopy(prog,
                              memo={
                                  id(image_features): fg11_features,
                                  id(JSONFunction.ARGS): JSONFunction.ARGS,
                                  id(JSONFunction.KWARGS): JSONFunction.KWARGS,
                              })
    fn = genson.JSONFunction(fg11_prog['result_w_cleanup'])

    ctrl = hyperopt.Ctrl()
    ctrl.attachments['decisions'] = dict(
        DevTrain=np.zeros(2200),
        DevTest=np.zeros(1000),
        )
    #print genson.dumps(fg11_prog, pretty_print=True)
    result = fn(ctrl=ctrl)
    print result
    assert result['test_accuracy'] > 81.0  # -- I just saw it score 82.3 (Feb 2012)


def test_imgs():
    root = '/home/bergstra/cvs/eccv12/eccv12/tests/data/'
    saved_paths = np.load(os.path.join(root, 'fg11_Xraw0-4.npy'))
    saved_processed_imgs = np.load(os.path.join(root, 'fg11_X0-4.npy'))

    imgs = plugins.get_images('float32', None)

    print saved_paths
    print saved_processed_imgs.shape

    assert saved_processed_imgs[0].shape == imgs[0].shape

    # the saved_preprocessed_imgs were designed to include only images
    # that appeared in view1, so it is normal that some images are omitted.
    assert np.allclose(saved_processed_imgs[0], imgs[0])
    assert np.allclose(saved_processed_imgs[1], imgs[1])
    assert np.allclose(saved_processed_imgs[2], imgs[3])

    if 0:
        from skdata.utils.glviewer import glumpy_viewer, command, glumpy
        glumpy_viewer(
                #img_array=saved_processed_imgs,
                img_array=imgs,
                arrays_to_print=[saved_paths],
                cmap=glumpy.colormap.Grey)

class FG11TopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.fg11_top,
            comparison='sqrtabsdiff',
            preproc={'global_normalize': 0},
            )

def test_fg11top():

    bandit = FG11TopBandit()
    config = bandit.template.sample(1)
    ctrl = hyperopt.Ctrl()
    ctrl.attachments['decisions'] = dict(
        DevTrain=np.zeros(2200),
        DevTest=np.zeros(1000),
        )
    # progkey == result means no cleanup memmaps
    result = bandit.evaluate(config, ctrl, progkey='result')

    print result['train_accuracy']
    print result['test_accuracy']
    print result['loss']
    # 100.0
    # 74.7
    # 0.253


def test_fg11top_mult():

    bandit = FG11TopBandit()
    config = bandit.template.sample(1)
    config['comparison'] = 'mult'
    ctrl = hyperopt.Ctrl()
    ctrl.attachments['decisions'] = dict(
        DevTrain=np.zeros(2200),
        DevTest=np.zeros(1000),
        )
    result = bandit.evaluate(config, ctrl)

    print result['train_accuracy']
    print result['test_accuracy']
    print result['loss']
    #    Pinto didn't screen on this, so it's not clear if these scores are
    #    good or bad
    # -- 100.0
    # -- 68.0
    # -- 0.32


class CVPRTopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.cvpr_top,
            comparison='mult',
            preproc={'global_normalize':1},
            )

data_root = '/home/bergstra/cvs/eccv12/eccv12/tests/data/'

def test_cvprtop_img_features():
    saved = np.load(os.path.join(data_root, 'fg11_features0-4.npy'))

    imgs = plugins.get_images('float32', None)  #--larray
    desc = copy.deepcopy(model_params.cvpr_top,
                memo={
                    id(model_params.null): None,
                    },)
    feat_fn = SLMFunction(desc, imgs.shape[1:])
    print saved[0].shape
    print feat_fn(imgs[0]).shape
    for ii, jj in (0, 0), (1, 1), (2, 3):
        assert np.allclose(saved[ii], feat_fn(imgs[jj]))

def test_cvprtop_features_all():
    filename='features_df448700aa91cef4c8bc666c75c393776a210177_0.dat'
    saved = np.memmap(os.path.join(data_root, filename),
            dtype='float32', mode='r',
            shape=(4992, 16, 16, 256))

    desc = copy.deepcopy(model_params.cvpr_top,
                memo={
                    id(model_params.null): None,
                    },)
    image_features = plugins.slm_memmap(
                desc=desc,
                X=plugins.get_images('float32', preproc=None),
                name='cvprtop_features_all_img_feat')
    vpairs_train = plugins.verification_pairs('DevTrain')
    vpairs_test = plugins.verification_pairs('DevTest')

    train_X, train_y = plugins.pairs_memmap(vpairs_train, image_features, 'mult', 'wtf_train')
    test_X, test_y =  plugins.pairs_memmap(vpairs_test , image_features, 'mult', 'wtf_test')

    # -- evaluate the whole set of pairs
    train_X = np.asarray(train_X)
    test_X = np.asarray(test_X)

    # -- check that there are 4992 valid entries in the image_features memmap
    #    and that our features match the saved ones

    print np.sum(image_features._valid)
    assert np.sum(image_features._valid) == 4992
    jj = 0
    for ii in range(4992):
        if image_features._valid[ii]:
            assert np.allclose(image_features._data[ii], saved[jj])
            jj += 1

    # -- check that our pair features match the saved ones
    saved_train_pairs_X = np.memmap(
            os.path.join(data_root,
                'train_pairs_df448700aa91cef4c8bc666c75c393776a210177.dat'),
            dtype='float32', mode='r',
            shape=(2200, 65536))
    saved_test_pairs_X = np.memmap(
            os.path.join(data_root,
                'test_pairs_df448700aa91cef4c8bc666c75c393776a210177.dat'),
            dtype='float32', mode='r',
            shape=(1000, 65536))

    assert np.allclose(train_X, saved_train_pairs_X)
    assert np.allclose(test_X, saved_test_pairs_X)

    train_d = np.zeros(len(train_y), dtype='float32')
    test_d  = np.zeros(len(test_y),  dtype='float32')

    train_Xyd_n, test_Xyd_n = plugins.normalize_Xcols(
        (train_X, train_y, train_d,),
        (test_X, test_y, test_d,))

    svm = plugins.train_svm(train_Xyd_n, l2_regularization=1e-3)

    new_d_train = plugins.svm_decisions(svm, train_Xyd_n)
    new_d_test = plugins.svm_decisions(svm, test_Xyd_n)

    result = plugins.result_binary_classifier_stats(
            train_Xyd_n,
            test_Xyd_n,
            new_d_train,
            new_d_test,
            result={})
    print 'Train_accuracy', result['train_accuracy']
    print 'Test accuracy', result['test_accuracy']
    print 'loss', result['loss'], np.sqrt(result['loss'] * (1 -
        result['loss']) / (len(test_y) - 1))

def test_cvprtop():
    bandit = CVPRTopBandit()
    config = bandit.template.sample(1)
    ctrl = hyperopt.Ctrl()
    ctrl.attachments['decisions'] = dict(
        DevTrain=np.zeros(2200),
        DevTest=np.zeros(1000),
        )
    result = bandit.evaluate(config, ctrl)

    print result['train_accuracy']
    print result['test_accuracy']
    print result['loss']
    # 100.0
    # 81.4
    # 0.186

