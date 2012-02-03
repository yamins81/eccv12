import copy
import os

import numpy as np
import scipy.io

import genson
from genson import JSONFunction
import hyperopt
import skdata
from skdata import larray

from eccv12.plugins import screening_program

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


def test_classifier():
    prog = screening_program({}, 'sqrtabsdiff', 'asdf_l3_150fd_sqrtabsdiff')

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
    fn = genson.JSONFunction(fg11_prog['result'])

    ctrl = hyperopt.Ctrl()
    ctrl.attachments['decisions'] = dict(
        DevTrain=np.zeros(2200),
        DevTest=np.zeros(1000),
        )
    #print genson.dumps(fg11_prog, pretty_print=True)
    result = fn(ctrl=ctrl)
    print result
    assert result['test_accuracy'] > 81.0  # -- I just saw it score 81.7 (Feb 2012)


