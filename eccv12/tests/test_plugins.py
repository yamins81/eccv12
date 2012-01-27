import cPickle
import numpy as np

#from eccv12.fson import fson_print, fson_eval, register

#from eccv12.lfw import fetch_train_decisions
#from eccv12.lfw import fetch_test_decisions
#from eccv12.lfw import lfw_images
#from eccv12.lfw import slm_memmap
import eccv12.lfw
from eccv12.fson import fson_print


class CtrlStub(object):
    def get_attachment(self, name):
        if name == 'train_decisions':
            return cPickle.dumps(np.zeros(100)) # XXX: return right number
        if name == 'test_decisions':
            return cPickle.dumps(np.zeros(100)) # XXX: return right number
        raise KeyError(name)


def test_print_screening_program():
    fson_print(eccv12.lfw.screening_program(
        slm_desc='slm_desc',
        comparison='sumsqdiff'))


def test_get_images():
    X = eccv12.lfw.get_images()
    # XXX: are these faces supposed to be greyscale?
    assert X.shape == (13233, 250, 250, 3), X.shape


def test_verification_pairs():
    lfw._verification_pairs_helper

