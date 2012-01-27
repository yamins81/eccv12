import cPickle
import numpy as np

#from eccv12.fson import fson_print, fson_eval, register

#from eccv12.plugins import fetch_train_decisions
#from eccv12.plugins import fetch_test_decisions
#from eccv12.plugins import lfw_images
#from eccv12.plugins import slm_memmap
from eccv12 import plugins
from eccv12.fson import fson_print


class CtrlStub(object):
    def get_attachment(self, name):
        if name == 'train_decisions':
            return cPickle.dumps(np.zeros(100)) # XXX: return right number
        if name == 'test_decisions':
            return cPickle.dumps(np.zeros(100)) # XXX: return right number
        raise KeyError(name)


def test_print_screening_program():
    fson_print(plugins.screening_program(
        slm_desc='slm_desc',
        comparison='sumsqdiff'))


def test_get_images():
    X = plugins.get_images(dtype='float32')
    # XXX: are these faces supposed to be greyscale?
    assert X.dtype == 'float32'
    assert X.shape == (13233, 250, 250, 3), X.shape


def test_verification_pairs_0():
    l, r = plugins._verification_pairs_helper(
        ['a', 'b', 'z', 'c'],
        ['b', 'z', 'a'],
        ['c', 'b', 'z'])
    assert list(l) == [1, 2, 0]
    assert list(r) == [3, 1, 2]

def test_verification_pairs_1():
    l, r, m = plugins.verification_pairs('DevTest')
    print set(m)
    # XXX: is there really just 1000 verification pairs in DevTest??
    # That's so hard for doing screening!?
    assert len(l) == len(r) == len(m) == 1000
    assert set(m) == set([-1, 1])

