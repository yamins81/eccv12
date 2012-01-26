import cPickle
import numpy as np

from eccv12.fson import fson_print, fson_eval, register

from eccv12.lfw import fetch_train_decisions
from eccv12.lfw import fetch_test_decisions
from eccv12.lfw import lfw_images
from eccv12.lfw import slm_memmap

@register()
def lfw_boosted_experiment(
        img_features,
        train_decisions,
        test_decisions):
    assert img_features.dtype == 'float32'
    assert len(train_decisions) == 100
    assert len(test_decisions) == 100
    return 'foo'


class CtrlStub(object):
    def get_attachment(self, name):
        if name == 'train_decisions':
            return cPickle.dumps(np.zeros(100)) # XXX: return right number
        if name == 'test_decisions':
            return cPickle.dumps(np.zeros(100)) # XXX: return right number
        raise KeyError(name)


def test_0():
    fson_print(lfw_boosted_experiment.son(6))


def test_1():
    thing = lfw_boosted_experiment.son(
        fetch_train_decisions.son(),
        fetch_test_decisions.son())
    fson_print(thing)
    print thing


def test_2():
    thing = lfw_boosted_experiment.son(
        slm_memmap.son(
            desc={},
            X=lfw_images.son()),
        fetch_train_decisions.son(),
        fetch_test_decisions.son())

    print thing
    fson_print(thing)

def test_eval_0():
    thing = lfw_boosted_experiment.son(
        slm_memmap.son(
            desc={},
            X=lfw_images.son(
                split='DevTrain')),
        fetch_train_decisions.son(),
        fetch_test_decisions.son(),
        )

    assert 'foo' == fson_eval(thing,
             scope=dict(
                 ctrl=CtrlStub()))
