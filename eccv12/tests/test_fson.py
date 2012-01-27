import cPickle
import numpy as np

from eccv12.fson import fson_print
from eccv12.fson import fson_eval
from eccv12.fson import register
from eccv12.fson import run_all

from eccv12.plugins import fetch_decisions
from eccv12.plugins import get_images
from eccv12.plugins import slm_memmap

@register()
def lfw_boosted_experiment(
        img_features,
        train_decisions,
        test_decisions):
    assert img_features.dtype == 'float32'
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
        fetch_decisions.son('a'),
        fetch_decisions.son('b'))
    fson_print(thing)
    print thing


def test_2():
    thing = lfw_boosted_experiment.son(
        slm_memmap.son(
            desc={},
            X=get_images.son()),
        fetch_decisions.son('DevTrain'),
        fetch_decisions.son('DevTest'),
        )

    print thing
    fson_print(thing)

def test_eval_0():
    thing = lfw_boosted_experiment.son(
        slm_memmap.son(
            desc={},
            X=get_images.son('float32'),
            name='asdf'),
        fetch_decisions.son('DevTrain'),
        fetch_decisions.son('DevTest'),
        )

    assert 'foo' == fson_eval(thing,
             scope=dict(
                 ctrl=CtrlStub()))

def test_eval_memo():
    calls = []
    def _dummy():
        calls.append(0)
    _dummy = register()(_dummy)

    aa = _dummy.son()
    thing = run_all.son(aa, aa)
    fson_eval(thing)
    assert len(calls) == 1
