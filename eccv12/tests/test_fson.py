import cPickle
import numpy as np

from eccv12.fson import fson_print, fson_eval, register


@register(call_w_scope=True)
def fetch_train_decisions(ctrl):
    return cPickle.loads(ctrl.get_attachment('train_decisions'))


@register(call_w_scope=True)
def fetch_test_decisions(ctrl):
    return cPickle.loads(ctrl.get_attachment('test_decisions'))


@register()
def lfw_images(split):
    import skdata.lfw
    return skdata.lfw.Aligned().img_verification_task(split=split)[0]

@register()
def lfw_labels(split):
    import skdata.lfw
    return skdata.lfw.Aligned().img_verification_task(split=split)[1]


@register()
def slm_memmap(desc, X):
    from thoreano.slm import SLMFunction
    from skdata import larray
    feat_fn = SLMFunction(desc, X.shape[1:])
    rval = larray.lmap(feat_fn, X)
    return rval


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
