
import eccv12.model_params as params
import pyll.stochastic as stochastic
import pyll
import hyperopt.base
import numpy as np
import eccv12.lfw as lfw
import eccv12.bandits as bandits

tiny_template = pyll.as_apply(params.test_params)
config_tiny = stochastic.sample(tiny_template, np.random.RandomState(0))

def test_lfw_basic():
    rec = lfw.get_performance(config_tiny['slm'], None, config_tiny['preproc'], 'mult')
    assert np.abs(rec['test_accuracy'] - 68.23) < .1
    assert np.array(rec['test_errors']).astype(int).sum() == 317
    bandits.validate_result(rec)
    assert rec['label_set'] == [-1, 1]
    return rec

def test_lfw_bandit():
    L = lfw.TestBandit()
    config = stochastic.sample(L.template, np.random.RandomState(0))
    config['decisions'] = None
    rec = L.evaluate(config, hyperopt.base.Ctrl())
    assert np.abs(rec['test_accuracy'] - 69.80) < .1
    return rec

    
