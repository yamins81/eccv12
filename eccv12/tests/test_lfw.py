import eccv12.model_params as params
import pyll.stochastic as stochastic
import pyll
import hyperopt.base
import numpy as np
import eccv12.lfw as lfw
import eccv12.bandits as bandits
import eccv12.experiments as experiments

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


def test_fg11_top_bandit():
    L = lfw.FG11Bandit()
    config = stochastic.sample(L.template, np.random.RandomState(0))
    config['decisions'] = None
    config['slm'] = stochastic.sample(pyll.as_apply(params.fg11_top), np.random.RandomState(0))
    config['comparison'] = 'sqrtabsdiff'
    rec = L.evaluate(config, hyperopt.base.Ctrl())
    assert np.abs(rec['loss'] - .194) < 1e-2
    return rec
    

NUM_ROUNDS = 2
ROUND_LEN = 1

def test_mixture_ensembles():
    bandit = lfw.TestBandit()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(
            trials,
            bandit_algo,
            async=False)
    exp.run(NUM_ROUNDS * ROUND_LEN)
    results = trials.results
    specs = trials.specs

    simple = experiments.SimpleMixture(trials, bandit)
    simple_inds, simple_weights = simple.mix_models(NUM_ROUNDS)
    assert simple_inds.tolist() == [1, 0]
    
    ada = experiments.AdaboostMixture(trials, bandit)
    ada_inds, ada_weights = ada.mix_models(NUM_ROUNDS)
    assert  np.abs(ada_weights.reshape((2,)) - np.array([.3812, .1975])).max() < 1e-3
        
    selected_specs = {'simple': simple_specs,
                      'ada': ada_specs}

    #really need lfw view 2 methods to test this properly
    
    return exp, selected_specs
