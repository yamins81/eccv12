from nose.plugins.attrib import attr
import numpy as np

import eccv12.model_params as params
import pyll.stochastic as stochastic
import pyll
import hyperopt.base
import eccv12.lfw as lfw
import eccv12.bandits as bandits
import eccv12.experiments as experiments

test_params = {
    'slm': [[('lnorm', params.lnorm)]],
    'preproc': {
        'global_normalize': 0,
        'crop': params.crop_choice,
        'size': [20, 20]}}

config_tiny_rnd0 = stochastic.sample(
        test_params,
        np.random.RandomState(0))

config_tiny_rnd2 = stochastic.sample(
        test_params,
        np.random.RandomState(2))

def test_sampling():
    """
    Test that pyll samples the same way as when we ran these experiments.
    """
    print config_tiny_rnd0
    print config_tiny_rnd2
    assert config_tiny_rnd0 == {
            'preproc': {
                'global_normalize': 0,
                'crop': (25, 25, 175, 175),
                'size': (20, 20)},
            'slm': (
                (
                    ('lnorm', {
                        'kwargs': {
                            'inker_shape': (9, 9),
                            'outker_shape': (9, 9),
                            'remove_mean': 0,
                            'threshold': 0.1,
                            'stretch': 5.928446182250183}}),),)}
    assert config_tiny_rnd2 == {'preproc': {'global_normalize': 0, 'crop':
        (88, 63, 163, 188), 'size': (20, 20)}, 'slm': ((('lnorm', {'kwargs':
            {'inker_shape': (7, 7), 'outker_shape': (7, 7), 'remove_mean': 0,
                'threshold': 0.1, 'stretch': array(1.8508207817320688)}}),),)}

@attr('slow')# -- takes about 30 secs with GPU
def test_lfw_tiny_rnd0():
    # -- if this test is failing
    #    -- if test_sampling passes, then the screening program has changed.
    #    -- if test_sampling fails, then pyll sampling may well be to blame
    rec = lfw.get_performance(
            slm=config_tiny_rnd0['slm'],
            decisions=None,
            preproc=config_tiny_rnd0['preproc'],
            comparison='mult')
    print rec['test_accuracy']
    print sum(rec['test_errors'])
    assert np.allclose(rec['test_accuracy'], 58.8, atol=.1)
    assert sum(rec['test_errors']) == 412
    bandits.validate_result(rec)
    assert rec['label_set'] == [-1, 1]
    return rec


@attr('slow')# -- takes about 30 secs with GPU
def test_lfw_tiny_rnd2():
    # -- if this test is failing
    #    -- if test_sampling passes, then the screening program has changed.
    #    -- if test_sampling fails, then pyll sampling may well be to blame
    print config_tiny_rnd2
    rec = lfw.get_performance(
            slm=config_tiny_rnd2['slm'],
            decisions=None,
            preproc=config_tiny_rnd2['preproc'],
            comparison='mult')
    print rec['test_accuracy']
    print sum(rec['test_errors'])
    assert np.allclose(rec['test_accuracy'], 57.5, atol=.1)
    assert sum(rec['test_errors']) == 425
    bandits.validate_result(rec)
    assert rec['label_set'] == [-1, 1]
    return rec


def test_fg11_top_bandit():
    L = lfw.FG11Bandit()
    config = stochastic.sample(L.template, np.random.RandomState(0))
    config['decisions'] = None
    config['slm'] = stochastic.sample(pyll.as_apply(params.fg11_top), np.random.RandomState(0))
    config['comparison'] = 'sqrtabsdiff'
    rec = L.evaluate(config, hyperopt.base.Ctrl(None))
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
    assert list(simple_inds) == [1, 0]

    ada = experiments.AdaboostMixture(trials, bandit)
    ada_inds, ada_weights = ada.mix_models(NUM_ROUNDS)
    assert  np.abs(ada_weights.reshape((2,)) - np.array([.3812, .1975])).max() < 1e-3

    selected_specs = {'simple': simple_specs,
                      'ada': ada_specs}

    #really need lfw view 2 methods to test this properly

    return exp, selected_specs


def test_pyll_resampling():
    bandit = lfw.MainBandit()
    s0 = pyll.stochastic.sample(bandit.template, np.random.RandomState(0))
    s1 = pyll.stochastic.sample(bandit.template, np.random.RandomState(0))
    print s0['model']['slm'][-1][-1][-1]['kwargs']['stretch']
    print s1['model']['slm'][-1][-1][-1]['kwargs']['stretch']

    assert s0 == s1
