import numpy as np
import pyll
import eccv12.lfw as lfw
from eccv12 import model_params


def test_pyll_resampling():
    bandit = lfw.MainBandit()
    for seed in range(20):
        s0 = pyll.stochastic.sample(bandit.template,
                np.random.RandomState(seed))
        s1 = pyll.stochastic.sample(bandit.template,
                np.random.RandomState(seed))
        s2 = pyll.stochastic.sample(bandit.template,
                np.random.RandomState(seed + 1))
        assert s0 == s1
        assert s0 != s2


def test_pyll_param_func_valid():

    # -- test that the pyll_param_func returns interchangable things
    #    and that those things deliver samples predictably
    pf0 = model_params.pyll_param_func()
    pf1 = model_params.pyll_param_func()
    sample = pyll.stochastic.sample

    for seed in range(20):
        s00 = sample(pf0, np.random.RandomState(seed))
        s10 = sample(pf1, np.random.RandomState(seed))
        s01 = sample(pf0, np.random.RandomState(seed + 1))
        s11 = sample(pf1, np.random.RandomState(seed + 1))
        assert s00 == s10
        assert s01 == s11
        assert s00 != s01


def test_pyll_lockdown():
    pf0 = model_params.pyll_param_func()
    sample = pyll.stochastic.sample
    s = sample(pf0, np.random.RandomState(123))
    # -- this was done with commit
    # f11026e235e57f85e9dd4f3a23a67dac2b33e8db
    print s
    assert s == {'preproc':
            {'global_normalize': 0, 'crop': (0, 0, 250, 250), 'size':
        (100, 100)}, 'slm': ((('lnorm', {'kwargs': {'inker_shape': (8.0, 8.0),
            'outker_shape': (8.0, 8.0), 'remove_mean': 1, 'threshold':
            0.11155144268807565, 'stretch': 0.10625644175250697}}),),
            (('fbcorr', {'initialize': {'n_filters': 16.0, 'filter_shape':
                (8.0, 8.0), 'generate': ('random:uniform', {'rseed': 1})},
                'kwargs': {}}), ('lpool', {'kwargs': {'ker_shape': (4.0, 4.0),
                    'order': 2.1093609092062242, 'stride': 2}}), ('lnorm',
                        {'kwargs': {'inker_shape': (7.0, 7.0), 'outker_shape':
                            (7.0, 7.0), 'remove_mean': 0, 'threshold':
                            0.87658837006354096, 'stretch':
                            0.47462712500820414}})), (('fbcorr',
                                {'initialize': {'n_filters': 16.0,
                                    'filter_shape': (8.0, 8.0), 'generate':
                                    ('random:uniform', {'rseed': 12})},
                                    'kwargs': {}}), ('lpool', {'kwargs':
                                        {'ker_shape': (6.0, 6.0), 'order':
                                            5.2378168706681931, 'stride':
                                            2}}), ('lnorm', {'kwargs':
                                                {'inker_shape': (8.0, 8.0),
                                                    'outker_shape': (8.0,
                                                        8.0), 'remove_mean':
                                                    0, 'threshold':
                                                    3.8851479053533247,
                                                    'stretch':
                                                    0.22825379389892816}})))}
