import unittest
from nose.plugins.attrib import attr
import numpy as np

from thoreano.slm import InvalidDescription
import eccv12.model_params as params
import pyll.stochastic as stochastic
import pyll
import hyperopt
import hyperopt.base
import eccv12.lfw as lfw
import eccv12.bandits as bandits
import eccv12.experiments as experiments
import os

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

def test_sampling_lockdown():
    """
    Test that pyll samples the same way as when we ran these experiments.
    """
    # done with commit
    # f11026e235e57f85e9dd4f3a23a67dac2b33e8db
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
                'threshold': 0.1, 'stretch': 1.8508207817320688}}),),)}


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


@attr('slow')# -- takes about 30 secs with GPU
def test_fg11_top_bandit():
    L = lfw.FG11Bandit()
    config = stochastic.sample(L.template, np.random.RandomState(0))
    config['decisions'] = None
    config['slm'] = stochastic.sample(pyll.as_apply(params.fg11_top),
            np.random.RandomState(0))
    config['comparison'] = 'sqrtabsdiff'
    rec = L.evaluate(config, hyperopt.base.Ctrl(None))
    assert np.abs(rec['loss'] - .194) < 1e-2
    return rec


@attr('slow')# -- takes about 30 secs with GPU
def test_mixture_ensembles():
    """
    Test that LFW bandit can be used by the Mixtures code
    """
    NUM_ROUNDS = 2
    ROUND_LEN = 1

    bandit = lfw.TestBandit()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(
            trials,
            bandit_algo,
            async=False)
    exp.run(NUM_ROUNDS * ROUND_LEN)

    simple = experiments.SimpleMixture(trials, bandit)
    simple_specs, simple_weights = simple.mix_models(NUM_ROUNDS)

    ada = experiments.AdaboostMixture(trials, bandit, test_mask=True)
    ada_specs, ada_weights = ada.mix_models(NUM_ROUNDS)

    print simple_specs
    print ada_specs
    print ada_weights
    # -- in this simple case adaboost and simple should return the same models
    #    in the same order
    assert simple_specs == ada_specs
    assert np.allclose(ada_weights, [[0.29856626], [0.1672368]], atol=1e-3)

    selected_specs = {'simple': simple_specs,
                      'ada': ada_specs}

    #really need lfw view 2 methods to test this properly

    return exp, selected_specs


@attr('slow')
def test_main_bandit():
    L = lfw.MainBandit()
    config = stochastic.sample(L.template, np.random.RandomState(999))
    print config['model']['slm'][0]
    assert np.allclose(
            config['model']['slm'][0][0][1]['kwargs']['threshold'],
            16.894686245715995)
    rec = L.evaluate(config, hyperopt.base.Ctrl(None))
    print rec['test_accuracy']
    print rec['train_accuracy']
    print rec['loss']
    assert np.allclose(rec['test_accuracy'], 58.3)
    assert np.allclose(rec['train_accuracy'], 84.9090909091)
    assert np.allclose(rec['loss'], 0.417)
    return rec


@attr('slow')
def test_multi_bandit():
    bandit = lfw.MultiBandit()
    algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(trials, algo)
    exp.catch_bandit_exceptions = False
    exp.run(1)
    assert len(trials) > 1
    docs = trials.trials
    assert docs[0]['state'] == docs[1]['state'] == hyperopt.JOB_STATE_DONE
    assert docs[0]['spec']['model'] == docs[1]['spec']['model']
    assert docs[0]['spec']['comparison'] != docs[1]['spec']['comparison']

    assert docs[0]['misc']['idxs'] == docs[1]['misc']['idxs']
    assert docs[0]['misc']['vals'] != docs[1]['misc']['vals']

    assert docs[0]['result']['status'] == docs[1]['result']['status']
    assert docs[0]['result']['loss'] != docs[1]['result']['loss']

    # -- this is admittedly a little f'd up but the current code makes it like
    # this and it's not that bad. The trouble is that the attachments are computed
    # before the newly inserted job is even created, so they must be attached to
    # the original job. XXX fix this.
    ctrl = hyperopt.Ctrl(trials, docs[0])
    assert 'packed_normalized_DevTrain_kernel_mult' in ctrl.attachments
    assert 'normalized_DevTrainTest_kernel_mult' in ctrl.attachments
    assert 'packed_normalized_DevTrain_kernel_sqrtabsdiff' in ctrl.attachments
    assert 'normalized_DevTrainTest_kernel_sqrtabsdiff' in ctrl.attachments


class ForInts(object):
    def test_1(self):
        self.forint(1)

    def test_2(self):
        self.forint(2)

    @attr('slow')
    def test_many(self):
        for seed in range(100, 150):
            self.forint(seed)


def fuzz_config(config):
    print config
    imgs = lfw.get_images('float32', preproc=config['preproc'])
    try:
        mm = lfw.slm_memmap(config['slm'], imgs,
                name='_test_thoreano_fuzz')
    except InvalidDescription:
        return

    # -- evaluate a single example
    a = mm[0]

    # -- evaluate a batch
    b = mm[10:14]

    # --re-evaluate
    ra = mm[0]
    rb = mm[10:14]

    assert np.all(a == ra)
    assert np.all(b == rb)

    lfw.delete_memmap(mm)


class TestLNormFuzz(unittest.TestCase, ForInts):
    def forint(self, seed):
        config = stochastic.sample(
                dict(
                    slm=[[params.pf_lnorm()]],
                    preproc=params.pf_preproc()[1]),
                np.random.RandomState(seed))
        fuzz_config(config)


class TestParamFuncFuzz(unittest.TestCase, ForInts):
    def forint(self, seed):
        L = lfw.MainBandit()
        config = stochastic.sample(L.template,
                np.random.RandomState(seed))
        fuzz_config(config['model'])


@attr('slow') #takes about 30 sec with cpu
def test_baby_view2():
    c = config_tiny_rnd0
    lfw.get_view2_features(c['slm'], c['preproc'], 'mult', '', os.getcwd(),
                           test=50)
    return lfw.train_view2([''],[os.getcwd()], test=50)


@attr('slow') #takes about 30 sec with cpu
def test_baby_view2_libsvm():
    c = config_tiny_rnd0
    test_set = range(20) + range(500, 520)
    lfw.get_view2_features(c['slm'], c['preproc'], 'mult', 'libsvm', os.getcwd(),
                           test=test_set)
    return lfw.train_view2(['libsvm'],[os.getcwd()],
                           test=test_set, use_libsvm=True)


@attr('slow') #takes about 30 sec with cpu
def test_baby_view2_libsvm_kernel():
    c = config_tiny_rnd0
    test_set = range(20) + range(500, 520)
    lfw.get_view2_features(c['slm'], c['preproc'], 'mult', 'libsvm', os.getcwd(),
                           test=test_set)
    return lfw.train_view2(['libsvm'],[os.getcwd()],
                           test=test_set, use_libsvm={'kernel': 'precomputed'})