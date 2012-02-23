import sys
import copy
import threading
import time
import numpy as np
import pyll

import hyperopt
from hyperopt.tests.test_mongoexp import with_mongo_trials
from hyperopt.tests.test_mongoexp import with_worker_threads

import eccv12.eccv12 as exps
import eccv12.experiments as experiments
from eccv12.bandits import BaseBandit
from nose.plugins.attrib import attr

try:
    from collections import OrderedDict
except ImportError:
    print "Python 2.7+ OrderedDict collection not available"
    try:
        from ordereddict import OrderedDict
        warn("Using backported OrderedDict implementation")
    except ImportError:
        raise ImportError("Backported OrderedDict implementation "
                          "not available. To install it: "
                          "'pip install -vUI ordereddict'")

@with_mongo_trials
def test_mixture_initializes(trials):
    S = exps.MixtureExp(experiments.AdaboostMixture,
                        {'test_mask': True},
                        5,
                        10,
                        exps.LFWBandit,
                        hyperopt.Random,
                        "localhost:22334/test_hyperopt",
                        "test_stuff")

    assert S.get_info() == OrderedDict([('bandit', 'eccv12.eccv12.LFWBandit'),
                            ('num_features', 10),
                            ('bandit_algo', 'hyperopt.base.Random'),
                            ('mixture', 'eccv12.experiments.AdaboostMixture'),
                            ('mixture_kwargs', {'test_mask': True}),
                            ('ensemble_size', 5)])

    assert S.get_exp_key() == "test_stuffbandit:eccv12.eccv12.LFWBandit_num_features:10_bandit_algo:hyperopt.base.Random_mixture:eccv12.experiments.AdaboostMixture_mixture_kwargs:{'test_mask': True}_ensemble_size:5"
    S = exps.MixtureExp(experiments.SimpleMixture,
                        {},
                        5,
                        10,
                        exps.LFWBandit,
                        hyperopt.Random,
                        "localhost:22334/test_hyperopt",
                        "test_stuff")


@with_mongo_trials
def test_meta_initializes(trials):
    S = exps.MetaExp(experiments.AsyncBoostingAlgo,
                    {"round_len":5, "look_back":1},
                    10,
                    exps.LFWBandit,
                    hyperopt.Random,
                    "localhost:22334/test_hyperopt",
                    "test_stuff")

    assert S.get_info() == OrderedDict([('bandit', 'eccv12.eccv12.LFWBandit'),
                 ('num_features', 10),
                 ('meta_algo', 'eccv12.experiments.AsyncBoostingAlgo'),
                 ('bandit_algo', 'hyperopt.base.Random'),
                 ('meta_kwargs', {'look_back': 1, 'round_len': 5})])


@with_mongo_trials
def test_search_initializes(trials):
    S = exps.SearchExp(10,
                       exps.LFWBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")

    assert S.get_info() == OrderedDict([('bandit', 'eccv12.eccv12.LFWBandit'),
               ('num_features', 10), ('bandit_algo', 'hyperopt.base.Random')])


class DummyDecisionsBandit(BaseBandit):
    param_gen = dict(
            seed=pyll.scope.randint(1000),
            decisions=None)
    fail_prob = 0
    time_delay = 0

    def __init__(self, n_features):
        BaseBandit.__init__(self)
        self.n_features = n_features

    def delay(self):
        time_delay = self.time_delay
        time.sleep(time_delay)

    def performance_func(self, config, ctrl):
        r34 = np.random.RandomState(34)
        y = np.sign(r34.randn(self.n_features))
        rs = np.random.RandomState(int(config['seed']))
        yhat = rs.normal(loc=0, scale=1, size=(1, self.n_features))
        decisions = config['decisions']
        if decisions is None:
            decisions = np.zeros((1, self.n_features))
        else:
            decisions = np.array(decisions)
        new_dec = yhat + decisions
        is_test = np.ones(decisions.shape)

        self.delay() #random time delay, 0 by default

        #sporadic failures, none by default
        fail_no = np.random.RandomState(config['seed']).uniform()
        print('fail_no', fail_no)
        fail = fail_no < self.fail_prob
        if fail:
            result = dict(
                    status=hyperopt.STATUS_FAIL,
                    labels=None,
                    decisions=None,
                    is_test=None)
        else:
            result = dict(
                    loss=np.mean(y != np.sign(new_dec)),
                    labels=y.tolist(),
                    decisions=new_dec.tolist(),
                    is_test=is_test.tolist())
        return result


class FailureDummyDecisionsBandit(DummyDecisionsBandit):
    fail_prob = 0.2


class HighFailureDummyDecisionsBandit(DummyDecisionsBandit):
    fail_prob = 0.5


@attr('mongo')
@attr('medium')
@with_mongo_trials
@with_worker_threads(3, 'test_hyperopt', timeout=5.0)
def test_search_dummy(trials):
    S = exps.SearchExp(10,
                       DummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
    S.delete_all()
    S.run(10)
    assert len(S.trials.results) == 10 #make sure number of jobs have been run
    assert 1 > np.mean([x['loss'] for x in S.trials.results]) > 0
    T = copy.deepcopy(S.trials.results)
    S.run(20)
    assert len(S.trials.results) == 20 #make sure right # of jobs have been run
    assert all([t == s for t, s in zip(T, S.trials.results[:10])])


@attr('mongo')
@attr('medium')
@with_mongo_trials
@with_worker_threads(3, 'test_hyperopt', timeout=5.0)
def test_search_dummy_failure(trials):
    S = exps.SearchExp(10,
                       FailureDummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
    S.delete_all()
    S.run(10)
    #make sure failure has caused 2 additional trials
    assert len(S.trials) == 12 


@attr('mongo')
@attr('medium')
@with_mongo_trials
@with_worker_threads(3, 'test_hyperopt', timeout=5.0)
def test_search_dummy_failure_highprob(trials):
    S = exps.SearchExp(10,
                       HighFailureDummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
    S.delete_all()
    S.run(10)
    #assert higher error prob has caused many more failures
    assert len(S.trials) == 21


@attr('mongo')
@attr('medium')
@with_mongo_trials
@with_worker_threads(3, 'test_hyperopt', timeout=5.0)
def test_search_dummy_failure_highprob_walltime_cutoff(trials):
    S = exps.SearchExp(10,
                       HighFailureDummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff",
                       walltime_cutoff=0)
    S.delete_all()
    S.run(10)
    #assert that even with a lot of failures, a 0 walltime_cutoff
    #means that all status=fail trials are still counted
    assert len(S.trials) == 10


@attr('mongo')
@attr('medium')
@with_mongo_trials
@with_worker_threads(3, 'test_hyperopt', timeout=5.0)
def test_mix_dummy(trials):
    S = exps.MixtureExp(experiments.AdaboostMixture,
                        {'test_mask': True},
                        5,
                        10,
                       DummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
    S.delete_all()
    S.run(20)
    res = S.get_result()
    assert len(res['mixture_inds']) == 5
    assert res['mixture_weights'].shape == (5, 1)


@attr('mongo')
@attr('medium')
@with_mongo_trials
@with_worker_threads(3, 'test_hyperopt', timeout=5.0)
def test_meta_dummy(trials):
    S = exps.MetaExp(experiments.SyncBoostingAlgo,
                    {"round_len": 5},
                    10,
                   DummyDecisionsBandit,
                   hyperopt.Random,
                   "localhost:22334/test_hyperopt",
                   "test_stuff")
    S.delete_all()
    S.run(10)
    assert len(S.trials.results) == 10
    selected = S.bandit_algo.boosting_best_by_round(S.trials, S.bandit)
    assert len(selected) == 2
    T = copy.deepcopy(S.trials.results)
    S.run(20)
    assert len(S.trials.results) == 20
    assert all([t == s for t, s in zip(T, S.trials.results[:10])])
    selected2 = S.bandit_algo.boosting_best_by_round(S.trials, S.bandit)
    assert len(selected2) == 4
    assert selected2[:2] == selected


@attr('slow')
@attr('mongo')
@with_mongo_trials
# -- I tried timeout of 10, and 1/3 workers timed out mid-experiment
@with_worker_threads(3, 'test_hyperopt', timeout=15.0)
def test_budget_experiment(trials):
    S = exps.BudgetExperiment(ntrials=4,
                       save=False,
                       num_features=10,
                       ensemble_sizes=[2],
                       bandit_func=DummyDecisionsBandit,
                       bandit_algo_class=hyperopt.Random,
                       exp_prefix='test_stuff',
                       mongo_opts='localhost:22334/test_hyperopt',
                       look_back=1,
                       run_parallel=False)
    assert S.get_info() == {'control': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 10), ('bandit_algo', 'hyperopt.base.Random')]),
 'fixed_features_2': {'ada_mix': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 5), ('bandit_algo', 'hyperopt.base.Random'), ('mixture', 'eccv12.experiments.AdaboostMixture'), ('mixture_kwargs', {'test_mask': True}), ('ensemble_size', 2)]),
  'asyncboost': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 5), ('meta_algo', 'eccv12.experiments.AsyncBoostingAlgo'), ('bandit_algo', 'hyperopt.base.Random'), ('meta_kwargs', {'look_back': 1, 'round_len': 4})]),
  'basic': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 5), ('bandit_algo', 'hyperopt.base.Random')]),
  'simple_mix': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 5), ('bandit_algo', 'hyperopt.base.Random'), ('mixture', 'eccv12.experiments.SimpleMixture'), ('mixture_kwargs', {}), ('ensemble_size', 2)]),
  'syncboost': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 5), ('meta_algo', 'eccv12.experiments.SyncBoostingAlgo'), ('bandit_algo', 'hyperopt.base.Random'), ('meta_kwargs', {'round_len': 4})])},
 'fixed_trials_2': {'ada_mix': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 10), ('bandit_algo', 'hyperopt.base.Random'), ('mixture', 'eccv12.experiments.AdaboostMixture'), ('mixture_kwargs', {'test_mask': True}), ('ensemble_size', 2)]),
  'asyncboost': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 10), ('meta_algo', 'eccv12.experiments.AsyncBoostingAlgo'), ('bandit_algo', 'hyperopt.base.Random'), ('meta_kwargs', {'look_back': 1, 'round_len': 2})]),
  'basic': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 10), ('bandit_algo', 'hyperopt.base.Random')]),
  'simple_mix': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 10), ('bandit_algo', 'hyperopt.base.Random'), ('mixture', 'eccv12.experiments.SimpleMixture'), ('mixture_kwargs', {}), ('ensemble_size', 2)]),
  'syncboost': OrderedDict([('bandit', 'test_eccv12_experiments.DummyDecisionsBandit'), ('num_features', 10), ('meta_algo', 'eccv12.experiments.SyncBoostingAlgo'), ('bandit_algo', 'hyperopt.base.Random'), ('meta_kwargs', {'round_len': 2})])}}
    S.delete_all()
    S.run()
    res = S.get_result()
    assert res.keys() == ['control', 'fixed_trials_2', 'fixed_features_2']
    assert res['control'].keys() == ['bandit', 'num_features', 'bandit_algo', 'trials']
    assert len(res['control']['trials']) == 4
    assert len(res['fixed_trials_2']['basic']['trials']) == 4
    assert len(res['fixed_features_2']['basic']['trials']) == 8

