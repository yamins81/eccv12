"""
Testing experiment classes
"""
import unittest
import nose

import numpy as np
import hyperopt
import pyll
import eccv12.experiments as experiments
from eccv12.toyproblem import BoostableDigits
from eccv12.bandits import BaseBandit


###########Experiment Sketch
def Experiment(*args, **kwargs):
    rval = hyperopt.Experiment(*args, **kwargs)
    rval.catch_bandit_exceptions = False
    return rval

NUM_ROUNDS = 5
BASE_NUM_FEATURES = 50
ROUND_LEN = 5

class DummyDecisionsBandit(BaseBandit):
    param_gen = dict(
            seed=pyll.scope.randint(1000),
            decisions=None)

    def performance_func(self, config, ctrl):
        r34 = np.random.RandomState(34)
        y = np.sign(r34.randn(100))
        rs = np.random.RandomState(config['seed'])
        yhat = rs.randn(100)
        decisions = config['decisions']
        if decisions is None:
            decisions = np.zeros((1, 100))
        new_dec = yhat + decisions
        result = dict(
                status=hyperopt.STATUS_OK,
                loss=np.mean(y != np.sign(new_dec)),
                labels=y,
                is_test=[[1] * 20 + [0] * 80],
                decisions=new_dec)
        return result


class FastBoostableDigits(BoostableDigits):
    param_gen = dict(BoostableDigits.param_gen)
    param_gen['svm_max_observations'] = 5000 # -- smaller value for speed


class LargerBoostableDigits(FastBoostableDigits):
    param_gen = dict(FastBoostableDigits.param_gen)
    param_gen['feat_spec']['n_features'] = BASE_NUM_FEATURES


class NormalBoostableDigits(FastBoostableDigits):
    param_gen = dict(FastBoostableDigits.param_gen)
    param_gen['feat_spec']['n_features'] = BASE_NUM_FEATURES / ROUND_LEN


class ForAllBoostinAlgos(object):
    """Mixin tests that call self.work(<BoostingAlgoCls>)
    """

    def test_sync(self):
        self.work(experiments.SyncBoostingAlgo)

    def test_asyncA(self):
        self.work(experiments.AsyncBoostingAlgoA)

    def test_asyncB(self):
        self.work(experiments.AsyncBoostingAlgoB)


@nose.SkipTest
class TestBoostingLossDecreases(unittest.TestCase, ForAllBoostinAlgos):
    def work(self, cls):
        n_trials = 12
        round_len = 3
        trials = hyperopt.Trials()
        calls = [0]
        bandit = FastBoostableDigits()
        class FakeAlgo(hyperopt.Random):
            def suggest(self, ids, specs, results, miscs):
                calls[0] += 1
                assert len(specs) <= round_len
                return hyperopt.Random.suggest(self,
                        ids, specs, results, miscs)
        algo = FakeAlgo(bandit)
        boosting_algo = experiments.SyncBoostingAlgo(algo,
                    round_len=round_len)
        exp = Experiment(
                trials,
                boosting_algo,
                async=False)

        last_mixture_score = float('inf')

        round_ii = 0
        while len(trials) < n_trials:
            round_ii += 1
            exp.run(round_len)
            assert len(trials) == round_ii * round_len
            assert calls[0] == round_ii * round_len
            best_trials = boosting_algo.best_by_round(list(trials))
            assert len(best_trials) == round_ii
            train_errs, test_errs = bandit.score_mixture_partial_svm(best_trials)
            print test_errs # train_errs, test_errs
            assert test_errs[-1] < last_mixture_score
            last_mixture_score = test_errs[-1]


class TestBoostingSubAlgoArgs(unittest.TestCase, ForAllBoostinAlgos):
    def work(self, boosting_cls):
        n_trials = 12
        round_len = 4
        trials = hyperopt.Trials()
        n_sub_trials = []
        bandit = DummyDecisionsBandit()
        class FakeAlgo(hyperopt.Random):
            def suggest(self, ids, specs, results, miscs):
                n_sub_trials.append(len(specs))
                assert len(specs) <= round_len
                return hyperopt.Random.suggest(self,
                        ids, specs, results, miscs)
        algo = FakeAlgo(bandit)
        boosting_algo = boosting_cls(algo,
                    round_len=round_len)
        exp = Experiment(trials, boosting_algo, async=False)
        round_ii = 0
        while len(trials) < n_trials:
            round_ii += 1
            exp.run(round_len)
            assert len(trials) == round_ii * round_len
            assert len(n_sub_trials) == round_ii * round_len
        # The FakeAlgo should be passed an increasing number of trials
        # during each round of boosting.
        print n_sub_trials
        if boosting_cls is experiments.AsyncBoostingAlgoB:
            assert n_sub_trials == range(n_trials)
        else:
            assert n_sub_trials == range(round_len) * (n_trials // round_len)


class TestIdxsContinuing(unittest.TestCase, ForAllBoostinAlgos):
    def work(self, cls):
        if 'Boosting' not in cls.__name__:
            return
        n_trials = 20
        round_len = 3

        algo = hyperopt.Random(DummyDecisionsBandit())
        trials = hyperopt.Trials()
        boosting_algo = cls(algo, round_len=round_len)
        exp = hyperopt.Experiment(trials, boosting_algo)
        exp.run(n_trials)
        for misc in trials.miscs:
            idxs = boosting_algo.idxs_continuing(trials.miscs, misc['tid'])
            myrounds = [miscs[idx]['boosting']['round']
                    for idxs in idxs]
            assert len(set(myrounds)) in (0, 1)


def test_mixtures():
    # -- run random search of M trials
    M = 5
    bandit = DummyDecisionsBandit()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = Experiment(trials, bandit_algo)
    exp.run(M)

    N = 3
    simple = experiments.SimpleMixture(trials, bandit)
    inds, weights = simple.mix_inds(N)

    results = trials.results
    specs = trials.specs
    losses = np.array(map(bandit.loss, results, specs))

    s = losses.argsort()
    assert list(inds) == s[:N].tolist()

    ada = experiments.AdaboostMixture(trials, bandit)
    ada_inds, ada_weights = ada.mix_inds(N)
    # -- assert that adaboost adds weights in decreasing order
    assert (ada_weights[:-1] >= ada_weights[1:]).all()

    #TODO: tests that shows that ensemble performance is increasing with
    #number of components


def test_parallel_algo():
    num_procs = 5
    num_sets = 3

    trials = hyperopt.Trials()
    bandit = DummyDecisionsBandit()

    n_specs_list = []

    class FakeRandom(hyperopt.Random):
        def suggest(self, ids, specs, results, miscs):
            if miscs:
                # -- test that the SubAlgo always sees only jobs from one of
                #     the 'procs'
                my_proc_num = miscs[0]['proc_num']
                assert all((my_proc_num == m['proc_num']) for m in miscs)
            n_specs_list.append(len(specs))
            return hyperopt.Random.suggest(self, ids, specs, results, miscs)

    algo = FakeRandom(bandit)
    parallel_algo = experiments.ParallelAlgo(algo, num_procs)
    exp = hyperopt.Experiment(trials, parallel_algo)
    exp.run(num_procs * num_sets)
    proc_nums = [m['proc_num'] for m in exp.trials.miscs]
    assert proc_nums == range(num_procs) * num_sets
    assert n_specs_list == [0] * num_procs + [1] * num_procs + [2] * num_procs



@nose.SkipTest
def test_random_ensembles():
    bandit = LargerBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(trials, bandit_algo)
    exp.run(NUM_ROUNDS)
    results = trials.results
    specs = trials.specs
    losses = np.array(map(bandit.loss, results, specs))
    s = losses.argsort()
    selected_specs = [list(trials)[s[0]]]
    er_partial = bandit.score_mixture_partial_svm(selected_specs)
    er_full = bandit.score_mixture_full_svm(selected_specs)
    errors = {'random_partial': er_partial,
              'random_full': er_full}
    selected_specs = {'random': selected_specs}

    assert np.abs(errors['random_full'][0] - .274) < 1e-2
    return exp, errors, selected_specs


@nose.SkipTest
def test_mixture_ensembles():
    bandit = NormalBoostableDigits()
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
    simple_specs, simple_weights = simple.mix_models(NUM_ROUNDS)
    simple_specs = [{'spec': spec} for spec in simple_specs]
    simple_er_partial = bandit.score_mixture_partial_svm(simple_specs)
    simple_er_full = bandit.score_mixture_full_svm(simple_specs)

    ada = experiments.AdaboostMixture(trials, bandit)
    ada_specs, ada_weights = ada.mix_models(NUM_ROUNDS)
    ada_specs = [{'spec': spec} for spec in ada_specs]
    ada_er_partial = bandit.score_mixture_partial_svm(ada_specs)
    ada_er_full = bandit.score_mixture_full_svm(ada_specs)    
    ada_specs_test, ada_weights_test = ada.mix_models(NUM_ROUNDS, test_mask=True)
    ada_specs_test = [{'spec': spec} for spec in ada_specs_test]
    ada_er_test_partial = bandit.score_mixture_partial_svm(ada_specs_test)
    ada_er_test_full = bandit.score_mixture_full_svm(ada_specs_test)     

    errors = {'simple_partial': simple_er_partial, 
              'simple_full': simple_er_full,
              'ada_partial': ada_er_partial,
              'ada_full': ada_er_full,
              'ada_test_partial': ada_er_test_partial,
              'ada_test_full': ada_er_test_full}
    
    selected_specs = {'simple': simple_specs,
                      'ada': ada_specs}
    
    assert np.abs(errors['simple_full'][0] - .234) < 1e-2
    ptl = np.array([0.2744,
                    0.2304,
                    0.236,
                    0.2256,
                    0.2232])
    assert np.abs(errors['simple_partial'][0] - ptl).max() < 1e-2
    assert np.abs(errors['ada_full'][0] - .1696) < 1e-2
    ptl = np.array([0.2744, 0.2336, 0.1968, 0.1832, 0.1688])
    assert np.abs(errors['ada_partial'][0] - ptl).max() < 1e-2
    assert np.abs(errors['ada_test_full'][0] - .1672) < 1e-2
    
    return exp, errors, selected_specs


@nose.SkipTest
def test_boosted_ensembles_async():
    """
    this test must be run via 
    """
    return boosted_ensembles_base(
            experiments.AsyncBoostingAlgoA)


@nose.SkipTest
def test_boosted_ensembles_sync():
    return boosted_ensembles_base(experiments.SyncBoostingAlgo)


@nose.SkipTest
def boosted_ensembles_base(boosting_algo_class):
    bandit = NormalBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)    
    boosting_algo = boosting_algo_class(bandit_algo,
                                        round_len=ROUND_LEN)
    trials = hyperopt.Trials()                                             
    exp = hyperopt.Experiment(
            trials,
            boosting_algo,
            async=False)
    exp.run(NUM_ROUNDS * ROUND_LEN)
    selected_specs = boosting_algo.best_by_round(list(trials))
    er_partial = bandit.score_mixture_partial_svm(selected_specs)
    er_full = bandit.score_mixture_full_svm(selected_specs)
    errors = {'boosted_partial': er_partial, 
              'boosted_full': er_full}
    selected_specs = {'boosted': selected_specs}
     
    assert np.abs(errors['boosted_full'][0] - .1528) < 1e-2
    assert np.abs(np.mean(errors['boosted_partial'][0]) - 0.21968) < 1e-2
    
    return exp, errors, selected_specs
