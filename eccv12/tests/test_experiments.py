"""
Testing experiment classes
"""

import numpy as np
import hyperopt
import eccv12.experiments as experiments
from eccv12.toyproblem import BoostableDigits


class FastBoostableDigits(BoostableDigits):
    param_gen = dict(BoostableDigits.param_gen)
    param_gen['svm_max_observations'] = 5000 # -- smaller value for speed



def test_boosting_algo():
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
    boosting_algo = experiments.BoostingAlgo(algo,
                round_len=round_len)
    exp = hyperopt.Experiment(
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


def test_mixtures():
    # -- run random search of M trials
    M = 5
    bandit = BoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(trials, bandit_algo)
    exp.run(M)

    N = 3
    simple = experiments.SimpleMixture(trials, bandit)
    inds, weights = simple.mix_inds(N)

    results = trials.results
    specs = trials.specs
    losses = np.array(map(bandit.loss, results, specs))
    s = losses.argsort()
    assert list(inds) == [2, 0, 3] 

    ada = experiments.AdaboostMixture(trials, bandit)
    ada_inds, ada_weights = ada.mix_inds(N)
    assert list(ada_inds) == [2, 0, 3]
    weights = np.array([[ 0.46038752,  0.44284515,  0.4722308 ,  0.4584254 ,  0.45255872],
                        [ 0.31151928,  0.32158091,  0.32195087,  0.30753429,  0.32146415],
                        [ 0.05972238,  0.02864797,  0.04602044,  0.04298737,  0.0497339 ]])
    assert np.abs(ada_weights - weights).max() < .001

    #TODO: tests that shows that ensemble performance is increasing with
    #number of components
    

def test_parallel_algo():
    num_procs = 5
    
    trials = hyperopt.Trials()
    calls = [0]
    bandit = FastBoostableDigits()
    class FakeAlgo(hyperopt.Random):
        def suggest(self, ids, specs, results, idxs, vals):
            calls[0] += 1
            return hyperopt.Random.suggest(self,
                    ids, specs, results, idxs, vals)
    algo = FakeAlgo(bandit)
    parallel_algo = experiments.ParallelAlgo(algo, num_procs)
    exp = hyperopt.Experiment(
            trials,
            parallel_algo,
            async=False)
    
    ##resolve issue with proc_num key
