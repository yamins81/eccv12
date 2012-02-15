"""
Testing experiment classes
"""

import hyperopt

import eccv12.experiments as experiments
from eccv12.experiments import BoostedSerialExperiment

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
    bandit_algo = hyperopt.Random(BoostableDigits)
    exp = hyperopt.Experiment(bandit_algo)
    exp.run(M)

    N = 2
    simple = experiments.SimpleMixture(exp)
    inds, weights = simple.mix_inds(N)
    losses = np.array([_r['loss'] for _r in exp.results])
    s = losses.argsort()
    assert (inds == s[:N]).all()

    ada = experiments.AdaboostMixture(exp)
    ada_inds, ada_weights = ada.mix_inds(N)
    assert len(ada_inds) == 2
    #I'm not 100 sure exactly what to test here ...


