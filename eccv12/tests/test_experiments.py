"""
Testing experiment classes
"""

import hyperopt

import eccv12.experiments as experiments
from eccv12.experiments import BoostedSerialExperiment

from eccv12.toyproblem import BoostableDigits


def test_boosting_algo():
    exp = hyperopt.Experiment(
            hyperopt.Trials(),
            experiments.BoostingAlgo(
                hyperopt.Random(
                    BoostableDigits()),
                round_len=3),
            async=False)
    exp.run(5)


def test_boosted_serial():
    exp = BoostedSerialExperiment(
            trials_class=hyperopt.Trials,
            bandit_algo_class=hyperopt.Random,
            bandit_class=BoostableDigits,
            boost_rounds=2,
            round_len=1)
    # smoke test
    exp.run()
    assert len(exp.results) == 2


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


