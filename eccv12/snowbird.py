"""
Testing experiment classes
"""

import hyperopt

from eccv12.experiments import BoostingAlgo

from eccv12.toyproblem import BoostableDigits


def boost_digits(n_trials=10, round_len=1):
    trials = hyperopt.Trials()
    calls = [0]
    bandit = BoostableDigits()
    algo = hyperop.Random(bandit)
    boosting_algo = experiments.BoostingAlgo(algo,
                round_len=round_len)
    exp = hyperopt.Experiment(
            trials,
            boosting_algo,
            async=False)
    exp.run(n_trials)

    cPickle.dump(dict(
        n_trials=n_trials,
        round_len=round_len,
        trials=trials
        ),
        open('snowbird.boost_digits_%i_%i.pkl' % (n_trials, round_len),
            'w'))

def boost_digits_show(n_trials=10, round_len=1):
    dct = cPickle.load(
            open('snowbird.boost_digits_%i_%i.pkl' % (n_trials, round_len)))
    trials = dct['trials']
    bandit = BoostableDigits()

    cutoff = 0
    while cutoff < len(trials):
        best_trials = boosting_algo.best_by_round(list(trials))
        new_mixture_score = bandit.score_mixture(best_trials,
                partial_svm=True)
        assert new_mixture_score < last_mixture_score
        last_mixture_score = new_mixture_score
        import matplotlib; matplotlib.use("Qt4Agg");
        import matplotlib.pyplot as plt;
        import numpy as np; plt.plot(np.random.rand(50)); plt.show()




def boost_digits_pure():

