"""
Testing experiment classes
"""
import cPickle
import sys
import matplotlib.pyplot

import hyperopt
import hyperopt.plotting

from eccv12.experiments import BoostingAlgo
from eccv12.experiments import AdaboostMixture

from eccv12.toyproblem import BoostableDigits

def fat_feature_boostable_digits(multiplier):
    from pyll import scope, clone, as_apply
    class FatFeatures(BoostableDigits):
        param_gen = dict(
                n_examples_train=1250,
                n_examples_test=500,
                n_folds=5,
                feat_spec=dict(
                    seed=scope.one_of(1, 2, 3, 4, 5),
                    n_features=scope.one_of(*[
                        multiplier * nn for nn in [2, 5, 10]]),
                    scale=scope.uniform(0, 5)
                    ),
                svm_l2_regularization=1e-3,
                decisions=None,
                svm_max_observations=20000,
                )
    return FatFeatures()


def boost_digits(n_trials, round_len, multiplier=1):
    trials = hyperopt.Trials()
    calls = [0]
    if multiplier > 1:
        bandit = fat_feature_boostable_digits(multiplier)
    else:
        bandit = BoostableDigits()
    algo = hyperopt.Random(bandit)
    boosting_algo = BoostingAlgo(algo, round_len=round_len)
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
    algo = hyperopt.Random(bandit)
    boosting_algo = BoostingAlgo(algo, round_len=round_len)

    best_trials = boosting_algo.best_by_round(list(trials))
    train_err_rates, test_err_rates = bandit.score_mixture_partial_svm(
            best_trials)

    # XXX put in .matplotlibrc
    hyperopt.plotting.main_plot_history(trials, bandit, do_show=False)

    best_times = range(0, len(trials), round_len)
    matplotlib.pyplot.scatter(best_times, train_err_rates, c='g')
    matplotlib.pyplot.scatter(best_times, test_err_rates, c='r')


    matplotlib.pyplot.show()


def after_the_fact_adaboost(orig_n_trials, orig_round_len, boosting_rounds):
    dct = cPickle.load(
            open('snowbird.boost_digits_%i_%i.pkl' % (n_trials, round_len)))
    trials = dct['trials']
    bandit = BoostableDigits()

    ada = AdaboostMixture(trials, bandit)
    ada_inds, ada_weights = ada.mix_inds(boosting_rounds)

    print ada_inds


def main():
    cmd = sys.argv[1]
    args = [eval(aa) for aa in sys.argv[1:]]
    return args[0](*args[1:])


if __name__ == '__main__':
    import sys
    sys.exit(main())

