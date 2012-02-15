"""
Testing experiment classes
"""
import cPickle
import sys
import matplotlib.pyplot

import hyperopt
import hyperopt.plotting
from pyll import scope, clone, as_apply

from eccv12.experiments import BoostingAlgo
from eccv12.experiments import AdaboostMixture

from eccv12.toyproblem import BoostableDigits

def fat_feature_boostable_digits(n_features):
    class FatFeatures(BoostableDigits):
        param_gen = dict(
                n_examples_train=1250,
                n_examples_test=500,
                n_folds=5,
                feat_spec=dict(
                    seed=scope.randint(10000) + 1,
                    n_features=n_features,
                    scale=scope.uniform(0, 5)
                    ),
                svm_l2_regularization=1e-3,
                decisions=None,
                svm_max_observations=20000,
                )
        def evaluate(self, config, ctrl):
            print 'FatFeatures.evaluate', config['feat_spec']
            return BoostableDigits.evaluate(self, config, ctrl)
    return FatFeatures()

def boost_digits_filename(n_trials, round_len, n_features):
    return 'snowbird.boost_digits_T%i_R%i_F%i.pkl' % (
            n_trials, round_len, n_features)


def boost_digits(n_trials, round_len, n_features):
    trials = hyperopt.Trials()
    calls = [0]
    bandit = fat_feature_boostable_digits(n_features)
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
        open(boost_digits_filename(n_trials, round_len, n_features), 'w'))


def boost_digits_show(n_trials, round_len, n_features):
    dct = cPickle.load(
            open(boost_digits_filename(n_trials, round_len, n_features)))
    trials = dct['trials']
    bandit = BoostableDigits()
    algo = hyperopt.Random(bandit)
    boosting_algo = BoostingAlgo(algo, round_len=round_len)

    best_trials = boosting_algo.best_by_round(list(trials))
    train_err_rates, test_err_rates = bandit.score_mixture_partial_svm(
            best_trials)

    train_err_rate_full, test_err_rate_full = bandit.score_mixture_full_svm(
            best_trials)

    hyperopt.plotting.main_plot_history(trials, bandit, do_show=False)

    best_times = range(round_len, len(trials) + round_len, round_len)
    matplotlib.pyplot.scatter(best_times, train_err_rates, c='g')
    matplotlib.pyplot.scatter(best_times, test_err_rates, c='r')

    matplotlib.pyplot.axhline(train_err_rate_full)
    matplotlib.pyplot.axhline(test_err_rate_full)
    matplotlib.pyplot.show()


def budget_feature_search(n_search, n_keep):
    """
    Compare approaches in the setting where there is a budget
    - for searching `n_search` features during training
    - for using `n_keep` features during testing.
    """
    assert n_search % n_keep == 0

    # -- using ensemble search
    ensemble_kwargs = dict(
            n_trials=n_search / n_keep,
            round_len=n_search / n_keep,
            n_features=n_keep)

    # -- using 5 boosting rounds
    assert n_keep % 5 == 0
    # n_keep / 5 * n_trials == n_search
    # n_trials == n_search * 5 / n_keep
    boost5_kwargs = dict(
            n_trials=n_search * 5 / n_keep,
            round_len=n_search / n_keep,
            n_features = n_keep / 5)

    # -- using 10 boosting rounds
    assert n_keep % 10 == 0
    boost10_kwargs = dict(
            n_trials=n_search * 10 / n_keep,
            round_len=n_search / n_keep,
            n_features = n_keep / 10)

    print 'ensemble_kwargs', ensemble_kwargs
    print 'boost5_kwargs', boost5_kwargs
    print 'boost10_kwargs', boost10_kwargs

    import multiprocessing
    PTH = multiprocessing.Process
    threads = []
    threads.append(PTH(target=boost_digits, kwargs=ensemble_kwargs))
    threads.append(PTH(target=boost_digits, kwargs=boost5_kwargs))
    threads.append(PTH(target=boost_digits, kwargs=boost10_kwargs))

    [th.start() for th in threads]
    [th.join() for th in threads]



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

