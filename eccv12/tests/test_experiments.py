"""
Testing experiment classes
"""
import unittest
import nose
from nose.plugins.attrib import attr

import numpy as np
import hyperopt
import pyll
import eccv12.experiments as experiments
from eccv12.toyproblem import BoostableDigits
from eccv12.bandits import BaseBandit


def Experiment(*args, **kwargs):
    rval = hyperopt.Experiment(*args, **kwargs)
    rval.catch_bandit_exceptions = False
    return rval


class DummyDecisionsBandit(BaseBandit):
    param_gen = dict(
            seed=pyll.scope.randint(1000),
            decisions=None)

    def __init__(self, n_train, n_test, n_splits, *args, **kwargs):
        BaseBandit.__init__(self, *args, **kwargs)
        self.n_train = n_train
        self.n_test = n_test
        self.n_splits = n_splits

    def performance_func(self, config, ctrl):
        n_train = self.n_train
        n_test = self.n_test
        n_splits = self.n_splits
        n_examples = n_train + n_test

        r34 = np.random.RandomState(10000)
        y = np.sign(r34.randn(n_examples))
        rs = np.random.RandomState(config['seed'])
        yhat = rs.randn(n_splits, n_examples)
        yhat_test = yhat[:, :n_test]
        # normalizing predictions to have unit length *helps* to ensure
        # that adaboost weights do not increase with increasing round.
        # Since the errors can flip from 0 to 1 with arbitrarily small changes
        # to the combined decision vector though, it's not clear that this
        # ensures non-increasing weights.
        yhat_test /= np.sqrt((yhat_test ** 2).sum(axis=1))[:, np.newaxis]
        decisions = config['decisions']
        if decisions is None:
            decisions = np.zeros((n_splits, n_examples))
        new_dec = yhat + decisions
        is_test = decisions * 0 + [[1] * n_test + [0] * n_train]
        loss = np.mean(y[:n_test] != np.sign(new_dec[:, :n_test]))
        #print 'PRED', np.sign(new_dec)
        #print 'Y    ', y
        #print 'LOSS', loss
        result = dict(
                status=hyperopt.STATUS_OK,
                loss=loss,
                labels=y,
                is_test=is_test,
                decisions=new_dec)
        return result


class FastBoostableDigits(BoostableDigits):
    NUM_ROUNDS = 5           # -- use this many rounds to do it
    BASE_NUM_FEATURES = 50   # -- construct ensembles with this many features
    ROUND_LEN = 5

    param_gen = dict(BoostableDigits.param_gen)
    #param_gen['svm_max_observations'] = 1000


class NormalBoostableDigits(FastBoostableDigits):
    param_gen = dict(FastBoostableDigits.param_gen)
    # usually we build en ensemble by evenly-sized incrememts
    param_gen['feat_spec']['n_features'] = (FastBoostableDigits.BASE_NUM_FEATURES
            / FastBoostableDigits.ROUND_LEN)


class LargerBoostableDigits(FastBoostableDigits):
    param_gen = dict(FastBoostableDigits.param_gen)
    # this class is used when searching the full ensemble space using random
    # search
    param_gen['feat_spec']['n_features'] = FastBoostableDigits.BASE_NUM_FEATURES


class ForAllBoostingAlgos(object):
    """Mixin tests that call self.work(<BoostingAlgoCls>)
    """

    def test_sync(self):
        self.work(experiments.SyncBoostingAlgo)

    def test_asyncA(self):
        self.work(experiments.AsyncBoostingAlgoA)

    def test_asyncB(self):
        self.work(experiments.AsyncBoostingAlgoB)


def assert_boosting_loss_decreases(cls):
    """Test that boosting decreases generalization loss in a not-so-fast-to-run
    actual learning setting.
    """
    n_trials = 12
    round_len = 3
    trials = hyperopt.Trials()
    bandit = FastBoostableDigits()
    algo = hyperopt.Random(bandit)
    boosting_algo = experiments.SyncBoostingAlgo(algo,
                round_len=round_len)
    exp = Experiment(trials, boosting_algo)

    last_mixture_score = float('inf')

    round_ii = 0
    while len(trials) < n_trials:
        round_ii += 1
        exp.run(round_len)
        assert len(trials) == round_ii * round_len
        best_trials = cls.boosting_best_by_round(trials, bandit)
        assert len(best_trials) == round_ii
        train_errs, test_errs = bandit.score_mixture_partial_svm(best_trials)
        print test_errs # train_errs, test_errs
        assert test_errs[-1] < last_mixture_score
        last_mixture_score = test_errs[-1]


@attr('slow')
def test_boosting_loss_decreases_sync():
    assert_boosting_loss_decreases(experiments.SyncBoostingAlgo)


@attr('slow')
def test_boosting_loss_decreases_asyncA():
    assert_boosting_loss_decreases(experiments.AsyncBoostingAlgoA)


@attr('slow')
def test_boosting_loss_decreases_asyncB():
    assert_boosting_loss_decreases(experiments.AsyncBoostingAlgoB)


class TestBoostingSubAlgoArgs(unittest.TestCase, ForAllBoostingAlgos):
    """
    Test that in a synchronous experiment, the various Boosting algorithms
    pass the Sub-Algorithm (here Random search) a plausible number of trials
    on every iteration.
    """
    n_trials = 12
    round_len = 4
    def work(self, boosting_cls):
        trials = hyperopt.Trials()
        n_sub_trials = []
        bandit = DummyDecisionsBandit(n_train=3, n_test=2, n_splits=1)
        class FakeAlgo(hyperopt.Random):
            def suggest(_self, ids, specs, results, miscs):
                n_sub_trials.append(len(specs))
                if boosting_cls != experiments.AsyncBoostingAlgoB:
                    assert len(specs) <= self.round_len
                return hyperopt.Random.suggest(_self,
                        ids, specs, results, miscs)
        algo = FakeAlgo(bandit)
        boosting_algo = boosting_cls(algo,
                    round_len=self.round_len)
        exp = Experiment(trials, boosting_algo, async=False)
        round_ii = 0
        while len(trials) < self.n_trials:
            round_ii += 1
            exp.run(self.round_len)
            assert len(trials) == round_ii * self.round_len
            assert len(n_sub_trials) == round_ii * self.round_len
        # The FakeAlgo should be passed an increasing number of trials
        # during each round of boosting.

        print n_sub_trials
        if boosting_cls is experiments.SyncBoostingAlgo:
            assert n_sub_trials == range(self.round_len) * (self.n_trials // self.round_len)
        elif boosting_cls is experiments.AsyncBoostingAlgoA:
            assert n_sub_trials == range(self.round_len) * (self.n_trials // self.round_len)
        elif boosting_cls is experiments.AsyncBoostingAlgoB:
            assert n_sub_trials[-1] > self.round_len
        else:
            raise NotImplementedError(boosting_cls)


class TestIdxsContinuing(unittest.TestCase, ForAllBoostingAlgos):
    """
    Test that all the trials that extend a particular trial have a round
    number that is one greater than the trial being extended.
    """
    def work(self, cls):
        if 'Boosting' not in cls.__name__:
            return
        n_trials = 20
        round_len = 3

        algo = hyperopt.Random(DummyDecisionsBandit(3, 2, 1))
        trials = hyperopt.Trials()
        boosting_algo = cls(algo, round_len=round_len)
        exp = hyperopt.Experiment(trials, boosting_algo)
        exp.run(n_trials)
        for misc in trials.miscs:
            idxs = boosting_algo.idxs_continuing(trials.miscs, misc['tid'])
            continued_in_rounds = [trials.miscs[idx]['boosting']['round']
                    for idx in idxs]
            #for s, m in zip(trials.specs, trials.miscs):
                #print m['tid'], m['boosting'], s['decisions']
            assert all(r > misc['boosting']['round'] for r in continued_in_rounds)


def test_mixtures():
    # -- run random search of M trials
    M = 5
    bandit = DummyDecisionsBandit(n_train=80, n_test=20, n_splits=1)
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

    # -- test_mask has to be True for weights to be guaranteed to be decrease
    #    because of the way DummyDecisionsBandit is normalizing the
    #    predictions
    ada = experiments.AdaboostMixture(trials, bandit, test_mask=True)
    ada_inds, ada_weights = ada.mix_inds(N)
    # -- assert that adaboost adds weights in decreasing magnitude (see
    # comment in DummyDecisionsBandit)
    assert (abs(ada_weights[:-1]) >= abs(ada_weights[1:])).all()

    #TODO: tests that shows that ensemble performance is increasing with
    #number of components


def test_parallel_algo():
    num_procs = 5
    num_sets = 3

    trials = hyperopt.Trials()
    bandit = DummyDecisionsBandit(n_train=50, n_test=10, n_splits=2)

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


def end_to_end_regression_helper(bandit, bandit_algo, best_target, atol,
        n_trials):

        trials = hyperopt.Trials()
        exp = Experiment(trials, bandit_algo)
        exp.run(n_trials)
        results = trials.results
        specs = trials.specs
        losses = np.array(map(bandit.loss, results, specs))
        assert None not in losses
        assert len(losses) == n_trials
        best = losses.min()
        print best
        if best_target is not None:
            assert np.allclose(best, best_target, atol=atol)
        return trials


def test_end_to_end_random():
    bandit = DummyDecisionsBandit(10, 20, 2)
    bandit_algo = hyperopt.Random(bandit)
    trials = end_to_end_regression_helper(bandit, bandit_algo, 0.3, 0.01, 35)


def test_end_to_end_simple_mixture():
    bandit = DummyDecisionsBandit(n_train=10, n_test=100, n_splits=1)
    bandit_algo = hyperopt.Random(bandit)
    trials = end_to_end_regression_helper(bandit, bandit_algo, 0.34, 0.01, 35)
    simple = experiments.SimpleMixture(trials, bandit)
    idxs, weights = simple.mix_inds(7) # rounds of len 5
    print list(idxs), weights
    # catch regressions due to e.g. sampler changes
    assert list(idxs) == [ 8, 28, 20, 31,  5, 18, 12]
    assert np.allclose(weights, 1.0 / 7)


def test_end_to_end_ada_mixture():
    bandit = DummyDecisionsBandit(n_train=10, n_test=100, n_splits=1)
    bandit_algo = hyperopt.Random(bandit)
    trials = end_to_end_regression_helper(bandit, bandit_algo, 0.34, 0.01, 35)
    ada = experiments.AdaboostMixture(trials, bandit, test_mask=True)
    idxs, weights = ada.mix_inds(7)
    assert weights.shape == (7, 1)
    print list(idxs), weights
    assert (abs(weights[:-1]) >= abs(weights[1:])).all()
    # -- N.B. first selected idx is same as first selected in simple mixture
    assert list(idxs) == [ 8, 24, 25,  7, 14, 29, 10]
    assert np.allclose(weights[0], 0.3316, atol=0.001)
    assert np.allclose(weights[6], -0.13519, atol=0.001)



def test_end_to_end_boost_sync():
    bandit = DummyDecisionsBandit(n_train=10, n_test=100, n_splits=1)
    sub_algo = hyperopt.Random(bandit)
    boosting_algo = experiments.SyncBoostingAlgo(sub_algo, round_len=5)
    trials = end_to_end_regression_helper(bandit, boosting_algo, 0.4, 0.01, 35)
    selected = boosting_algo.ensemble_member_tids(trials, bandit)
    print 'SEL', selected
    assert selected == [0, 6, 12, 15, 20, 28, 34]


def test_end_to_end_boost_asyncA():
    bandit = DummyDecisionsBandit(n_train=10, n_test=100, n_splits=1)
    sub_algo = hyperopt.Random(bandit)
    boosting_algo = experiments.AsyncBoostingAlgoA(sub_algo, round_len=5)
    trials = end_to_end_regression_helper(bandit, boosting_algo, 0.4, 0.01, 35)
    selected = boosting_algo.ensemble_member_tids(trials, bandit)
    print 'SEL', selected
    # This performs same as sync boosting -- how are they different again?
    assert selected == [0, 6, 12, 15, 20, 28, 34]


def test_end_to_end_boost_asyncB():
    bandit = DummyDecisionsBandit(n_train=10, n_test=100, n_splits=1)
    sub_algo = hyperopt.Random(bandit)
    boosting_algo = experiments.AsyncBoostingAlgoB(sub_algo, round_len=5)
    # This should score better than boost_sync but worse than AdaBoost mixture
    # because this bandit doesn't even use the decisions.
    # Consequently, AdaBoost is free to choose the ensemble at the end,
    # whereas asyncB, despite the lookback, is forced to make additions to the
    # ensemble as the trials come in. Consequently, the selected members will
    # necessarily have increasing IDs, unlike AdaBoostMixture.
    trials = end_to_end_regression_helper(bandit, boosting_algo, 0.38, 0.01, 35)
    selected = boosting_algo.ensemble_member_tids(trials, bandit)
    print 'SEL', selected
    # This performs same as sync boosting -- how are they different again?
    assert selected == [0, 6, 28, 34]


@attr('slow')
def test_random_ensembles():
    """
    It runs experiments on "LargerBoostableDigits" and asserts that the
    results come out consistently with our expectation regarding order.
    """
    bandit = LargerBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(trials, bandit_algo)
    # XXX: is this right? Comment on justification.
    exp.run(FastBoostableDigits.NUM_ROUNDS)
    results = trials.results
    specs = trials.specs
    losses = np.array(map(bandit.loss, results, specs))
    # these are pretty consistent:
    # [ 0.53227277  0.49521992  0.51694432  0.51861616  0.51243517]
    print losses
    selected_spec = trials.specs[np.argmin(losses)]
    er_partial = bandit.score_mixture_partial_svm([selected_spec])
    er_full = bandit.score_mixture_full_svm([selected_spec])
    errors = {'random_partial': er_partial,
              'random_full': er_full}

    assert np.allclose(errors['random_full'][0], .144, atol=1e-3)
    assert np.allclose(errors['random_full'][1], .142, atol=1e-3)
    return exp, errors, {'random': [selected_spec]}


@attr('slow')
def test_digits_random_ensembles():
    """test_digits_random_ensembles
    """
    bandit = LargerBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = hyperopt.Experiment(trials, bandit_algo)
    exp.run(FastBoostableDigits.NUM_ROUNDS)
    results = trials.results
    specs = trials.specs
    losses = np.array(map(bandit.loss, results, specs))
    print losses
    #[ 0.78281302  0.74598096  0.8264129   0.76735577  0.77177276]
    s = losses.argmin()
    selected_specs = [trials.specs[s]]
    er_partial = bandit.score_mixture_partial_svm(selected_specs)
    er_full = bandit.score_mixture_full_svm(selected_specs)
    errors = {'random_partial': er_partial,
              'random_full': er_full}
    selected_specs = {'random': selected_specs}

    assert np.abs(errors['random_full'][0] - .274) < 1e-2
    return exp, errors, selected_specs


@attr('slow')
def test_mixture_ensembles():
    """
    It runs experiments on "LargerBoostableDigits" and asserts that the
    results come out consistently with our expectation regarding order.
    """
    NUM_ROUNDS = FastBoostableDigits.NUM_ROUNDS
    ROUND_LEN = FastBoostableDigits.ROUND_LEN

    bandit = NormalBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = Experiment(trials, bandit_algo)
    exp.run(NUM_ROUNDS * ROUND_LEN)

    results = trials.results
    specs = trials.specs

    simple = experiments.SimpleMixture(trials, bandit)
    simple_specs, simple_weights = simple.mix_models(NUM_ROUNDS)
    simple_er_partial = bandit.score_mixture_partial_svm(simple_specs)
    simple_er_full = bandit.score_mixture_full_svm(simple_specs)

    ada_nomask = experiments.AdaboostMixture(trials, bandit, test_mask=False)
    ada_nomask_specs, ada_nomask_weights = ada_nomask.mix_models(NUM_ROUNDS)
    ada_nomask_er_partial = bandit.score_mixture_partial_svm(ada_nomask_specs)
    ada_nomask_er_full = bandit.score_mixture_full_svm(ada_nomask_specs)

    ada_mask = experiments.AdaboostMixture(trials, bandit, test_mask=True)
    ada_mask_specs, ada_mask_weights= ada_mask.mix_models(NUM_ROUNDS)
    ada_mask_er_partial = bandit.score_mixture_partial_svm(ada_mask_specs)
    ada_mask_er_full = bandit.score_mixture_full_svm(ada_mask_specs)

    errors = {'simple_partial': simple_er_partial,
              'simple_full': simple_er_full,
              'ada_nomask_partial': ada_nomask_er_partial,
              'ada_nomask_full': ada_nomask_er_full,
              'ada_mask_partial': ada_mask_er_partial,
              'ada_mask_full': ada_mask_er_full}

    print errors

    ##
    ## SimpleMixture tests
    ##
    ##   -  errors[...][0] is training error
    ##   -  errors[...][1] is test error

    print 'simple_full', errors['simple_full']
    assert np.allclose(errors['simple_full'][0], 0.068, atol=1e-3)
    assert np.allclose(errors['simple_full'][1], 0.11, atol=1e-3)

    print 'simple_partial[0]', errors['simple_partial'][0]
    print 'simple_partial[1]', errors['simple_partial'][1]
    assert np.allclose(errors['simple_partial'][0],
            [0.1336, 0.088800000000000004, 0.076799999999999993,
                0.067199999999999996, 0.068000000000000005],
        atol=1e-3)
    assert np.allclose(errors['simple_partial'][1],
            [0.14199999999999999, 0.112, 0.114, 0.11, 0.104],
        atol=1e-3)

    ### XXX: is it normal that the partial fit results get so much lower in
    #         both train and test, than the full-fit errors?  Usually I saw
    #         the reverse -- full fitting should certainly get lower training
    #         error.

    ##
    ## AdaBoostMixture tests
    ##
    ##   -  errors[...][0] is training error
    ##   -  errors[...][1] is test error


    print 'ANF', list(errors['ada_nomask_full'])
    print 'ANP0', list(errors['ada_nomask_partial'][0])
    print 'ANP1', list(errors['ada_nomask_partial'][1])
    print 'AMF', list(errors['ada_mask_full'])
    print 'AMP0', list(errors['ada_mask_partial'][0])
    print 'AMP1', list(errors['ada_mask_partial'][1])

    assert np.allclose(errors['ada_nomask_full'][0], 0.056, atol=1e-3)
    assert np.allclose(errors['ada_nomask_full'][1], 0.108, atol=1e-3)

    assert np.allclose(errors['ada_nomask_partial'][0],
            [0.1336, 0.1024, 0.076799999999999993, 0.066400000000000001,
                0.059999999999999998],
            atol=1e-3)
    assert np.allclose(errors['ada_nomask_partial'][1],
            [0.14199999999999999, 0.14399999999999999, 0.12, 0.126, 0.122],
            atol=1e-3)

    assert np.allclose(errors['ada_mask_full'][0], 0.063200000000000006, atol=1e-3)
    assert np.allclose(errors['ada_mask_full'][1],  0.106, atol=1e-3)

    assert np.allclose(errors['ada_mask_partial'][0],
            [0.14399999999999999, 0.1032, 0.073599999999999999,
                0.073599999999999999, 0.061600000000000002],
            atol=1e-3)
    assert np.allclose(errors['ada_mask_partial'][1],
            [0.14199999999999999, 0.13400000000000001, 0.126, 0.122, 0.12],
            atol=1e-3)

    selected_specs = {'simple': simple_specs,
                      'ada_nomask': ada_nomask_specs,
                      'ada_mask': ada_mask_specs,
                      }

    return exp, errors, selected_specs


def boosted_ensembles_helper(boosting_algo_class, full_0, full_1, part_0,
        part_1, atol):
    """
    It runs experiments on "LargerBoostableDigits" and asserts that the
    results come out consistently with our expectation regarding order.
    """
    NUM_ROUNDS = FastBoostableDigits.NUM_ROUNDS
    ROUND_LEN = FastBoostableDigits.ROUND_LEN

    bandit = NormalBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    boosting_algo = boosting_algo_class(bandit_algo, round_len=ROUND_LEN)
    trials = hyperopt.Trials()
    exp = Experiment(trials, boosting_algo)
    exp.run(NUM_ROUNDS * ROUND_LEN)
    selected_specs = boosting_algo.boosting_best_by_round(trials, bandit)
    selected_tids = boosting_algo.ensemble_member_tids(trials, bandit)
    er_partial = bandit.score_mixture_partial_svm(selected_specs)
    er_full = bandit.score_mixture_full_svm(selected_specs)
    errors = {'boosted_partial': er_partial, 'boosted_full': er_full}
    selected_specs = {'boosted': selected_specs}

    print 'TIDs=', selected_tids
    print 'full_0=', errors['boosted_full'][0]
    print 'full_1=', errors['boosted_full'][1]
    print "part_0=", list(errors['boosted_partial'][0])
    print "part_1=", list(errors['boosted_partial'][1])

    assert np.allclose(errors['boosted_full'][0], full_0, atol=atol)
    assert np.allclose(errors['boosted_full'][1], full_1, atol=atol)

    assert np.allclose(errors['boosted_partial'][0], part_0, atol=atol)
    assert np.allclose(errors['boosted_partial'][1], part_1, atol=atol)

    return exp, errors, selected_specs


@attr('slow')
def test_boosted_ensembles_asyncA():
    return boosted_ensembles_helper(
            experiments.AsyncBoostingAlgoA,
            full_0=0.0512,
            full_1=0.11,
            part_0=[0.14399999999999999, 0.089599999999999999,
                0.071199999999999999, 0.0608, 0.054399999999999997],
            part_1=[0.14199999999999999, 0.128, 0.11600000000000001,
                0.11600000000000001, 0.106],
            atol=1e-2,
            )


@attr('slow')
def test_boosted_ensembles_asyncB():
    # -- this behaves identically to asyncA
    #    It's plausible that this is correct.
    #    Other unit-tests show that the implementations differ when they're
    #    supposed to.
    return boosted_ensembles_helper(
            experiments.AsyncBoostingAlgoB,
            full_0=0.0512,
            full_1=0.11,
            part_0=[0.14399999999999999, 0.089599999999999999,
                0.071199999999999999, 0.0608, 0.054399999999999997],
            part_1=[0.14199999999999999, 0.128, 0.11600000000000001,
                0.11600000000000001, 0.106],
            atol=1e-2,
            )


@attr('slow')
def test_boosted_ensembles_sync():
    return boosted_ensembles_helper(
            experiments.SyncBoostingAlgo,
            #TIDs=[1, 5, 10], -- not currently tested
            full_0=0.0512,
            full_1=0.11,
            part_0=[0.1664, 0.1136, 0.096, 0.09034, 0.084],
            part_1=[0.17, 0.142, 0.116, 0.12, 0.118],
            atol=1e-2,
            )



# XXX TEST WITH SPORADIC FAILURES

# XXX TEST WITH ASYNC EXECUTION

