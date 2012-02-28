"""
Testing experiment classes

The "test_digits" tests show validate that the four
basic algorithms for ensemble selection produce loss results in the 
right ordering, e.g. random ensembles > Top-A mixture > Adaboost > HTBoost
"""
import copy
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
    param_gen = dict(BoostableDigits.param_gen)
    param_gen['svm_max_observations'] = 1000


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
            def suggest(_self, ids, trials):
                n_sub_trials.append(len(trials))
                if boosting_cls != experiments.AsyncBoostingAlgoB:
                    assert len(trials) <= self.round_len
                return hyperopt.Random.suggest(_self,
                        ids, trials)
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
        def suggest(self, ids, trials):
            trials.refresh()
            if len(trials):
                # -- test that the SubAlgo always sees only jobs from one of
                #     the 'procs'
                my_key = trials.trials[0]['exp_key']
                assert all((my_key == t['exp_key']) for t in trials)
            n_specs_list.append(len(trials))
            return hyperopt.Random.suggest(self, ids, trials)

    algo = FakeRandom(bandit)
    parallel_algo = experiments.ParallelAlgo(algo, num_procs)
    exp = hyperopt.Experiment(trials, parallel_algo)
    exp.run(num_procs * num_sets)
    proc_nums = [int(t['exp_key'][3:]) for t in exp.trials]
    assert proc_nums == range(num_procs) * num_sets
    print n_specs_list
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


##
## Slow tests actually doing scientifically relevant tests on the
## BoostableDigits toy problem.
##

BASE_NUM_FEATURES = 16   # -- construct ensembles with this many features
NUM_ROUNDS = 4           # -- use this many rounds to do it
ROUND_LEN = 3            # -- try this many guesses in each round


class NormalBoostableDigits(BoostableDigits):
    """
    This class is for searching the full ensemble space in NUM_ROUNDS
    iterations of ensemble construction.
    """

    param_gen = copy.deepcopy(FastBoostableDigits.param_gen)
    param_gen['feat_spec']['n_features'] = BASE_NUM_FEATURES / NUM_ROUNDS


class LargerBoostableDigits(BoostableDigits):
    """
    This class is for searching the full ensemble space all at once
    """
    param_gen = copy.deepcopy(FastBoostableDigits.param_gen)
    param_gen['feat_spec']['n_features'] = BASE_NUM_FEATURES


@attr('slow')
def test_random_ensembles():
    """
    It runs experiments on "LargerBoostableDigits" and asserts that the
    results come out consistently with our expectation regarding order.
    """
    bandit = LargerBoostableDigits()
    bandit_algo = hyperopt.Random(bandit)
    trials = hyperopt.Trials()
    exp = Experiment(trials, bandit_algo)
    exp.run(ROUND_LEN)
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

    assert np.allclose(errors['random_full'][0], 0.3656, atol=1e-3)
    assert np.allclose(errors['random_full'][1], 0.354, atol=1e-3)

    return exp, errors, {'random': [selected_spec]}


@attr('slow')
def test_mixture_ensembles():
    """
    It runs experiments on "LargerBoostableDigits" and asserts that the
    results come out consistently with our expectation regarding order.
    """

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


    ##
    ## SimpleMixture tests
    ##
    ##   -  errors[...][0] is training error
    ##   -  errors[...][1] is test error

    ##
    ## AdaBoostMixture tests
    ##
    ##   -  errors[...][0] is training error
    ##   -  errors[...][1] is test error

    print errors
    print '-' * 80
    print """
    CUT AND PASTE THIS IN TO UPDATE TESTS:

    assert np.allclose(errors['simple_full'][0], %.5f, atol=1e-3)
    assert np.allclose(errors['simple_full'][1], %.5f, atol=1e-3)

    assert np.allclose(errors['ada_nomask_full'][0], %.5f, atol=1e-3)
    assert np.allclose(errors['ada_nomask_full'][1], %.5f, atol=1e-3)

    assert np.allclose(errors['ada_mask_full'][0], %.5f, atol=1e-3)
    assert np.allclose(errors['ada_mask_full'][1],  %.5f, atol=1e-3)

    """ % (errors['simple_full']
            + errors['ada_nomask_full']
            + errors['ada_mask_full'])
    print '-' * 80

    assert np.allclose(errors['simple_full'][0], 0.27840, atol=1e-3)
    assert np.allclose(errors['simple_full'][1], 0.28000, atol=1e-3)

    assert np.allclose(errors['ada_nomask_full'][0], 0.25760, atol=1e-3)
    assert np.allclose(errors['ada_nomask_full'][1], 0.28000, atol=1e-3)

    assert np.allclose(errors['ada_mask_full'][0], 0.27280, atol=1e-3)
    assert np.allclose(errors['ada_mask_full'][1], 0.27800, atol=1e-3)


    selected_specs = {'simple': simple_specs,
                      'ada_nomask': ada_nomask_specs,
                      'ada_mask': ada_mask_specs,
                      }

    return exp, errors, selected_specs


def boosted_ensembles_helper(boosting_algo_class, full_0, full_1, atol=1e-3):
    """
    It runs experiments on "LargerBoostableDigits" and asserts that the
    results come out consistently with our expectation regarding order.
    """
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

    print boosting_algo_class
    print 'TIDs=', selected_tids
    print '-' * 80
    print """
    CUT AND PASTE THIS IN TO UPDATE TESTS:

    full_0=%.5f,
    full_1=%.5f,

    """ % (errors['boosted_full'])
    print '-' * 80

    assert np.allclose(errors['boosted_full'][0], 0.27360, atol=1e-3)
    assert np.allclose(errors['boosted_full'][1], 0.27400, atol=1e-3)


    return exp, errors, selected_specs


@attr('slow')
def test_boosted_ensembles_asyncA():
    return boosted_ensembles_helper(
            experiments.AsyncBoostingAlgoA,
            full_0=0.2736,
            full_1=0.274,
            )


@attr('slow')
def test_boosted_ensembles_asyncB():
    # -- this behaves identically to asyncA
    #    It's plausible that this is correct.
    #    Other unit-tests show that the implementations differ when they're
    #    supposed to.
    return boosted_ensembles_helper(
            experiments.AsyncBoostingAlgoB,
            full_0=0.2736,
            full_1=0.274,
            )


@attr('slow')
def test_boosted_ensembles_sync():
    return boosted_ensembles_helper(
            experiments.SyncBoostingAlgo,
            #TIDs=[1, 5, 10], -- not currently tested
            full_0=0.2736,
            full_1=0.274,
            )



class TestAsyncError(unittest.TestCase):
    """
    Tests that ensure that each Algo behaves sensibly in the context of jobs
    whose finish times are spread across multiple job start times.
    """

    def setUp(self):
        self.trials = hyperopt.Trials()
        self.bandit = DummyDecisionsBandit(n_train=50, n_test=10, n_splits=2)
        # -- the list of tids used to get info back from FakeRandom
        self.miscs_tids = []
        class FakeRandom(hyperopt.Random):
            def suggest(_self, ids, trials):
                if trials:
                    # -- test that the SubAlgo always sees only jobs from one of
                    #     the 'procs'
                    my_proc_num = trials.miscs[0]['proc_num']
                    assert all((my_proc_num == m['proc_num'])
                            for m in trials.miscs)
                # pass back the tids used in this call to suggest
                self.miscs_tids.append([m['tid'] for m in trials])
                return hyperopt.Random.suggest(_self, ids, trials)
        self.sub_algo = FakeRandom(self.bandit)

    #
    # Define a mini-language for describing asynchronous and error-ridden
    # experiments as an event sequence.
    #

    def push_job(self, tid, expected_ids):
        trials = self.trials
        trials.refresh()
        new_ids = [tid]
        # -- clear out the tid-passing buffer
        self.miscs_tids[:] = []
        print '--'
        #print [m['proc_num'] for m in trials.miscs]

        # -- self.algo is FakeRandom above, which populates miscs_tids
        new_docs = self.algo.suggest(new_ids, trials)
        if expected_ids is not None:
            # -- assert that FakeRandom got the expected tids
            #    to work with
            assert len(self.miscs_tids) == 1
            print "---"
            print 'EXPECTED', expected_ids
            print 'GOT', self.miscs_tids
            assert self.miscs_tids[0] == expected_ids
        new_docs[0]['misc']['cmd'] = None
        trials.insert_trial_docs(new_docs)
        trials.refresh()
        return new_ids

    def do_job_ok(self, tid):
        for trial in self.trials._dynamic_trials:
            if trial['tid'] == tid:
                assert trial['state'] == hyperopt.JOB_STATE_NEW
                trial['state'] = hyperopt.JOB_STATE_DONE
                trial['result'] = self.bandit.evaluate(trial['spec'], None)
        self.trials.refresh()

    def do_job_fail(self, tid):
        for trial in self.trials._dynamic_trials:
            if trial['tid'] == tid:
                assert trial['state'] == hyperopt.JOB_STATE_NEW
                trial['state'] = hyperopt.JOB_STATE_DONE
                trial['result'] = self.bandit.evaluate(trial['spec'], None)
                assert trial['result']['status'] == hyperopt.STATUS_OK
                trial['result']['status'] = hyperopt.STATUS_FAIL
                trial['result'].pop('loss', None)
                trial['result'].pop('decisions', None)
        self.trials.refresh()

    def do_job_error(self, tid):
        for trial in self.trials._dynamic_trials:
            if trial['tid'] == tid:
                assert trial['state'] == hyperopt.JOB_STATE_NEW
                trial['state'] = hyperopt.JOB_STATE_ERROR
                trial['misc']['error'] = (
                        str(type(Exception)),
                        'TestAsyncError crashed your trial for lolz')
        self.trials.refresh()

    def assert_counts(self, new, inprog, done, error):
        """
        Assert that self.trials has exactly the corresponding number of trials
        in each state.

        Pass None to ignore the count of a state.
        """
        if new is not None:
            assert new == self.trials.count_by_state_unsynced(
                hyperopt.JOB_STATE_NEW)
        if inprog is not None:
            assert inprog == self.trials.count_by_state_unsynced(
                hyperopt.JOB_STATE_RUNNING)
        if done is not None:
            assert done == self.trials.count_by_state_unsynced(
                hyperopt.JOB_STATE_DONE)
        if error is not None:
            assert error == self.trials.count_by_state_unsynced(
                hyperopt.JOB_STATE_ERROR)

    def get_cmds(self):
        return (self.push_job, self.do_job_ok, self.do_job_fail,
                self.do_job_error, self.assert_counts)

    #
    # Test various event sequences on various algos
    #

    def test_parallel_algo_0(self):
        self.algo = experiments.ParallelAlgo(self.sub_algo, 5)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()

        # -- test that even if some of the first set of jobs are done async
        #    that the first member of each process gets no data to work from.
        push(0, [])
        do_ok(0)
        push(1, [])
        assert_counts(1, 0, 1, 0)
        push(2, [])
        assert_counts(2, 0, 1, 0)
        push(3, [])
        assert_counts(3, 0, 1, 0)
        do_ok(2)
        assert_counts(2, 0, 2, 0)
        do_ok(1)
        assert_counts(1, 0, 3, 0)
        push(4, [])
        assert_counts(2, 0, 3, 0)
        do_ok(4)
        assert_counts(1, 0, 4, 0)

        # -- at this point, 5 tracks (procs?) of the ParallelAlgo should start passing
        #    some evidence, if the corresponding trials are done

        push(5, [0])
        assert_counts(2, 0, 4, 0)
        push(6, [1])
        assert_counts(3, 0, 4, 0)
        push(7, [2])
        assert_counts(4, 0, 4, 0)
        push(8, [3])
        assert_counts(5, 0, 4, 0)
        push(9, [4])
        assert_counts(6, 0, 4, 0)

    def test_parallel_algo_1(self):
        self.algo = experiments.ParallelAlgo(self.sub_algo, 5)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()

        push(0, [])
        assert_counts(1, 0, 0, 0)
        do_err(0)
        assert_counts(0, 0, 0, 1)
        push(1, [])
        assert_counts(1, 0, 0, 1)
        do_ok(1)
        assert_counts(0, 0, 1, 1)
        push(2, [])
        assert_counts(1, 0, 1, 1)
        do_ok(2)
        assert_counts(0, 0, 2, 1)
        push(3, [])
        assert_counts(1, 0, 2, 1)
        push(4, [])
        assert_counts(2, 0, 2, 1)
        do_ok(3)
        assert_counts(1, 0, 3, 1)
        push(5, [])
        assert_counts(2, 0, 3, 1)
        push(6, [1])
        assert_counts(3, 0, 3, 1)
        push(7, [2])
        assert_counts(4, 0, 3, 1)
        do_err(4)
        assert_counts(3, 0, 3, 2)
        push(8, [])
        push(9, [3])
        push(10, [8])
        push(11, [5])

    def test_asyncA_0(self):
        self.algo = experiments.AsyncBoostingAlgoA(self.sub_algo, round_len=4)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()

        # XXX
        # -- test that even if some of the first set of jobs are done async
        #    that the first member of each process gets no data to work from.
        raise nose.SkipTest()

    def test_asyncA_1(self):
        self.algo = experiments.AsyncBoostingAlgoA(self.sub_algo, round_len=4)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()
        # XXX
        # put in some errors in the first round
        raise nose.SkipTest()

    def test_asyncB_0(self):
        self.algo = experiments.AsyncBoostingAlgoB(self.sub_algo, round_len=4)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()
        raise nose.SkipTest()

    def test_asyncB_1(self):
        self.algo = experiments.AsyncBoostingAlgoB(self.sub_algo, round_len=4)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()
        # XXX
        # put in some errors in the first round
        raise nose.SkipTest()

    def test_sync_0(self):
        self.algo = experiments.SyncBoostingAlgo(self.sub_algo, round_len=4)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()
        raise nose.SkipTest()

    def test_sync_1(self):
        self.algo = experiments.SyncBoostingAlgo(self.sub_algo, round_len=4)
        push, do_ok, do_fail, do_err, assert_counts = self.get_cmds()
        # XXX
        # put in some errors in the first round
        raise nose.SkipTest()


# XXX TEST WITH SPORADIC FAILURES
