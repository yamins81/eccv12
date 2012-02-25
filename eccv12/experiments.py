"""
Experiment classes
"""

import logging
import sys
logger = logging.getLogger(__name__)

import numpy as np
import hyperopt
from hyperopt.base import trials_from_docs


def filter_oks(specs, results, miscs):
    ok_idxs = [ii for ii, result in enumerate(results)
            if result['status'] == hyperopt.STATUS_OK]
    specs = [specs[ii] for ii in ok_idxs]
    results = [results[ii] for ii in ok_idxs]
    miscs = [miscs[ii] for ii in ok_idxs]
    return specs, results, miscs


def filter_ok_trials(trials):
    return filter_oks(trials.specs, trials.results, trials.miscs)


##############
###MIXTURES###
##############

class NotEnoughTrialsError(Exception):
    def __init__(self, A, N):
        self.msg = 'Not enough ok trials: requires %d, has %d.' % (A, N)


class SimpleMixture(object):
    """
    Uses a top-A heuristic to select an ensemble from among completed trials.
    """
    def __init__(self, trials, bandit):
        self.trials = trials
        self.bandit = bandit

    def mix_inds(self, A):
        """Identify the top `A` trials.

        Return list of positions in self.trials, list of weights.
        """
        specs, results, miscs = filter_ok_trials(self.trials)

        if len(results) < A:
            raise NotEnoughTrialsError(A, len(results))
        specs = self.trials.specs
        losses = np.array(map(self.bandit.loss, results, specs))
        s = losses.argsort()
        return s[:A], np.ones((A,)) / float(A)

    def mix_models(self, A, **kwargs):
        """Identify the top `A` trials.

        Return list of specs, list of weights.
        """
        specs, results, miscs = filter_ok_trials(self.trials)
        inds, weights = self.mix_inds(A, **kwargs)
        return [specs[ind] for ind in inds], weights


class AdaboostMixture(SimpleMixture):
    """
    Uses AdaBoost to select an ensemble from among completed trials.
    """

    def __init__(self, bandit, bandit_algo, test_mask):
        """
        test_mask - a BOOLEAN, True means use a test_mask
        """
        SimpleMixture.__init__(self, bandit, bandit_algo)
        self.test_mask = test_mask

    def fetch_labels(self):
        """Return the 1D vector of +-1 labels for all examples.
        """
        specs, results, miscs = filter_ok_trials(self.trials)
        # -- take a detour and do a sanity check:
        #    All trials should have actually stored the *same* labels.
        labels = np.array([_r['labels'] for _r in results])
        assert labels.ndim == 2
        assert (labels == labels[0]).all()
        labels = labels[0]
        assert set(np.unique(labels)) == set([-1, 1])

        split_mask = np.array([_r['is_test'] for _r in results]).astype(int)
        assert (split_mask == split_mask[0]).all()
        split_mask = split_mask[0]
        assert split_mask.shape[1] == len(labels)
        assert set(np.unique(split_mask)) <= set([0, 1]), set(np.unique(split_mask))
        return labels, split_mask

    def fetch_decisions(self):
        """Return the 3D decisions array of each (trial, split, example)
        """
        specs, results, miscs = filter_ok_trials(self.trials)
        decisions = np.array([_r['decisions'] for _r in results])
        assert decisions.ndim == 3
        return decisions

    def predictions_from_decisions(self, decisions):
        """Return 3D prediction array (elements in +-1)
        """
        return np.sign(decisions).astype(np.int)

    def fetch_predictions(self):
        """Return 3D prediction array (elements in +-1)
        """
        decisions = self.fetch_decisions()
        return self.predictions_from_decisions(decisions)

    def mix_inds(self, A):
        """Identify `A` trials to use in boosted ensemble

        Return (list of positions in self.trials), (list of weights).
        """
        specs, results, miscs = filter_ok_trials(self.trials)
        if len(results) < A:
            raise NotEnoughTrialsError(A, len(results))
        labels, split_mask = self.fetch_labels()
        L = len(labels)
        predictions = self.fetch_predictions()
        assert predictions.shape[2] == L, '%d != %d' % (predictions.shape[2], L)
        assert predictions.shape[1:] == split_mask.shape, '%d,%d != %d,%d' % (
                predictions.shape[1:], split_mask.shape)
        # -- compute a 3D array errors on all trials, splits, examples.
        errors = (predictions != labels).astype(np.int)
        selected_inds = []
        alphas = []
        for round in range(A):
            # -- set weights across examples for each split
            if round:
                prediction = predictions[selected_inds[-1]]
                weights *= np.exp(-alpha[:, np.newaxis] * labels * prediction)
            else:
                weights = np.ones(predictions.shape[1:])
                if self.test_mask:
                    weights *= split_mask
            weights /= weights.sum(1)[:, np.newaxis]

            # -- weighted error rate for each trial, split
            ep_array = (errors * weights).sum(2)
            ep_diff_array = np.abs(0.5 - ep_array)

            # -- pick the trial whose mean across splits is best
            ind = ep_diff_array.mean(1).argmax()
            selected_inds.append(ind)

            # -- determine the weight of the new ensemble member
            #    (within in eatch split... alpha is vector here)
            ep = ep_array[ind]
            alpha = 0.5 * np.log((1 - ep) / ep)
            alphas.append(alpha)

        return np.array(selected_inds), np.array(alphas)



##############
###PARALLEL###
##############

class InterleaveAlgo(hyperopt.BanditAlgo):
    """Interleaves calls to `num_procs` independent Trials sets

    This class is implemented carefully to work around some awkwardness
    in the design of hyperopt.Trials.  The purpose of this banditalgo is to
    facilitate running several independent BanditAlgos at once.  You might
    want to do this if you are trying to compare random search to an optimized
    search for example.

    Trial documents inserted by this suggest function can be tagged with
    identifying information, but trial documents inserted in turn by the
    Bandit.evaluate() functions themselves will not be tagged, except by their
    experiment key (exp_key). So this class uses the exp_key to keep each
    experiment distinct.  Every sub_algo passed to the constructor requires a
    corresponding exp_key. If you want to get tricky, you can combine
    sub-experiments by giving them identical keys. As a consequence of this
    strategy, this class REQUIRES AN UNRESTRICTED VIEW of the trials object
    used to make suggestions, so the trials object CANNOT have an exp_key
    of its own.

    The InterleaveAlgo works on the basis of growing the sub-experiments at
    the same pace. On every call to suggest(), this function counts up the
    number of non-error jobs in each sub-experiment, and asks the sub_algo
    corresponding to the smallest sub-experiment to propose a new document.

    The InterleaveAlgo stops when all of the sub_algo.suggest methods have
    returned `hyperopt.StopExperiment`.

    """
    def __init__(self, sub_algos, sub_exp_keys, **kwargs):
        hyperopt.BanditAlgo.__init__(self, sub_algos[0].bandit, **kwargs)
        # XXX: assert all bandits are the same
        self.sub_algos = sub_algos
        self.sub_exp_keys = sub_exp_keys
        if len(sub_algos) != len(sub_exp_keys):
            raise ValueError('algos and keys should have same len')
        # -- will be rebuilt naturally if experiment is continued
        self.stopped = set()

    def suggest(self, new_ids, trials):
        assert trials._exp_key is None # -- see explanation above
        sub_trials = [trials.view(exp_key, refresh=False)
                for exp_key in self.sub_exp_keys]
        # -- views are not refreshed
        states_that_count = [
                hyperopt.JOB_STATE_NEW,
                hyperopt.JOB_STATE_RUNNING,
                hyperopt.JOB_STATE_DONE]
        counts = [st.count_by_state_unsynced(states_that_count)
                for st in sub_trials]
        logger.info('counts: %s' % str(counts))
        new_docs = []
        for new_id in new_ids:
            pref = np.argsort(counts)
            # -- try to get one of the sub_algos to make a suggestion
            #    for new_id, in order of sub-experiment size
            for active in pref:
                if active not in self.stopped:
                    sub_algo = self.sub_algos[active]
                    sub_trial = sub_trials[active]
                    # XXX This may well transfer data... make sure that's OK
                    #     In future consider adding a refresh=False
                    #     to constructor, to prevent this transfer.
                    sub_trial.refresh()
                    smth = sub_algo.suggest([new_id], sub_trial)
                    if smth is hyperopt.StopExperiment:
                        logger.info('stopping experiment (%i: %s)' %
                                    (active, sub_algo))
                        self.stopped.add(active)
                    elif smth:
                        logger.info('suggestion %i from (%i: %s)' %
                                    (new_id, active, sub_algo))
                        new_doc, = smth
                        counts[active] += 1
                        new_docs.append(new_doc)
                        break
                    else:
                        if list(smth) != []:
                            raise ValueError('bad suggestion',
                                    (sub_algo, smth))
        if len(self.stopped) == len(self.sub_algos):
            return hyperopt.StopExperiment
        else:
            return new_docs


def ParallelAlgo(sub_algo, num_progs, **kwargs):
    sub_algos = [sub_algo] * num_progs
    sub_keys = ['EK_%i' % i for i in range(num_progs)]
    return InterleaveAlgo(sub_algos, sub_keys, **kwargs)


##############
###BOOSTING###
##############

###XXXX In all these, consider using filter_ok just for purposes of picking
###a thing to continue .. but passing on all the passed stuff to sub bandit algo


class BoostingAlgoBase(hyperopt.BanditAlgo):

    # -- keep these methods static to avoid temptation to put info into self
    # without reason. It's important that a new instance be able to function
    # as stateless-ly as possible so that it can work on new trials objects.

    @staticmethod
    def idxs_continuing(miscs, tid):
        """Return the positions in the trials object of
        all trials that continue trial tid.
        """
        rval = [idx for idx, misc in enumerate(miscs)
                if misc['boosting']['continues'] == tid]
        return rval

    @staticmethod
    def boosting_best_by_round(trials, bandit):
        specs, results, miscs = filter_ok_trials(trials)
        losses = np.array(map(bandit.loss, results, specs))
        rounds = np.array([m['boosting']['round'] for m in miscs])
        urounds = np.unique(rounds)
        urounds.sort()
        assert urounds.tolist() == range(urounds.max() + 1)
        rval = []
        for u in urounds:
            _inds = (rounds == u).nonzero()[0]
            min_ind = losses[_inds].argmin()
            rval.append(specs[_inds[min_ind]])
        return rval

    @staticmethod
    def ensemble_member_tids(trials, bandit):
        """
        Return the list of tids of members selected by the boosting algorithm.
        """
        specs, results, miscs = filter_ok_trials(trials)
        losses = np.array(map(bandit.loss, results, specs))
        assert None not in losses
        cur_idx = np.argmin(losses)
        reversed_members_idxs = [cur_idx]
        while miscs[cur_idx]['boosting']['continues'] != None:
            cur_idx = [m['tid'] for m in miscs].index(
                    miscs[cur_idx]['boosting']['continues'])
            reversed_members_idxs.append(cur_idx)
        rval = [miscs[ii]['tid'] for ii in reversed(reversed_members_idxs)]
        return rval


class AsyncBoostingAlgo(BoostingAlgoBase):
    """
    sub_algo

    round_len - start a new round when there are at least this
        many trials in the latest round.

    look_back - positive integer
        1 requires continuing a member of the previous round
        2 allows continuing a member among previous 2 rounds...
        3 ...

    """
    def __init__(self, sub_algo, round_len, look_back, **kwargs):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit, **kwargs)
        self.sub_algo = sub_algo
        self.round_len = round_len
        self.look_back = look_back

    def suggest(self, new_ids, trials):
        specs, results, miscs = trials.specs, trials.results, trials.miscs

        assert len(specs) == len(results) == len(miscs)
        if len(new_ids) > 1:
            raise NotImplementedError()

        specs, results, miscs = filter_oks(specs, results, miscs)

        round_len = self.round_len
        look_back = self.look_back

        my_round = 0
        cont_decisions = None
        cont_tid = None
        others = []

        if miscs:
            # -- pick a trial to continue
            rounds_counts = np.bincount([m['boosting']['round']
                for m in miscs])
            assert np.all(rounds_counts > 0)
            assert np.all(rounds_counts[:-1] >= round_len)
            # -- this is the round of the trial we're going to suggest
            if rounds_counts[-1] >= round_len:
                my_round = len(rounds_counts)
            else:
                my_round = len(rounds_counts) - 1
            horizon = my_round - look_back
            consider_continuing = [idx
                    for idx, misc in enumerate(miscs)
                    if horizon <= misc['boosting']['round'] < my_round]

            #print 'losses', np.array(map(self.bandit.loss, results, specs))

            if consider_continuing:
                cc_specs = [specs[idx] for idx in consider_continuing]
                cc_results = [results[idx] for idx in consider_continuing]
                cc_miscs = [miscs[idx] for idx in consider_continuing]

                cc_losses = np.array(map(self.bandit.loss, cc_results, cc_specs))
                cont_idx = cc_losses.argmin()

                cont_decisions = cc_results[cont_idx]['decisions']
                cont_tid = cc_miscs[cont_idx]['tid']
                assert cont_tid != None
                others = self.idxs_continuing(miscs, cont_tid)
            else:
                others = self.idxs_continuing(miscs, None)


        continuing_trials_docs = [trials.trials[idx] for idx in others]
        continuing_trials = trials_from_docs(continuing_trials_docs,
                                                     exp_key=trials._exp_key)

        new_trial_docs = self.sub_algo.suggest(new_ids, continuing_trials)

        for trial in new_trial_docs:
            # -- patch in decisions of the best current model from previous
            #    round
            # -- This is an assertion because the Bandit should be written
            #    to use these values, and thus be written with the awareness
            #    that they are coming...
            spec = trial['spec']
            assert spec['decisions'] == None
            spec['decisions'] = cont_decisions

            misc = trial['misc']
            assert 'boosting' not in misc
            misc['boosting'] = {
                    'variant': 'sync',
                    'round': my_round,
                    'continues': cont_tid}

        return new_trial_docs


class AsyncBoostingAlgoA(AsyncBoostingAlgo):
    def __init__(self, bandit_algo, round_len, **kwargs):
        AsyncBoostingAlgo.__init__(self, bandit_algo, round_len,
                look_back=1, **kwargs)


class AsyncBoostingAlgoB(AsyncBoostingAlgo):
    def __init__(self, bandit_algo, round_len, **kwargs):
        AsyncBoostingAlgo.__init__(self, bandit_algo, round_len,
                look_back=sys.maxint, **kwargs)


class SyncBoostingAlgo(BoostingAlgoBase):
    def __init__(self, sub_algo, round_len, **kwargs):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit, **kwargs)
        self.sub_algo = sub_algo
        self.round_len = round_len

    def suggest(self, new_ids, trials):
        round_len = self.round_len

        specs, results, miscs = filter_ok_trials(trials)

        if miscs:
            rounds = [m['boosting']['round'] for m in miscs]
            # -- actually the rounds of completed trials
            complete_rounds = [m['boosting']['round']
                    for m, r in zip(miscs, results) if 'loss' in r]

            max_round = max(rounds)
            urounds = np.unique(rounds)
            urounds.sort()
            assert list(urounds) == range(max_round + 1)

            rounds_counts = [rounds.count(j) for j in urounds]
            complete_rounds_counts = [complete_rounds.count(j)
                    for j in urounds]
            assert all([rc == crc >= round_len
                for crc, rc in zip(rounds_counts[:-1],
                    complete_rounds_counts[:-1])])

            round_decs = [[s['decisions']
                for m, s in zip(miscs, specs)
                if m['boosting']['round'] == j] for j in urounds]
            assert all([all([_rd == rd[0]
                for _rd in rd]) for rd in round_decs])
            round_decs = [rd[0] for rd in round_decs]

            if complete_rounds_counts[-1] >= round_len:
                my_round = max_round + 1
                last_specs = [s
                        for s, m in zip(specs, miscs)
                        if m['boosting']['round'] == max_round]
                last_results = [s
                        for s, m in zip(results, miscs)
                        if m['boosting']['round'] == max_round]
                last_miscs = [m for m in miscs
                        if m['boosting']['round'] == max_round]
                losses = np.array(map(self.bandit.loss, last_results, last_specs))
                last_best = losses.argmin()
                decisions = last_results[last_best]['decisions']
                decisions_src = last_miscs[last_best]['tid']
            else:
                my_round = max_round
                decisions = round_decs[-1]
                decisions_src = miscs[-1]['boosting']['continues']
        else:
            decisions = None
            my_round = 0
            decisions_src = None

        selected_trial_docs = [t for t in trials
                                  if t['misc']['boosting']['round'] == my_round]        
        selected_trials = trials_from_docs(selected_trial_docs,
                                                  exp_key=trials._exp_key)
        new_trial_docs = self.sub_algo.suggest(new_ids, selected_trials)

        for trial in new_trial_docs:
            # -- patch in decisions of the best current model from previous
            #    round
            spec = trial['spec']
            assert spec['decisions'] == None
            spec['decisions'] = decisions

            misc = trial['misc']
            misc['boosting'] = {
                    'variant': 'sync',
                    'round': my_round,
                    'continues': decisions_src}

        return new_trial_docs

