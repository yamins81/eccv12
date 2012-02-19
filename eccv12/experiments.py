"""
Experiment classes
"""

import cPickle
import functools
import hashlib
import os
import sys

import numpy as np
import hyperopt


##############
###MIXTURES###
##############

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
        results = self.trials.results
        assert len(results) >= A
        specs = self.trials.specs
        losses = np.array(map(self.bandit.loss, results, specs))
        s = losses.argsort()
        return s[:A], np.ones((A,)) / float(A)

    def mix_models(self, A):
        """Identify the top `A` trials.

        Return list of specs, list of weights.
        """
        specs = self.trials.specs
        inds, weights = self.mix_inds(A)
        return [specs[ind] for ind in inds], weights


class AdaboostMixture(SimpleMixture):
    """
    Uses AdaBoost to select an ensemble from among completed trials.
    """
    def fetch_labels(self):
        """Return the 1D vector of +-1 labels for all examples.
        """
        results = self.trials.results
        # -- take a detour and do a sanity check:
        #    All trials should have actually stored the *same* labels.
        labels = np.array([_r['labels'] for _r in results])
        assert (labels == labels[0]).all()
        assert labels.ndim == 2
        assert set(np.unique(labels)) == set([-1, 1])
        labels = labels[0]
        return labels

    def fetch_decisions(self):
        """Return the 3D decisions array of each (trial, split, example)
        """
        results = self.trials.results
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
        results = self.trials.results
        assert len(results) >= A
        labels = self.fetch_labels()
        L = len(labels)
        predictions = self.fetch_predictions()
        assert predictions.shape[2] == L, '%d != %d' % (predictions.shape[2], L)
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
                weights = np.ones((predictions.shape[1], L))
            weights /= weights.sum(1)[:, np.newaxis]

            # -- weighted error rate for each trial, split
            ep_array = (errors * weights).sum(2)
            ep_diff_array = np.abs(0.5 - ep_array)

            # -- pick the trial whose mean across splits is best
            ind = ep_diff_array.mean(1).argmax()
            selected_inds.append(ind)

            # -- determine the weight of the new ensemble member
            ep = ep_array[ind]
            alpha = 0.5 * np.log((1 - ep) / ep)
            alphas.append(alpha)

        return np.array(selected_inds), np.array(alphas)



##############
###PARALLEL###
##############

class ParallelAlgo(hyperopt.BanditAlgo):
    """Interleaves calls to `num_procs` independent Trials sets
    
    Injects a `proc_num` field into the miscs document for book-keeping.
    """
    def __init__(self, sub_algo, num_procs):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.num_procs = num_procs

    def suggest(self, new_ids, specs, results, miscs):
        num_procs = self.num_procs
        proc_nums = [s['proc_num'] for s in miscs]
        proc_counts = np.array([proc_nums.count(j) for j in range(num_procs)])
        proc_num = int(proc_counts.argmin())
        proc_nums = np.array(proc_nums)
        proc_idxs = (proc_nums == proc_num).nonzero()[0]
        proc_specs = [specs[idx] for idx in proc_idxs]
        proc_results = [results[idx] for idx in proc_idxs]
        proc_miscs = [miscs[idx] for idx in proc_idxs]
        new_specs, new_results, new_miscs = self.sub_algo.suggest(new_ids,
                                           proc_specs, proc_results, proc_miscs)
        for doc in new_miscs:
            assert 'proc_num' not in doc
            doc['proc_num'] = proc_num
        return new_specs, new_results, new_miscs
        

##############
###BOOSTING###
##############

def filter_oks(specs, results, miscs):
    ok_idxs = [ii for ii, result in enumerate(results)
            if result['status'] == hyperopt.STATUS_OK]
    specs = [specs[ii] for ii in ok_idxs]
    results = [results[ii] for ii in ok_idxs]
    miscs = [miscs[ii] for ii in ok_idxs]
    return specs, results, miscs


class BoostingAlgoBase(hyperopt.BanditAlgo):
    def best_by_round(self, trials):
        results = [t['result'] for t in trials]
        specs = [t['spec'] for t in trials]
        miscs = [t['misc'] for t in trials]
        losses = np.array(map(self.bandit.loss, results, specs))
        rounds = np.array([m['boosting']['round'] for m in miscs])
        urounds = np.unique(rounds)
        urounds.sort()
        assert urounds.tolist() == range(urounds.max() + 1)
        rval = []
        for u in urounds:
            _inds = (rounds == u).nonzero()[0]
            min_ind = losses[_inds].argmin()
            rval.append(trials[_inds[min_ind]])
        return rval

    def idxs_continuing(self, miscs, tid):
        """Return the positions in the trials object of
        all trials that continue trial tid.
        """
        rval = [idx for idx, misc in enumerate(miscs)
                if misc['boosting']['continues'] == tid]
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
    def __init__(self, sub_algo, round_len, look_back):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.round_len = round_len
        self.look_back = look_back

    def suggest(self, new_ids, specs, results, miscs):
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
                    if (horizon <= misc['boosting']['round'] < my_round
                        and results[idx]['status'] == hyperopt.STATUS_OK)]

            if consider_continuing:
                cc_specs = [specs[idx] for idx in consider_continuing]
                cc_results = [results[idx] for idx in consider_continuing]
                cc_miscs = [miscs[idx] for idx in consider_continuing]

                cc_losses = np.array(map(self.bandit.loss, cc_results, cc_specs))
                cont_idx = cc_losses.argmin()

                cont_decisions = cc_results[cont_idx]['decisions']
                cont_tid = cc_miscs[cont_idx]['tid']
                others = self.idxs_continuing(miscs, cont_tid)

        new_specs, new_results, new_miscs = self.sub_algo.suggest(new_ids,
                [results[idx] for idx in others],
                [specs[idx] for idx in others],
                [miscs[idx] for idx in others])
        
        for spec in new_specs:
            # -- patch in decisions of the best current model from previous
            #    round
            # -- This is an assertion because the Bandit should be written
            #    to use these values, and thus be written with the awareness
            #    that they are coming...
            assert spec['decisions'] == None
            spec['decisions'] = cont_decisions
        
        for misc in new_miscs:
            assert 'boosting' not in misc
            misc['boosting'] = {
                    'variant': 'sync',
                    'round': my_round,
                    'continues': cont_tid}
            
        return new_specs, new_results, new_miscs


class AsyncBoostingAlgoA(AsyncBoostingAlgo):
    def __init__(self, bandit_algo, round_len):
        AsyncBoostingAlgo.__init__(self, bandit_algo, round_len,
                look_back=1)

class AsyncBoostingAlgoB(AsyncBoostingAlgo):
    def __init__(self, bandit_algo, round_len):
        AsyncBoostingAlgo.__init__(self, bandit_algo, round_len,
                look_back=sys.maxint)


class SyncBoostingAlgo(BoostingAlgoBase):
    def __init__(self, sub_algo, round_len):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.round_len = round_len

    def suggest(self, new_ids, specs, results, miscs):
        assert len(specs) == len(results) == len(miscs)
        round_len = self.round_len

        specs, results, miscs = filter_oks(specs, results, miscs)
        
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
                decisions_src = miscs[-1]['tid']
        else:
            decisions = None
            my_round = 0
            decisions_src = None

        selected_specs = [s
                for s, m in zip(specs, miscs)
                if m['boosting']['round'] == my_round]
        selected_results = [s
                for s, m in zip(results, miscs)
                if m['boosting']['round'] == my_round]
        selected_miscs = [m
                for m in miscs if m['boosting']['round'] == my_round]

        new_specs, new_results, new_miscs = self.sub_algo.suggest(new_ids,
                selected_specs,
                selected_results,
                selected_miscs)
        
        for spec in new_specs:
            # -- patch in decisions of the best current model from previous
            #    round
            assert spec['decisions'] == None
            spec['decisions'] = decisions
        
        for misc in new_miscs:
            misc['boosting'] = {
                    'variant': 'sync',
                    'round': my_round,
                    'continues': decisions_src}
            
        return new_specs, new_results, new_miscs

