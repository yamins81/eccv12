"""
Experiment classes
"""

import os
import sys
import cPickle
import hashlib

import numpy as np
import hyperopt


##############
###MIXTURES###
##############

class SimpleMixture(object):
    def __init__(self, trials, bandit):
        self.trials = trials
        self.bandit = bandit

    def mix_inds(self, A):
        results = self.trials.results
        assert len(results) >= A
        specs = self.trials.specs
        losses = map(self.bandit.loss, results[:cutoff], specs[:cutoff])
        s = losses.argsort()
        return s[:A], np.ones((A,)) / float(A)

    def mix_models(self, A):
        specs = self.trials.sepcs
        inds, weights = self.mix_inds(A)
        return [specs[ind] for ind in inds], weights


class AdaboostMixture(SimpleMixture):
    def fetch_labels(self):
        results = self.trials.results
        labels = np.array([_r['labels'] for _r in results])
        assert (labels == labels[0]).all()
        assert labels.ndim == 2
        assert set(np.unique(labels)) == set([-1, 1])
        labels = labels[0]
        return labels

    def fetch_decisions(self):
        results = self.trials.results
        decisions = np.array([_r['decisions'] for _r in results])
        assert decisions.ndim in 3
        return decisions

    def predictions_from_decisions(self, decisions):
        return np.sign(decisions).astype(np.int)

    def fetch_predictions(self):
        decisions = self.fetch_decisions()
        return self.predictions_from_decisions(decisions)

    def mix_inds(self, A):
        results = self.trials.results
        assert len(results) >= A
        labels = self.fetch_labels()
        L = len(labels)
        predictions = self.fetch_predictions()
        assert predictions.shape[1] == L
        weights = (1./L) * np.ones((L, predictions.shape[2]))
        labels = labels[:, np.newaxis]        
        errors = (predictions != labels).astype(np.int)
        selected_inds = []
        alphas = []
        for round in range(A):
            ep_array = (errors * weights).sum(1)
            ep_diff_array = np.abs(0.5 - ep_array)
            ind = ep_diff_array.mean(1).argmax()
            selected_inds.append(ind)
            ep = ep_array[ind]
            alpha = 0.5 * np.log((1 - ep) / ep)
            alphas.append(alpha)
            prediction = predictions[ind]
            weights = weights * np.exp(-alpha * labels * prediction)
            weights = weights / weights.sum(0)
        alphas = np.array(alphas)
        return selected_inds, np.array(alphas)



##############
###PARALLEL###
##############

class ParallelAlgo(hyperopt.BanditAlgo):
    def __init__(self, sub_algo, num_procs):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.num_procs = num_procs

    def suggest(self,
            new_ids,
            specs,
            results,
            stochastic_idxs,
            stochastic_vals):
        trial_num = len(specs)
        proc_num = trial_num % self.num_procs
        proc_specs = [s for s, r in zip(specs, results) if r['proc_num'] == proc_num]
        proc_results = [_res in results if res['proc_num'] == proc_num]
        proc_idxs = {}
        proc_vals = {}
        for key in stochastic_idxs:
            proc_idxs[key] = [idx
                    for idx in stochastic_idxs[key]
                    if idx >= cutoff]
            proc_vals[key] = [val
                    for val, idx in zip(stochastic_vals[key], stochastic_idxs[key])
                    if idx >= cutoff]
        docs, idxs, vals = self.sub_algo.suggest(new_ids, proc_specs,
                                          proc_results, round_idxs, round_vals)
        for doc in docs:
            assert 'proc_num' not in doc
            doc['proc_num'] = proc_num
        return docs, idxs, vals


##############
###BOOSTING###
##############

class BoostingAlgo(hyperopt.BanditAlgo):
    def __init__(self, sub_algo, round_len):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.round_len = round_len

    def best_by_round(self, trials):
        ii = 0
        rval = []
        while ii < len(trials):
            rtrials = trials[ii:ii + self.round_len]

            # -- deal with error / unfinished trials
            # XXX: this assumes that triald id == position in specs list
            #      It is currently true, but need not always be!!
            # XXX: REFACTOR WITH CODE BELOW
            results_ii = [tt['result'] for tt in rtrials]
            specs_ii = [tt['spec'] for tt in rtrials]
            tids_losses = enumerate(map(self.bandit.loss, results_ii, specs_ii))
            losses_tids = [(loss, tid)
                    for (tid, loss) in tids_losses
                    if loss != None]
            losses_tids.sort()
            selected_ind = losses_tids[0][1]
            rval.append(rtrials[selected_ind])
            ii += self.round_len
        return rval

    def suggest(self,
            new_ids,
            specs,
            results,
            stochastic_idxs,
            stochastic_vals):
        n_trials = len(specs)
        cutoff = (n_trials // self.round_len) * self.round_len
        round_specs = specs[cutoff:]
        round_results = results[cutoff:]
        round_idxs = {}
        round_vals = {}
        for key in stochastic_idxs:
            round_idxs[key] = [idx
                    for idx in stochastic_idxs[key]
                    if idx >= cutoff]
            round_idxs[key] = [val
                    for val, idx in zip(stochastic_vals[key], stochastic_idxs[key])
                    if idx >= cutoff]
        if cutoff:
            # -- deal with error / unfinished trials
            # XXX: this assumes that triald id == position in specs list
            #      It is currently true, but need not always be!!
            tids_losses = enumerate(map(self.bandit.loss, results[:cutoff], specs[:cutoff]))
            losses_tids = [(loss, tid)
                    for (tid, loss) in tids_losses
                    if loss != None]
            losses_tids.sort()
            selected_ind = losses_tids[0][1]
            decisions = results[selected_ind]['decisions']
        else:
            decisions = None
        docs, idxs, vals = self.sub_algo.suggest(new_ids, round_specs, round_results,
                round_idxs, round_vals)
        for doc in docs:
            # -- patch in decisions of the best current model from previous
            #    round
            assert doc['decisions'] == None
            doc['decisions'] = decisions
        return docs, idxs, vals


