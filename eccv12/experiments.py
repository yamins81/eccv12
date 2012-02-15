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
        losses = np.array(map(self.bandit.loss, results, specs))
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
        assert decisions.ndim == 3
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
        assert predictions.shape[2] == L, '%d != %d' % (predictions.shape[2], L)
        weights = (1./L) * np.ones((predictions.shape[1], L))
        labels = labels[np.newaxis, :]  
        errors = (predictions != labels).astype(np.int)
        selected_inds = []
        alphas = []
        for round in range(A):
            ep_array = (errors * weights).sum(2)
            ep_diff_array = np.abs(0.5 - ep_array)
            ep_diff_array[selected_inds] = -1 #hacky way to prevent re-selection
            ind = ep_diff_array.mean(1).argmax()
            selected_inds.append(ind)
            ep = ep_array[ind]
            alpha = 0.5 * np.log((1 - ep) / ep)
            alpha = alpha.reshape((len(alpha), 1))
            alphas.append(alpha)
            prediction = predictions[ind]
            weights = weights * np.exp(-alpha * labels * prediction)
            weights = weights / weights.sum(1).reshape((weights.shape[0], 1))
        alphas = np.array(alphas)
        return np.array(selected_inds), alphas.reshape(alphas.shape[:2])



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
            miscs):
        trial_num = len(specs)
        proc_num = trial_num % self.num_procs
        proc_idxs = [idx for idx, s in enumerate(miscs) if s['proc_num'] == proc_num]
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
            miscs):
        assert len(specs) == len(results) == len(miscs)
        n_trials = len(specs)
        cutoff = (n_trials // self.round_len) * self.round_len

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
        new_specs, new_results, new_miscs = self.sub_algo.suggest(new_ids,
                specs[cutoff:],
                results[cutoff:],
                miscs[cutoff:])
        for spec in new_specs:
            # -- patch in decisions of the best current model from previous
            #    round
            assert spec['decisions'] == None
            spec['decisions'] = decisions
        return new_specs, new_results, new_miscs


