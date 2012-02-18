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
        specs = self.trials.specs
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
        num_procs = self.num_procs
        proc_nums = [s['proc_num'] for s in miscs]
        proc_counts = np.array([proc_nums.count(j) for j in range(num_procs)])
        proc_num = proc_counts.argmin()
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

class BoostingAlgoBase(hyperopt.BanditAlgo):
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


class AsyncBoostingAlgo(BoostingAlgoBase):
    def __init__(self, sub_algo, round_len, look_back=1):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.round_len = round_len
        self.look_back = look_back

    def suggest(self,
            new_ids,
            specs,
            results,
            miscs):
        assert len(specs) == len(results) == len(miscs)
        round_len = self.round_len
        look_back = self.look_back
       
        cont_decisions = None
        cont_id = None
        selected_results = []
        selected_specs = []
        selected_miscs = []
        my_round = 0
        if miscs:
            rounds = [m['round'] for m in miscs]
            max_round = max(rounds)
            urounds = np.unique(rounds)
            urounds.sort()
            assert list(urounds) == range(max_round+1)
            rounds_counts = [rounds.count(j) for j in urounds]
            assert all([rc >= round_len for rc in rounds_counts[:-1]])
            if rounds_counts[-1] >= round_len:
                my_round = max_round + 1
            else:
                my_round = max_round
            consider_continuing = []
            for idx, x in enumerate(miscs):
                if my_round - look_back <= x['round'] < my_round:
                    if 'loss' in results[idx]:  #better way to check completion?
                        consider_continuing.append(idx)
            if consider_continuing:
                cc_results = [results[idx] for idx in consider_continuing]
                cc_specs = [specs[idx] for idx in consider_continuing]
                cc_miscs = [miscs[idx] for idx in consider_continuing]
                cc_losses = np.array(map(self.bandit.loss, cc_results, cc_specs))
                cont_idx = cc_losses.argmin()
                cont_misc = cc_miscs[cont_idx]
                cont_id = cont_misc['id']
                cont_res = cc_results[cont_idx]
                cont_decisions = cont_res['decisions']
                selected = [idx for idx, m in enumerate(cc_miscs) if m['continues'] == cont_id]
                selected_results = [cc_results[idx] for idx in selected]
                selected_specs = [cc_specs[idx] for idx in selected]
                selected_miscs = [cc_miscs[idx] for idx in selected]
        new_specs, new_results, new_miscs = self.sub_algo.suggest(new_ids,
                selected_specs,
                selected_results,
                selected_miscs)
        
        for spec in new_specs:
            # -- patch in decisions of the best current model from previous
            #    round
            assert spec['decisions'] == None
            spec['decisions'] = cont_decisions
        
        if miscs:
            id = max([m['id'] for m in miscs]) + 1
        else:
            id = 0
        for misc in new_miscs:
            misc['round'] = my_round
            misc['continues'] = cont_id
            misc['id'] = id
            
        return new_specs, new_results, new_miscs


class SyncBoostingAlgo(BoostingAlgoBase):
    def __init__(self, sub_algo, round_len):
        hyperopt.BanditAlgo.__init__(self, sub_algo.bandit)
        self.sub_algo = sub_algo
        self.round_len = round_len

    def suggest(self,
            new_ids,
            specs,
            results,
            miscs):
        assert len(specs) == len(results) == len(miscs)
        round_len = self.round_len
        
        if miscs:
            rounds = [m['round'] for m in miscs]
            complete_rounds = [m['round'] for m, r in zip(miscs, results) if 'loss' in r]

            max_round = max(rounds)
            urounds = np.unique(rounds)
            urounds.sort()
            assert list(urounds) == range(max_round+1)
            
            rounds_counts = [rounds.count(j) for j in urounds]
            complete_rounds_counts = [complete_rounds.count(j) for j in urounds]          
            assert all([rc == crc >= round_len for crc, rc in zip(rounds_counts[:-1], complete_rounds_counts[:-1])])
            
            round_decs = [[s['decisions'] for m, s in zip(miscs, specs) if m['round'] == j] for j in urounds]
            assert all([all([_rd == rd[0] for _rd in rd]) for rd in round_decs])
            round_decs = [rd[0] for rd in round_decs]
            
            if complete_rounds_counts[-1] >= round_len:
                my_round = max_round + 1
                last_specs = [s for s, m in zip(specs, miscs) if m['round'] == max_round]
                last_results = [s for s, m in zip(results, miscs) if m['round'] == max_round]
                losses = np.array(map(self.bandit.loss, last_results, last_specs))
                last_best = losses.argmin()
                decisions = last_results[last_best]['decisions']
            else:
                my_round = max_round
                decisions = round_decs[-1]
        else:
            decisions = None
            my_round = 0

        selected_specs = [s for s, m in zip(specs, miscs) if m['round'] == my_round]
        selected_results = [s for s, m in zip(results, miscs) if m['round'] == my_round]
        selected_miscs = [m for m in miscs if m['round'] == my_round]

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
            misc['round'] = my_round
            
        return new_specs, new_results, new_miscs

