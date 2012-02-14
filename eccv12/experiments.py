"""
Experiment classes
"""

import os
import sys
import cPickle
import hashlib
from math import log

import numpy as np

import hyperopt

###XXX:  Code in this module requires that experiment result records satisfy
###validation in bandit.py


class SerialMixin(object):
    def init_experiment(self, *args, **kwargs):
        bandit = self.bandit_class(*args, **kwargs)
        bandit_algo = self.bandit_algo_class(bandit)
        return hyperopt.Experiment(bandit_algo)


##############
###MIXTURES###
##############

class SimpleMixture(object):
    def __init__(self, trials, bandit):
        self.trials = trials
        self.bandit = bandit

    def mix_inds(self, A):
        specs = self.trials.specs
        results = self.trials.results
        assert len(results) >= A
        losses = map(self.bandit.loss, results, specs)
        if None in losses:
            raise NotImplementedError()
        s = np.asarray(losses).argsort()
        return s[:A], np.ones((A,)) / float(A)

    def mix_models(self, A):
        specs = self.trials.specs
        inds, weights = self.mix_inds(A)
        return [specs[ind] for ind in inds], weights


class AdaboostMixture(SimpleMixture):
    def fetch_labels(self, splitname):
        specs = self.trials.specs
        results = self.trials.results
        labels = np.array([_r['labels'][splitname] for _r in results])
        assert (labels == labels[0]).all()
        assert labels.ndim == 2
        assert set(np.unique(labels)) == set([-1, 1])
        labels = labels[0]
        return labels

    def fetch_decisions(self, splitname):
        exp = self.exp
        decisions = np.array([_r['decisions'][splitname] for _r in exp.results])
        assert decisions.ndim == 2
        return decisions

    def predictions_from_decisions(self, decisions):
        return np.sign(decisions).astype(np.int)

    def fetch_predictions(self, splitname):
        decisions = self.fetch_decisions(splitname)
        return self.predictions_from_decisions(decisions)

    def mix_inds(self, A, splitname):
        exp = self.exp
        assert len(exp.results) >= A
        labels = self.fetch_labels(splitname)
        predictions = self.fetch_predictions(splitname)
        errors = (predictions != labels).astype(np.int)
        L = len(labels)
        weights = (1./L) * np.ones((L,))
        selected_inds = []
        alphas = []
        for round in range(A):
            ep_array = np.dot(errors, weights)
            ep_diff_array = np.abs(0.5 - ep_array)
            ind = ep_diff_array.argmax()
            selected_inds.append(ind)
            ep = ep_array[ind]
            alpha = 0.5 * log((1 - ep) / ep)
            alphas.append(alpha)
            prediction = np.array(predictions[ind])
            weights = weights * np.exp(-alpha * labels * prediction)
            weights = weights / weights.sum()
        return selected_inds, np.array(alphas)



##############
###PARALLEL###
##############

class ParallelExperiment(hyperopt.base.Experiment):
    def __init__(self, bandit_algo_class, bandit_class, num_proc, opt_runs,
                 proc_args=None, **init_kwargs):
        self.bandit_algo_class = bandit_algo_class
        self.bandit_class = bandit_class
        self.num_proc = num_proc
        self.opt_runs = opt_runs
        self.experiments = None
        self.trials = []
        self.results = []
        if proc_args is not None:
            assert len(proc_args) == num_proc
        else:
            proc_args = [((),{}) for _ind in range(num_proc)]
        for _ind, (a, b) in enumerate(proc_args):
            b['parallel_round'] = _ind
        self.proc_args = proc_args
        for k, v in init_kwargs.items():
            setattr(self, k, v)

    def run(self):
        if self.experiments is None:
            for a, b in self.proc_args:
                self.experiments.append(self.init_experiment(*a,**b))
        num_dones = np.array([len(exp.results) for exp in self.experiments])
        num_lefts = self.opt_runs - num_dones
        for exp, num_left in zip(self.experiments, num_lefts):
            exp.run_exp(num_left)
        for _ind, exp in enumerate(self.experiments):
            for tr, res in zip(exp.trials, exp.results):
                tr['parallel_round'] = res['parallel_round'] = _ind
                self.trials.append(tr)
                self.results.append(res)

    def run_exp(self, exp, N):
        exp.run(N)




##############
###BOOSTING###
##############

class BoostedExperiment(hyperopt.base.Experiment):
    """
    boosted experiment base
    expects init_experiment to be defined by mixin or subclass
    """
    def __init__(self, bandit_algo_class, bandit_class, boost_rounds, opt_runs,
                 **init_kwargs):
        self.bandit_algo_class = bandit_algo_class
        self.bandit_class = bandit_class
        self.boost_rounds = boost_rounds
        self.opt_runs = opt_runs
        self.experiments = []
        self.exp = None
        self.trials = []
        self.results = []
        self.boost_round = 0
        self.train_decisions = 0
        self.decisions = None
        self.selected_inds = []
        for k, v in init_kwargs.items():
            setattr(self, k, v)

    def run(self):
        while self.boost_round < self.boost_rounds:
            ###do number of trials always line with num results??
            ###how does this relate to errors, esp. in mongoexp?
            if not self.exp or len(self.exp.trials) >= self.opt_runs:
                self.exp = exp = self.init_experiment(self.decisions)
                self.experiments.append(self.exp)
            else:
                exp = self.exp
            num_done = len(exp.trials)
            num_left = self.opt_runs - num_done
            self.run_exp(exp, num_left)
            for tr, res in zip(exp.trials, exp.results):
                tr['boosting_round'] = res['boosting_round'] = self.boost_round
                self.trials.append(tr)
                self.results.append(res)
            loss = np.array([_r['loss'] for _r in exp.results])
            selected_ind = loss.argmin()
            self.selected_inds.append(selected_ind)
            self.decisions = exp.results[selected_ind]['decisions']
            self.boost_round += 1

    def run_exp(self, exp, N):
        exp.run(N)


class BoostedSerialExperiment(BoostedExperiment, SerialMixin):
    pass


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


