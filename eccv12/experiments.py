"""
put bandit algos here
"""
from math import log

import numpy as np

import hyperopt.base
import hyperopt.utils
import hyperopt.experiments 

class SerialBoostedBinaryExperiment(hyperopt.base.Experiment):
    """
    """

    def __init__(self, bandit_algo_class, bandit_class):
        self.bandit_algo_class = bandit_algo_class
        self.bandit_class = bandit_class
        self.trials = []
        self.results = []
        self.trial_rounds = []
        self.result_rounds = []
        self.boost_round = 0
        self.train_decisions = 0
        self.test_decisions = 0
        self.selected_inds = []

    def run(self, boost_rounds, opt_runs):
        algo_class = self.bandit_algo_class
        bandit_class = self.bandit_class
        for _round in xrange(boost_rounds):
            bandit = bandit_class(self.train_decisions,
                                  self.test_decisions,
                                  attach_weights=False)
            bandit_algo = bandit_algo_class(bandit)
            exp = hyperopt.experiment.SerialExperiment(bandit_algo)
            exp.run(opt_runs)
            self.trial_rounds.append(exp.trials)
            self.result_rounds.append(exp.results)
            for trial, result in zip(exp.trials, exp.results):
                trial['boosting_round'] = self.boost_round
                result['boosting_round'] = self.boost_round
                self.trials.append(trial)
                self.results.append(result)
            loss = np.array([_r['loss'] for r in exp.results])
            selected_ind = loss.argmin()
            self.selected_inds.append(selected_ind)
            self.train_decisions = np.array(exp.results[selected_ind]['train_decisions'])
            self.test_decisions = np.array(exp.results[selected_ind]['test_decisions'])
            self.boost_round += 1
            
