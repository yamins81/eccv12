"""
put bandit algos here
"""
from math import log

import hyperopt.base
import hyperopt.utils
import hyperopt.experiments 

class SerialBoostedBinaryExperiment(hyperopt.base.Experiment):
    """
    """

    def __init__(self, bandit_algo):
        super(SerialBoostedBinaryExperiment, self).__init__(bandit_algo)
        self.weights = None
        self.labels = None
        self.selected_inds = []
        self.eps = []
        self.predictions = []
        self.alphas = []
        self.trial_rounds = []
        self.result_rounds = []
        self.boost_round = 0

    def run(self, boost_rounds, opt_runs):
        algo = self.bandit_algo
        bandit = algo.bandit
        for _round in xrange(boost_rounds):
            bandit.training_weights = self.weights
            exp = hyperopt.experiments.SerialExperiment(bandit_algo)
            exp.run(opt_runs)
            self.trial_rounds.append(exp.trials)
            self.result_rounds.append(exp.results)
            for trial, result in zip(exp.trials, exp.results):
                trial['boosting_round'] = self.boost_round
                result['boosting_round'] = self.boost_round
                self.trials.append(trial)
                self.results.append(result)
            if self.labels is None:
                self.labels = np.array(exp.results[0]['training_labels'])
            else:
                assert self.labels == np.array(exp.results[0]['training_labels']), "Labels can't change!"  
            L = len(self.labels)
            errors = np.array([_r['training_errors'] for _r in exp.results])
            if self.weights is None:
                self.weights = (1./L) * np.ones((L,))
            ep_array = np.dot(errors, self.weights)
            ep_diff_array = np.abs(0.5 - ep_array)
            model_ind = ep_diff_array.argmax()
            self.selected_inds.append(model_ind)
            ep = ep_array[model_ind]
            self.eps.append(ep)
            alpha = 0.5 * log((1 - ep) / ep)
            self.alphas.append(alpha)
            prediction = np.array(exp.trials[model_ind]['training_prediction'])
            self.predictions.append(prediction)
            W = self.weights * np.exp(-alpha * self.labels * prediction)
            self.weights  = W / W.sum()
            self.boost_round += 1
            
    def get_prediction(self, to_round=None):
        alphas = np.array(self.alphas)
        predictions = np.array(self.predictions)
        if to_round is None:
            to_round = len(alphas)
        return np.sign((alphas[: to_round] * predictions[: to_round].T ).sum(1))

        