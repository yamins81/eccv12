"""
experiment classes
"""
import os
import sys
import cPickle
import hashlib
from math import log

import numpy as np

import hyperopt.base
import hyperopt.utils as utils
from hyperopt.experiments import SerialExperiment
from hyperopt.mongoexp import MongoJobs, MongoExperiment, as_mongo_str

###XXX:  Code here requires that experiment result records pass validation from
###bandit.py 


class SimpleMixer(object):    
    def __init__(self, exp):
        self.exp = exp
        
    def mix_inds(self, A):
        exp = self.exp
        assert len(exp.results) >= A
        losses = np.array([x['loss'] for x in exp.results])
        s = losses.argsort()
        return s[:A], np.ones((A,)) / float(A)
    
    def mix_models(self, A):
        exp = self.exp
        inds, weights = self.mix_inds(A)
        return [exp.trials[ind] for ind in inds], weights


class AdaboostMixer(SimpleMixer):
    def fetch_labels(self, splitname):
        exp = self.exp
        labels = np.array([_r['labels'][splitname] for _r in exp.results])
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


class BoostedExperiment(hyperopt.base.Experiment):
    """
    boosted experiment base
    expects init_experiment to be defined by subclasses
    """
    def __init__(self, bandit_algo_class, bandit_class, boost_rounds, opt_runs):
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
        
    def run(self):
        algo_class = self.bandit_algo_class
        bandit_class = self.bandit_class
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


class BoostedSerialExperiment(BoostedExperiment):
    def init_experiment(self, decisions):
        bandit = self.bandit_class(decisions)
        bandit_algo = self.bandit_algo_class(bandit)
        return SerialExperiment(bandit_algo)
    

class BoostedMongoExperiment(BoostedExperiment):
    """
    boosted mongo experiment
    """
    def __init__(self,
                 bandit_algo_class,
                 bandit_class,
                 boost_rounds,
                 opt_runs,
                 workdir=None,
                 mongo_opts=None):
        BoostedExperiment.__init__(self,
                                   bandit_algo_class,
                                   bandit_class,
                                   boost_rounds,
                                   opt_runs)
        self.workdir = workdir
        self.mongo_opts = mongo_opts
        
    def init_experiment(self, decisions):
        return init_mongo_exp(self.bandit_algo_class,
                              self.bandit_class,
                              bandit_argv=(decisions,),
                              workdir=self.workdir,
                              mongo_opts=self.mongo_opts)


    def run_exp(self, exp, N):
        exp.run(N, block_until_done=True)


def init_mongo_exp(algo_name,
                   bandit_name,
                   bandit_argv=(),
                   bandit_kwargs=None,
                   algo_argv=(),
                   algo_kwargs=None,
                   workdir=None,
                   clear_existing=False,
                   force_lock=False,
                   mongo_opts='localhost/hyperopt'):

    ###XXX:  OK so this is ripped off from the code in hyperopt.mongoexp
    ###Perhaps it would be useful to modulized this functionality in a 
    ###new function in hyperopt.mongoexp, and then just call it here?

    if bandit_kwargs is None:
        bandit_kwargs = {}
    if algo_kwargs is None:
        algo_kwargs = {}
    if workdir is None:
        workdir = os.path.expanduser('~/.hyperopt.workdir')

    bandit = utils.json_call(bandit_name, bandit_argv, bandit_kwargs)
    algo = utils.json_call(algo_name, (bandit,) + algo_argv, algo_kwargs)

    bandit_argfile_text = cPickle.dumps((bandit_argv, bandit_kwargs))
    algo_argfile_text = cPickle.dumps((algo_argv, algo_kwargs))
 
    ###XXX: why do we use md5 and not sha1?  
    m = hashlib.md5()
    m.update(bandit_argfile_text)
    m.update(algo_argfile_text)
    exp_key = '%s/%s[arghash:%s]' % (bandit_name, algo_name, m.hexdigest())
    del m

    worker_cmd = ('driver_attachment', exp_key)

    mj = MongoJobs.new_from_connection_str(as_mongo_str(mongo_opts) + '/jobs')

    experiment = MongoExperiment(bandit_algo=algo,
                                 mongo_handle=mj,
                                 workdir=workdir,
                                 exp_key=exp_key,
                                 cmd=worker_cmd)

    experiment.ddoc_get()  
    
    # XXX: this is bad, better to check what bandit_tuple is already there
    #      and assert that it matches if something is already there
    experiment.ddoc_attach_bandit_tuple(bandit_name,
                                        bandit_argv,
                                        bandit_kwargs)

    if clear_existing:
        print >> sys.stdout, "Are you sure you want to delete",
        print >> sys.stdout, ("all %i jobs with exp_key: '%s' ?"
                % (mj.jobs.find({'exp_key':exp_key}).count(),
                    str(exp_key)))
        print >> sys.stdout, '(y/n)'
        y, n = 'y', 'n'
        if input() != 'y':
            print >> sys.stdout, "aborting"
            del self
            return 1
        experiment.ddoc_lock(force=force_lock)
        experiment.clear_from_db()
        experiment.ddoc_get()
        experiment.ddoc_attach_bandit_tuple(bandit_name,
                                            bandit_argv,
                                            bandit_kwargs)

    return experiment
