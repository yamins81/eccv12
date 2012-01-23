"""
put bandit algos here
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


class BoostedSerialExperiment(hyperopt.base.Experiment):
    """
    boosted serial experiment
    """

    def __init__(self, bandit_algo_class, bandit_class):
        self.bandit_algo_class = bandit_algo_class
        self.bandit_class = bandit_class
        
        self.experiments = []
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
            exp = SerialExperiment(bandit_algo)
            self.experiments.append(exp)
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


class BoostedMongoExperiment(hyperopt.base.Experiment):
    """
    boosted mongo experiment
    """

    def __init__(self,
                 bandit_algo_class,
                 bandit_class,
                 workdir=None,
                 mongo_opts=None):
        self.bandit_algo_class = bandit_algo_class
        self.bandit_class = bandit_class
        self.workdir = workdir
        self.mongo_opts = mongo_opts
        
        self.experiments = []
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
            exp = init_mongo_exp(bandit_algo_class,
                                 bandit_class,
                                 bandit_argv=(self.train_decisions,
                                              self.test_decisions),
                                 bandit_kwargs={'attach_weights': True},
                                 workdir=self.workdir,
                                 mongo_opts=mongo_opts)  
            self.experiments.append(exp)
            exp.run(opt_runs, block_until_done=True)
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

    bandit_argfile_text = cPickle.dumps(bandit_argv, bandit_kwargs)
    algo_argfile_text = cPickle.dumps(algo_argv, algo_kwargs)
 
    ###XXX: why do we use md5 and not sha1?  
    m = hashlib.md5()
    m.update(bandit_argfile_text)
    m.update(algo_argfile_text)
    exp_key = '%s/%s[arghash:%s]' % (bandit_name, algo_name, m.hexdigest())
    del m

    worker_cmd = ('driver_attachment', exp_key)

    mj = MongoJobs.new_from_connection_str(as_mongo_str(mongo_opts) + '/jobs')

    ###XXX: should these really by passed as kwargs?
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
