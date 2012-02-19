"""
Experiment generator functions

Currently this stuff is untested mostly.

Entry point is to call something like "random_experiment()" or "tpe_experiment()"

"""
import cPickle
import sys
import matplotlib.pyplot
import copy

from hyperopt import Trials
from hyperopt.mongoexp import MongoTrials, as_mongo_str

try:
    from collections import OrderedDict
except ImportError:
    print "Python 2.7+ OrderedDict collection not available"
    try:
        from ordereddict import OrderedDict
        warn("Using backported OrderedDict implementation")
    except ImportError:
        raise ImportError("Backported OrderedDict implementation "
                          "not available. To install it: "
                          "'pip install -vUI ordereddict'")


import hyperopt
import hyperopt.plotting
from pyll import scope, clone, as_apply

import lfw 
import model_params

from .experiments import (SyncBoostingAlgo,
                                AsyncBoostingAlgo,
                                AdaboostMixture,
                                SimpleMixture,
                                ParallelAlgo)
from .model_params import main_param_func


def cname(cls):
    return cls.__class__.__module__ + '.' + cls.__class__.__name__
              
              
def num_features_lfw(nf):
    class LFWBandit(lfw.MainBandit):
        param_gen = main_param_func(nf)
    return LFWBandit()


class SearchExp(object):
    def __init__(self, num_features, bandit_func, bandit_algo_class, mongo_opts,
                 exp_prefix, trials=None):
        self.num_features = num_features
        self.bandit_algo_class = bandit_algo_class
        self.bandit = bandit_func(num_features)
        self.mongo_opts = mongo_opts 
        self.init_bandit_algo()
        self.exp_prefix = exp_prefix
        self.exp_key = self.get_exp_key()
        if trials is None:
            #trials = MongoTrials(as_mongo_str(self.mongo_opts) + '/jobs',
            #                          exp_key=self.exp_key)
            trials = Trials()
        self.trials = trials
                
    def init_bandit_algo(self):
        self.bandit_algo = self.bandit_algo_class(self.bandit)
        
    def get_info(self):
        return OrderedDict(
            num_features=self.num_features,
            bandit=cname(self.bandit),
            bandit_algo=cname(self.bandit_algo)
            )
    
    def get_exp_key(self):
        info = self.get_info()
        tag = '_'.join([k + ':' + str(v) for (k, v) in info.items()])
        return self.exp_prefix + tag 
        
    def get_filename(self, ntrials):
        info = self.get_info()
        tag = '_'.join([k + ':' + str(v) for (k, v) in info])
        return self.exp_prefix + tag + ('_%d' % ntrials) + '.pkl'
 
    def get_result(self):
        trial_info = self.get_info()
        trial_info['trials'] = self.trials
        return trial_info

    def run(self, n_trials):        
        exp = hyperopt.Experiment(
                self.trials,
                self.bandit_algo,
                async=False)
                #cmd=None) ##THIS NEEDS TO BE REPLACED WITH PROPER ATTACHMENT CMD)
                
        num_left = n_trials - len(self.trials.results)  ##count results differently/better
        exp.run(num_left, block_until_done=True)
        

    def save(self):
        result = self.get_result()
        info = self.get_info()
        ntrials = len(self.trials.results)
        cPickle.dump(result, open(self.get_filename(ntrials), 'w'))


class MixtureExp(SearchExp):
    def __init__(self, mixture_class, ensemble_size, *args, **kwargs):
        self.mixture_class = mixture_class
        self.ensemble_size = ensemble_size
        SearchExp.__init__(self, *args, **kwargs)
        self.mixture = self.mixture_class(self.trials, self.bandit)

    def get_info(self):
        info = SearchExp.get_info(self)
        info['mixture'] = cname(self.mixture_class(None, None))
        info['ensemble_size'] = self.ensemble_size
        return info
            
    def get_result(self):
        trial_info = SearchExp.get_result(self)
        inds, weights = self.mixture(self.ensemble_size)
        trial_info['mixture_inds'] = inds
        trial_info['mixture_weights'] = weights
        return trial_info
        
    
class MetaExp(SearchExp):
    def __init__(self, meta_algo_class, meta_kwargs, *args, **kwargs):
        self.meta_algo_class = meta_algo_class
        self.meta_kwargs = meta_kwargs
        SearchExp.__init__(self, *args, **kwargs)
    
    def init_bandit_algo(self):
        self.base_bandit_algo = self.bandit_algo_class(self.bandit)
        self.bandit_algo = self.meta_algo_class(self.base_bandit_algo,
                                                **self.meta_kwargs)
                                        
    def get_info(self):
        info = SearchExp.get_info(self)
        info['meta_algo'] = info.pop('bandit_algo')
        info['bandit_algo'] = cname(self.base_bandit_algo)
        info['meta_kwargs'] = self.meta_kwargs
        return info


def budget_experiments(num_features,
                       ntrials, 
                       ensemble_sizes,
                       bandit_func,
                       bandit_algo_class,
                       exp_prefix,
                       mongo_opts,
                       run_parallel=False):

    control_exp = SearchExp(num_features=num_features,
                  bandit_func=bandit_func,
                  bandit_algo_class=bandit_algo_class,
                  mongo_opts=mongo_opts,
                  exp_prefix=exp_prefix)
    control_exp.run(n_trials)
    control_exp.save()

    for es in ensemble_sizes:
        do_all_experiments(num_features=num_features / es, 
                           ntrials=ntrials * es, 
                           round_len=ntrials,
                           ensemble_size=es,
                           bandit_func=bandit_func,
                           bandit_algo_class=bandit_algo_class,
                           mongo_ops=mongo_opts,
                           exp_prefix=exp_prefix,
                           run_parallel=run_parallel)
        do_all_experiments(num_features=num_features, 
                           ntrials=ntrials, 
                           round_len=ntrials / es,
                           ensemble_size=es,
                           bandit_func=bandit_func,
                           bandit_algo_class=bandit_algo_class,
                           mongo_ops=mongo_opts,
                           exp_prefix=exp_prefix,
                           run_parallel=run_parallel)
                           
        
def do_all_experiments(num_features, ntrials, round_len, ensemble_size, 
                       bandit_func, bandit_algo_class, mongo_opts, exp_prefix,
                       run_parallel):

        basic_exp = SearchExp(num_features=num_features,
                      bandit_func=bandit_func,
                      bandit_algo_class=bandit_algo_class,
                      mongo_opts=mongo_opts,
                      exp_prefix=exp_prefix)
        basic_exp.run(ntrials)
        basic_exp.save()
        
        simple_mix = MixtureExp(mixture_class=SimpleMixture,
                            ensemble_size=ensemble_size,
                            num_features=num_features,
                            bandit_func=bandit_func,
                            bandit_algo_class=bandit_algo_class,
                            mongo_opts=mongo_opts,
                            exp_prefix=exp_prefix,
                            trials=basic_exp.trials)
        simple_mix.save()
        
        ada_mix = MixtureExp(mixture_class=AdaboostMixture,
                            ensemble_size=ensemble_size,
                            num_features=num_features,
                            bandit_func=bandit_func,
                            bandit_algo_class=bandit_algo_class,
                            mongo_opts=mongo_opts,
                            exp_prefix=exp_prefix,
                            trials=basic_exp.trials)
        ada_mix.save()
        
        syncboost_exp = MetaExp(meta_algo_class=SyncBoostingAlgo,
                                meta_kwargs={"round_len": round_len},
                                num_features=num_features,
                                bandit_func=bandit_func,
                                bandit_algo_class=bandit_algo_class,
                                mongo_opts=mongo_opts,
                                exp_prefix=exp_prefix)
        syncboost_exp.run(ntrials)
        syncboost_exp.save()

        asyncboost_exp = MetaExp(meta_algo_class=AsyncBoostingAlgo,
                                meta_kwargs={"round_len": round_len,
                                             "look_back": None},
                                num_features=num_features,
                                bandit_func=bandit_func,
                                bandit_algo_class=bandit_algo_class,
                                mongo_opts=mongo_opts,
                                exp_prefix=exp_prefix)
        asyncboost_exp.run(ntrials)
        asyncboost_exp.save()
        
        if run_parallel:
            parallel_exp = MetaExp(meta_algo_class=ParallelBoostingAlgo,
                                   meta_kwargs={"num_procs": ensemble_size},
                                   num_features=num_features,
                                   bandit_func=bandit_func,
                                   bandit_algo_class=bandit_algo_class,
                                   mongo_opts=mongo_opts,
                                   exp_prefix=exp_prefix)
            parallel_exp.run(ntrials)
            parallel_exp.save()
        

def random_experiment():    
    budget_experiments(num_features=128,
                       num_trials=100, 
                       ensemble_sizes=[2, 5],
                       bandit_func=num_features_lfw,
                       bandit_algo_class=hyperopt.Random,
                       exp_prefix='eccv12_experiments',
                       mongo_opts='localhost:27017/eccv12',
                       run_parallel=False)
                      
                      
def tpe_experiment():    
    budget_experiments(num_features=128,
                       num_trials=100, 
                       ensemble_sizes=[2, 5],
                       bandit_func=num_features_lfw,
                       bandit_algo_class=hyperopt.TreeParzenEstimator,
                       exp_prefix='eccv12_experiments',
                       mongo_opts='localhost:27017/eccv12',
                       run_parallel=True)
                       
#####I expect plotting code to go here as well