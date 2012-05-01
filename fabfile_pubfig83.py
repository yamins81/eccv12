import os
import time
import subprocess
import sys

import numpy as np
import pymongo as pm
import bson

import pyll
import hyperopt
from hyperopt import TreeParzenEstimator, Random
from hyperopt.mongoexp import MongoTrials

import eccv12.pubfig83 as pubfig83
from eccv12.experiments import InterleaveAlgo


def pubfig83_random_l3(host, port, dbname, N=3):
    bandit = pubfig83.pubfig83_bandit()
    cmd = ('bandit_json evaluate', 'eccv12.lfw2.lfw_bandit')

    trials = MongoTrials(
            'mongo://%s:%d/%s/jobs' % (host, int(port), dbname),
            refresh=True)
    algos = []
    keys = []
    for i in range(int(N)):
        algo = Random(
                bandit)
        algos.append(algo)
        keys.append('tpe_l3_exit3_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)


def suggest_from_name(dbname, host, port, bandit_algo_name, bandit_name, 
                     exp_key, N, bandit_args, bandit_kwargs, 
                     bandit_algo_args, bandit_algo_kwargs):
    port = int(port)
    trials = MongoTrials('mongo://%s:%d/%s/jobs' % (host, port, dbname),
                         exp_key=exp_key, refresh=False)
    bandit = json_call(bandit_name, bandit_args, bandit_kwargs)
    if bandit_args or bandit_kwargs:
        trials.attachments['bandit_data'] = cPickle.dumps((bandit_name,
                                                 bandit_args, bandit_kwargs))
        bandit_algo_kwargs['cmd'] = ('driver_attachment', 'bandit_data')
    else:
        bandit_algo_kwargs['cmd'] = ('bandit_json evaluate', bandit_name)
    algo = json_call(bandit_algo_name, (bandit,) + bandit_algo_args, bandit_algo_kwargs)
    exp = hyperopt.Experiment(trials, algo, max_queue_len=1)
    if N is not None:
        exp.run(N, block_until_done=True, async=False)
    else:
        return exp


def pubfig83_random_experiment(dbname, host, port):
    bandit_algo_name = 'hyperopt.Random'
    bandit_name = 'eccv12.pubfig83.pubfig83_bandit'
    exp_key = 'random'
    N = 
    bandit_args = ()
    bandit_kwargs = {'ntrain':20,
                     'nvalidate': 60,
                     'ntest': 20,
                     'nfolds': 3,
                     'use_decisions': False,
                     'use_raw_decisions': False,
                     'npatches': 50000,
                     'n_imgs_for_patches': 2000,
                     'max_n_features': 16000,
                     'max_layer_sizes': [64, 128],
                     'pipeline_timeout': 90.0,
                     'memmap_name': ''
                    }
    bandit_algo_args = ()
    bandit_algo_kwargs = {}
    suggest_from_name(dbname, host, port,
                     bandit_algo_name, bandit_name, 
                     exp_key, N, bandit_args, bandit_kwargs, 
                     bandit_algo_args, bandit_algo_kwargs)

    
    