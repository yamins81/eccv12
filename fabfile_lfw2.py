import os
import time
import subprocess
import sys

import numpy as np
import pymongo as pm
import bson

import pyll
import hyperopt
from hyperopt import TreeParzenEstimator
from hyperopt.mongoexp import MongoTrials

import eccv12.lfw2 as lfw2
from eccv12.experiments import InterleaveAlgo

import fabfile_common


def lfw2_launch_workers(host, port, dbname, N, walltime='24:00:00'):
    rsync_data_local = """
    export SCIKIT_DATA=/scratch_local/skdata
    L=$SCIKIT_DATA/lfw/aligned
    mkdir -p $L
    rsync -a ~/.skdata/lfw/aligned/ $L/
    """
    return fabfile_common.launch_workers_helper(host, port, dbname, N,
                                                walltime, rsync_data_local)


def lfw2_tpe_l3(host, port, dbname, N=3):
    bandit = lfw2.lfw_bandit()
    cmd = ('bandit_json evaluate', 'eccv12.lfw2.lfw_bandit')

    trials = MongoTrials(
            'mongo://%s:%d/%s/jobs' % (host, int(port), dbname),
            refresh=True)
    algos = []
    keys = []
    for i in range(int(N)):
        algo = TreeParzenEstimator(
                bandit,
                cmd=cmd,
                n_startup_jobs=50,
                gamma=.25
                )
        algos.append(algo)
        keys.append('tpe_l3_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)


def lfw2_bandit_sample(rseed=1):
    bandit = lfw2.lfw_bandit(n_train=10, n_test=10, n_view2_per_split=10)
    #print 'EXPR'
    #print bandit.expr
    result = pyll.stochastic.sample(bandit.expr,
            np.random.RandomState(int(rseed)),
            print_trace=False)
    # attachments are big and binary,
    # don't print them explicitly
    attachments = result.pop('attachments', {})
    print 'RESULT', result
    if attachments:
        print "RESULT attachments keys"
        for k, v in attachments.items():
            print '  name=%s, size=%i' % (k, len(v))

if __name__ == '__main__':
    lfw2_bandit_sample(sys.argv[1])
