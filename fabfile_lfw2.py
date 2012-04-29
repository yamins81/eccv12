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
                                                walltime, rsync_data_local,
                                               mem='6G')


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
                n_startup_jobs=20, # -- random draws fail a lot
                gamma=.25
                )
        algos.append(algo)
        keys.append('tpe_l3_exit3_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)


def lfw2_rerun(host, port, dbname, _id):
    conn = pm.Connection(host=host, port=int(port))
    J = conn[dbname]['jobs']
    doc = J.find_one({'_id': bson.objectid.ObjectId(_id)})
    spec = hyperopt.base.spec_from_misc(doc['misc'])
    print 'SPEC', spec
    cmd = doc['misc']['cmd']
    assert cmd[0] == 'bandit_json evaluate'
    bandit = hyperopt.utils.json_call(cmd[1])
    result = bandit.evaluate(config=spec, ctrl=None)
    attachments = result.pop('attachments', {})
    print 'RESULT', result
    if attachments:
        print "RESULT attachments keys"
        for k, v in attachments.items():
            print '  name=%s, size=%i' % (k, len(v))


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

def lfw2_fig_tpe_vs_random():
    host='honeybadger.rowland.org'
    port=44556
    dbname='lfw_apr_20'

    import matplotlib.pyplot as plt
    trials = MongoTrials(
            'mongo://%s:%s/%s/jobs' % (host, port, dbname),
            refresh=False)
    query = {'result.status': hyperopt.STATUS_OK}
    docs = list(trials.handle.jobs.find( query,
        {'tid': 1, 'result.loss': 1, 'result.true_loss': 1, 'exp_key': 1}))
    tdocs = [(d['tid'], d) for d in docs]
    tdocs.sort()
    by_key = {}
    by_key_true_loss = {}
    for tid, d in tdocs:
        by_key.setdefault(d['exp_key'], []).append(d['result']['loss'])
        by_key_true_loss.setdefault(d['exp_key'], []).append(
                d['result'].get('true_loss'))

    plt.title('Random vs. TPE: LFW-A')
    losses = by_key['random_l3_exit3_0']
    plt.scatter(range(len(losses)),
            np.minimum(losses, .5),
            c=(.5, .5, .5), label='Random')
    losses = by_key['tpe_l3_exit3_0']
    plt.scatter(range(len(losses)),
            np.minimum(losses, .5),
            c=(.1, .9, .1), label='TPE')
    plt.ylim(0, .52)
    plt.xlabel('n. trials')
    plt.ylabel('validation error')
    plt.legend(loc='upper right')
    if 0:
        plt.show()
    else:
        plt.savefig('lfw2_fig_tpe_vs_random.svg')

