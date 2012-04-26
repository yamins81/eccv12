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
    bandit = lfw2.lfw_bandit(memmap_name='lfw2_rerun_%s' % _id)
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


def lfw2_rerun_view2_from_memmap(host, port, dbname, _id):
    conn = pm.Connection(host=host, port=int(port))
    J = conn[dbname]['jobs']
    doc = J.find_one({'_id': bson.objectid.ObjectId(_id)})
    spec = hyperopt.base.spec_from_misc(doc['misc'])
    print 'SPEC', spec
    pipeline = {}
    for key in spec:
        if key.endswith('remove_std0'):
            print key
            pipeline['remove_std0'] = spec[key]
        if key.endswith('varthresh'):
            print key
            pipeline['varthresh'] = spec[key]
        if key.endswith('l2_reg'):
            print key
            pipeline['l2_reg'] = spec[key]

    from skdata import larray
    image_features = larray.cache_memmap(None, name='lfw2_rerun_%s' % _id)

    view2_xy = {}
    for fold in range(10):
        fold_vp = lfw2.lfw_verification_pairs(split='fold_%i' % fold,
            subset=None,
            interleaved=True)
        for comparison in ['mult', 'sqdiff', 'sqrtabsdiff', 'absdiff']:
        #for comparison in ['sqrtabsdiff']:
        #for comparison in ['mult']:
            # if we could get view1 extracted in time, then assume it's fast
            # enough here. Don't want view2 results cancelled by a delay that
            # puts one feature calculation over the limit.
            pd = lfw2.cache_feature_pairs(fold_vp, image_features,
                comparison_name=comparison, pair_timelimit=None)
            view2_xy.setdefault(fold, {}).setdefault(comparison, pd)

    # N.B. Not patching in any decisions to the solver
    result = {'attachments': {}}
    result = lfw2.lfw_view2_results(view2_xy, pipeline, result,
            trace_normalize=True,
            solver=(lfw2.SVM_SOLVER[0],
                dict(lfw2.SVM_SOLVER[1],
                    n_runs=1,
                    #bfgs_factr=1e3, # -- tried once, no effect
                    )))


