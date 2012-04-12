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

import eccv12.cifar10 as ec10
from eccv12.experiments import InterleaveAlgo


def cifar10_launch_workers(host, port, dbname, N, walltime='24:00:00'):
    text = """#!/bin/bash
    export SCIKIT_DATA=/scratch_local/skdata
    L=$SCIKIT_DATA/cifar10
    mkdir -p $L
    rsync -a ~/.skdata/cifar10/ $L/
    . VENV/eccv12/bin/activate
    VENV/eccv12/src/eccv12/hyperopt/bin/hyperopt-mongo-worker \
        --mongo=%(host)s:%(port)s/%(dbname)s \
        --workdir=/scratch_local/eccv12.workdir \
        --reserve-timeout=180.0 \
        --max-consecutive-failures=4
    """ % locals()

    qsub_script_name = '.worker.sh.%.3f' % time.time()

    script = open(qsub_script_name, 'w')
    script.write(text)
    script.close()

    subprocess.check_call(['chmod', '+x', qsub_script_name])
    qsub_cmd = ['qsub', '-lnodes=1:gpus=1', '-lwalltime=%s' % walltime]
    qsub_cmd.extend(
            ['-e', os.path.expanduser('~/.qsub/%s.err' % qsub_script_name)])
    qsub_cmd.extend(
            ['-o', os.path.expanduser('~/.qsub/%s.out' % qsub_script_name)])
    if int(N) > 1:
        qsub_cmd.extend(['-t', '1-%s' % N])
    qsub_cmd.append(qsub_script_name)
    print qsub_cmd
    subprocess.check_call(qsub_cmd)


def cifar10_suggest1(host, port, dbname, N=3):
    Bandit = ec10.Cifar10Bandit1
    cmd = ('bandit_json evaluate', 'eccv12.cifar10.Cifar10Bandit1')

    trials = MongoTrials(
            'mongo://%s:%d/%s/jobs' % (host, int(port), dbname),
            refresh=True)
    algos = []
    keys = []
    for i in range(int(N)):
        algo = TreeParzenEstimator(
                Bandit(),
                cmd=cmd,
                )
        algo.n_EI_candidates = 16
        algos.append(algo)
        keys.append('suggest1_sqrtgcap_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)


def cifar10_repeat_trial(host, port, dbname, _id):
    conn = pm.Connection(host=host, port=int(port))
    J = conn[dbname]['jobs']
    doc = J.find_one({'_id': bson.objectid.ObjectId(_id)})
    print 'SPEC', doc['spec']
    cmd = doc['misc']['cmd']
    assert cmd[0] == 'bandit_json evaluate'
    bandit = hyperopt.utils.json_call(cmd[1])
    result = bandit.evaluate(config=doc['spec'], ctrl=None)
    print 'ORIG'
    print '-' * 80
    orig = doc['result']
    del orig['xmean']
    del orig['xstd']
    for k, v in orig.items():
        print k, v
    print 'THIS'
    print '-' * 80
    del result['xmean']
    del result['xstd']
    for k, v in result.items():
        print k, v


def cifar10_run_large(host, port, dbname, _id):
    conn = pm.Connection(host=host, port=int(port))
    J = conn[dbname]['jobs']
    doc = J.find_one({'_id': bson.objectid.ObjectId(_id)})
    print 'SPEC', doc['spec']
    cmd = doc['misc']['cmd']
    assert cmd[0] == 'bandit_json evaluate'
    bandit = hyperopt.utils.json_call(cmd[1])
    if 'Bandit1' in cmd[1]:
        bandit = ec10.Cifar10Bandit1()
    elif 'Bandit2' in cmd[1]:
        bandit = ec10.Cifar10Bandit2()
    elif 'Bandit3' in cmd[1]:
        bandit = ec10.Cifar10Bandit3()
    else:
        raise Exception(cmd[1])
    result = bandit.evaluate(config=doc['spec'], ctrl=None)
    del result['xmean']
    del result['xstd']
    print result


def cifar10_bandit1_sample(n_train=100, n_valid=100, n_test=10):
    bandit = ec10.cifar10bandit(
            n_train=int(n_train),
            n_valid=int(n_valid),
            n_test=int(n_test))
    print 'EXPR'
    print bandit.expr
    result = pyll.stochastic.sample(bandit.expr,
            np.random.RandomState(34))
    print 'RESULT', result


def cifar10_bandit1_medium(start=0, stop=5):
    bandit = ec10.Cifar10Bandit1(n_train=10000, n_valid=1000, n_test=100)
    for i in range(int(start), int(stop)):
        config = pyll.stochastic.sample(bandit.template,
                np.random.RandomState(i))
        print 'CONFIG', config
        result = bandit.evaluate(config, ctrl=None)
        print 'RESULT', result


def cifar10_bandit3_large():
    bandit = ec10.Cifar10Bandit1(n_train=40000, n_valid=10000, n_test=10000,
            nfilt_ubounds=[64, 128, 256])
    print 'TEMPLATE', bandit.template
    config = pyll.stochastic.sample(bandit.template,
            np.random.RandomState(34))
    print 'CONFIG', config
    result = bandit.evaluate(config, ctrl=None)
    print result


def cifar10_compare_verrs(host, port, dbname, *keys):
    conn = pm.Connection(host=host, port=int(port))
    J = conn[dbname]['jobs']

    from skdata.cifar10 import CIFAR10
    import matplotlib.pyplot as plt
    from skdata.utils.glviewer import glumpy_viewer

    imgs, labels = CIFAR10().img_classification_task('uint8')
    imgs = imgs[40000: 50000]
    itarg = labels[40000: 50000]
    targ = np.asarray(list(''.join(str(i) for i in itarg)))
    print 'training   label counts:', np.bincount(labels[:40000])
    print 'validation label counts:', np.bincount(itarg)

    all_preds = []
    vpreds = {}
    for jj, key in enumerate(keys):
        docs = J.find({'exp_key': key, 'result.status': 'ok'})
        vpreds[key] = trial_preds = []
        for ii, doc in enumerate(docs):
            trial_preds.append(list(str(doc['result']['val_pred'])))
            #print ii, doc['result']['loss']
            assert np.allclose(
                    doc['result']['loss'],
                    np.mean(targ != trial_preds[-1]))
        all_preds.extend(trial_preds)
        trial_preds = np.asarray(trial_preds)
        np.save('val_preds_%s.npy' % key, trial_preds)
        errmat = (trial_preds != targ)
        if 0:
            svd = np.linalg.svd(errmat)
            eigvals = np.linalg.svd(errmat, compute_uv=False)
            plt.plot(eigvals)
            plt.show()
        avg_per_vimg = errmat.mean(axis=0)
        easy_to_hard = np.argsort(avg_per_vimg)

        if 0:
            glumpy_viewer(imgs[easy_to_hard],
                    [
                        avg_per_vimg[easy_to_hard],
                        itarg[easy_to_hard],
                        ])

            # first 221 images are birds!
            # image 221 is the most horse-looking image ever.
            #
            # last images are mainly unusual compositions - the subject is off
            # to the side, or presented at a funny angle.

        for k, size in enumerate((15, 100, 1000)):
            trial_preds_limited = trial_preds[:size]
            baseline = np.random.rand(*trial_preds_limited.shape) < .2
            baseline = np.sort(baseline.mean(axis=0))

            avg_per_vimg = (trial_preds_limited != targ).mean(axis=0)
            avg_per_vimg = np.sort(avg_per_vimg)
            #plt.subplot(1, 3, k + 1)
            plt.plot(avg_per_vimg, label='%s (%i)' % (key, size))
            #plt.plot(baseline, c='k')
        #plt.show()

    all_preds = np.asarray(all_preds)
    avg_per_vimg = (all_preds != targ).mean(axis=0)
    avg_per_vimg = np.sort(avg_per_vimg)
    plt.plot(avg_per_vimg, '--', label='all runs', )

    plt.xlabel('validation example')
    plt.ylabel('fraction of models that got it wrong')
    plt.legend(loc='upper left')
    plt.show()






##############################################################################

def coates_icml2011_patches(frac=1.0):
    config = {
            'nfbf0_nfilters': 1600.0,
            'nfb0_beta': 1070., # -- converted from var() + 10
            'nfb0_algo_i': 2,
            'nfb0_remove_mean': 1,
            'nfb0_size': 6,
            'nfb0_hard': 0,
            'nfb0_wp_rseed': 1,
            'nfb0_wp_gamma': 0.001, # -- should correspond to original .1
            'qp_alpha': 0.25 / np.sqrt(107),
            'qp_use_mid': 0,
            'qp_order': 1.0,
            'qp_order_logu': 1.0,
            'qp_grid_res': 2,
            'l2_reg': 0.0025,
            # -- XXX: DELETE ME
            'nfb0_af_rseed': 0,
            'nfb0_af_normalize': 0,
            'nfb0_pwf_gamma': 0.1,
            'classif_squash_lowvar': 0.0001, # --original had .01
            }
    bandit = ec10.Cifar10Bandit1(
            n_train=int(float(frac) * 49000),
            n_valid=int(float(frac) *  1000),
            n_test=int(float(frac) * 10000))
    result = bandit.evaluate(config, ctrl=None)
    print 'RESULT', result


def coates_algo_debug1():
    import eccv12.sc_vq_demo
    eccv12.sc_vq_demo.track_matlab()


def coates_classif():
    import eccv12.sc_vq_demo
    eccv12.sc_vq_demo.coates_classif()

