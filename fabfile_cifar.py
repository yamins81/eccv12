import os
import time
import subprocess
import sys

import hyperopt
from hyperopt import TreeParzenEstimator
from hyperopt.mongoexp import MongoTrials

import eccv12.cifar10 as ec10
from eccv12.experiments import InterleaveAlgo


def cifar10_launch_workers(dbname, N, walltime='24:00:00'):
    text = """#!/bin/bash

    export SCIKIT_DATA=/scratch_local/skdata
    L=$SCIKIT_DATA/cifar10
    mkdir -p $L
    rsync -a ~/.skdata/cifar10/ $L/

    . VENV/eccv12/bin/activate
    VENV/eccv12/src/eccv12/hyperopt/bin/hyperopt-mongo-worker \
        --mongo=honeybadger:44556/%(dbname)s \
        --workdir=/scratch_local/eccv12.workdir \
        --reserve-timeout=180.0 \
        --max-consecutive-failures=3
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


def cifar10_suggest3small(dbname, port=44556, N=3):
    Bandit = ec10.Cifar10Bandit3Small
    cmd = ('bandit_json evaluate', 'eccv12.cifar10.Cifar10Bandit3Small')

    trials = MongoTrials(
            'mongo://localhost:%d/%s/jobs' % (port, dbname),
            refresh=True)
    algos = []
    keys = []
    for i in range(int(N)):
        algos.append(
            TreeParzenEstimator(
                Bandit(),
                cmd=cmd,
                ))
        keys.append('b3small_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)

def cifar10_suggest2small(dbname, port=44556, N=3):
    Bandit = ec10.Cifar10Bandit2Small
    cmd = ('bandit_json evaluate', 'eccv12.cifar10.Cifar10Bandit2Small')

    trials = MongoTrials(
            'mongo://localhost:%d/%s/jobs' % (port, dbname),
            refresh=True)
    algos = []
    keys = []
    for i in range(int(N)):
        algos.append(
            TreeParzenEstimator(
                Bandit(),
                cmd=cmd,
                ))
        keys.append('b2small_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)
