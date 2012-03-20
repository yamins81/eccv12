import time
import subprocess

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
    if int(N) > 1:
        qsub_cmd.extend(['-t', '1-%s' % N])
    qsub_cmd.append(qsub_script_name)
    subprocess.check_call(qsub_cmd)


def foo():
    print 'hello'
