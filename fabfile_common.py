import copy
import cPickle
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import numpy as np
import pymongo as pm

import hyperopt
from hyperopt.mongoexp import MongoTrials



def show_history(host, port, dbname, key=None):
    """
    Show loss vs. time scatterplots for one experiment or all experiments in a
    database.
    """
    import matplotlib.pyplot as plt
    trials = MongoTrials(
            'mongo://%s:%s/%s/jobs' % (host, port, dbname),
            refresh=False)
    # XXX: Does not use bandit.loss
    query = {'result.status': hyperopt.STATUS_OK}
    docs = list(trials.handle.jobs.find( query,
        {'tid': 1, 'result.loss': 1, 'result.true_loss': 1, 'exp_key': 1}))
    #tdocs = [(d['tid'], d) for d in docs if d['exp_key'].startswith('tpe_l3')]
    tdocs = [(d['tid'], d) for d in docs]
    tdocs.sort()
    by_key = {}
    by_key_true_loss = {}
    for tid, d in tdocs:
        by_key.setdefault(d['exp_key'], []).append(d['result']['loss'])
        by_key_true_loss.setdefault(d['exp_key'], []).append(
                d['result'].get('true_loss'))

    if key is None:
        print len(by_key)
        kl_items = by_key.items()
        kl_items.sort()

        ROWS = int(np.ceil(len(kl_items) / 5.0))

        iii = 1
        for i, (k, losses) in enumerate(kl_items):
            minloss = min(losses)
            print k, 'min', minloss, [d['_id']
                    for t, d in tdocs
                    if d['result']['loss'] == minloss and d['exp_key'] == k]
            plt.subplot(ROWS, 5, iii)
            plt.title(k)
            plt.scatter(range(len(losses)), losses, c='b')
            xlist = [i for i, v in enumerate(by_key_true_loss[k]) if v != None]
            ylist = [v for i, v in enumerate(by_key_true_loss[k]) if v != None]
            if xlist:
                plt.scatter(xlist, ylist, c='g')
            plt.ylim(0, 1.02)
            iii += 1
    else:
        losses = by_key[key]
        plt.title(key)
        plt.scatter(range(len(losses)), losses, c='b')
        xlist = [i for i, v in enumerate(by_key_true_loss[key]) if v != None]
        ylist = [v for i, v in enumerate(by_key_true_loss[key]) if v != None]
        if xlist:
            plt.scatter(xlist, ylist, c='g')
        plt.ylim(0, 1.02)

    plt.show()


def show_vars(host, port, dbname, key, colorize=-1, columns=5):
    """
    Show loss vs. time scatterplots for one experiment or all experiments in a
    database.
    """
    conn = pm.Connection(host=host, port=int(port))
    J = conn[dbname]['jobs']
    K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
    for k in K:
        print k
    if key is None:
        raise NotImplementedError()
    else:
        docs = list(
                conn[dbname]['jobs'].find(
                    {'exp_key': key},
                    {
                        'tid': 1,
                        'state': 1,
                        'result.loss':1,
                        'result.status':1,
                        'spec':1,
                        'misc.cmd': 1,
                        'misc.tid':1,
                        'misc.idxs':1,
                        'misc.vals': 1,
                    }))
    doc0 = docs[0]
    cmd = doc0['misc']['cmd']
    print cmd
    if cmd[0] == 'bandit_json evaluate':
        bandit = hyperopt.utils.json_call(cmd[1])
    else:
        bandit = None
    print 'bandit', bandit

    trials = hyperopt.trials_from_docs(docs, validate=False)
    hyperopt.plotting.main_plot_vars(trials, bandit=bandit,
            colorize_best=int(colorize),
            columns=int(columns)
            )


def show_runtime(host, port, dbname, key):
    """
    Show runtime vs. trial_id
    """
    conn = pm.Connection(host=host, port=int(port))
    K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
    print 'Experiments in database', dbname
    for k in K:
        print '* ', k
    docs = list(
            conn[dbname]['jobs'].find(
                {'exp_key': key},
                {
                    'tid': 1,
                    'state': 1,
                    'book_time': 1,
                    'refresh_time': 1,
                    #'result.loss':1,
                    #'result.status':1,
                    #'spec':1,
                    #'misc.cmd': 1,
                    #'misc.tid':1,
                    #'misc.idxs':1,
                    #'misc.vals': 1,
                }))
    import matplotlib.pyplot as plt
    for state in hyperopt.JOB_STATES:
        x = [d['tid'] for d in docs if d['state'] == state]
        if state != hyperopt.JOB_STATE_NEW:
            y = [(d['refresh_time'] - d['book_time']).total_seconds() / 60.0
                 for d in docs if d['state'] == state]
        else:
            y = [0] * len(x)

        if x:
            plt.scatter(x, y,
                    c=['g', 'b', 'k', 'r'][state],
                    label=['NEW', 'RUNNING', "DONE", "ERROR"][state])
    plt.ylabel('runtime (minutes)')
    plt.xlabel('trial identifier')
    plt.legend(loc='upper left')
    plt.show()


def list_dbs(host, port, dbname=None):
    """
    List the databases and experiments being hosted by a mongo server
    """
    conn = pm.Connection(host=host, port=int(port))
    if dbname is None:
        dbnames = conn.database_names()
    else:
        dbnames = [dbname]

    for dbname in dbnames:
        J = conn[dbname]['jobs']
        K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
        print ''
        print 'Database:', dbname
        for k in K:
            print ' ', k


def transfer_trials(fromdb, todb):
    """
    Insert all of the documents in `fromdb` into `todb`.
    """
    from_trials = MongoTrials('mongo://localhost:44556/%s/jobs' % fromdb)
    to_trials = MongoTrials('mongo://localhost:44556/%s/jobs' % todb)
    from_docs = [copy.deepcopy(doc) for doc in from_trials]
    for doc in from_docs:
        del doc['_id']
    to_trials.insert_trial_docs(doc)


def list_errors(host, port, dbname, key=None, spec=0):
    conn = pm.Connection(host=host, port=int(port))
    if key is None:
        query = {}
    else:
        query = {'exp_key': key}
    query['state'] = hyperopt.JOB_STATE_ERROR
    retrieve = {'tid': 1, 'state': 1, 'result.status':1, 'misc.cmd': 1,
                'book_time': 1, 'error': 1, 'owner': 1}
    if int(spec):
        retrieve['spec'] = 1
    for doc in conn[dbname]['jobs'].find(query, retrieve):
        print doc['_id'], doc['tid'], doc['book_time'], doc['owner'], doc['error']
        if int(spec):
            print doc['spec']


def list_failures(host, port, dbname, key=None, spec=0):
    conn = pm.Connection(host=host, port=int(port))
    if key is None:
        query = {}
    else:
        query = {'exp_key': key}
    query['state'] = hyperopt.JOB_STATE_DONE
    query['result.status'] = hyperopt.STATUS_FAIL
    retrieve = {'tid': 1,
            'result.failure':1,
            'misc.cmd': 1, 'book_time': 1, 'owner': 1}
    if int(spec):
        retrieve['spec'] = 1
    for doc in conn[dbname]['jobs'].find(query, retrieve):
        print doc['_id'], doc['tid'], doc['book_time'], doc['owner'],
        print doc['result']['failure']
        if int(spec):
            print doc['spec']


def delete_trials(host, port, dbname, key=None):
    # TODO: replace this with an input() y/n type thing
    y, n = 'y', 'n'
    db = 'mongo://%s:%s/%s/jobs' % (host, port, dbname)
    if key is None:
        print 'Are you sure you want to delete ALL trials from %s? (y/n)' % db
        if input() != y:
            print 'Aborting'
            return
        mongo_trials = MongoTrials(db)
        mongo_trials.delete_all()
    else:
        mongo_trials = MongoTrials(db, exp_key=key)
        print 'Confirm: delete %i trials matching %s? (y/n)' % (
            len(mongo_trials), key)
        if input() != y:
            print 'Aborting'
            return
        mongo_trials.delete_all()


def snapshot(dbname, ofilename=None, plevel=-1):
    """
    Save the trials of a mongo database to a pickle file.
    """
    print 'fetching trials'
    from_trials = MongoTrials('mongo://honeybadger.rowland.org:44556/%s/jobs' % dbname)
    to_trials = hyperopt.base.trials_from_docs(
            from_trials.trials,
            # -- effing slow, but required for un-pickling??
            validate=True)
    if ofilename is None:
        ofilename = dbname+'.snapshot.pkl'
    print 'saving to', ofilename
    ofile = open(ofilename, 'w')
    cPickle.dump(to_trials, ofile, int(plevel))


def launch_workers_helper(host, port, dbname, N, walltime, rsync_data_local,
                          mem=None):
    text = """#!/bin/bash
    %(rsync_data_local)s
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
    if mem is not None:
        qsub_cmd.append('-lmem=%s' % mem)
    qsub_cmd.extend(
            ['-e', os.path.expanduser('~/.qsub/%s.err' % qsub_script_name)])
    qsub_cmd.extend(
            ['-o', os.path.expanduser('~/.qsub/%s.out' % qsub_script_name)])
    if int(N) > 1:
        qsub_cmd.extend(['-t', '1-%s' % N])
    qsub_cmd.append(qsub_script_name)
    print qsub_cmd
    subprocess.check_call(qsub_cmd)


