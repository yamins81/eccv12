import copy
import cPickle
import logging
import os
import sys

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
    trials = MongoTrials(
            'mongo://%s:%s/%s/jobs' % (host, port, dbname),
            refresh=False)
    # XXX: Does not use bandit.loss
    query = {'result.status': hyperopt.STATUS_OK}
    docs = list(trials.handle.jobs.find( query,
        {'tid': 1, 'result.loss': 1, 'exp_key': 1}))
    #tdocs = [(d['tid'], d) for d in docs if d['exp_key'].startswith('tpe_l3')]
    tdocs = [(d['tid'], d) for d in docs]
    tdocs.sort()
    by_key = {}
    for tid, d in tdocs:
        by_key.setdefault(d['exp_key'], []).append(d['result']['loss'])

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
            plt.scatter(range(len(losses)), losses)
            plt.ylim(0, 1)
            iii += 1
    else:
        losses = kl_items[key]
        plt.title(key)
        plt.scatter(range(len(losses)), losses)
        plt.ylim(0, 1)

    plt.show()


def show_vars(host, port, dbname, key):
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
    hyperopt.plotting.main_plot_vars(trials, bandit=bandit)


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
    trials = MongoTrials(
            'mongo://%s:%s/%s/jobs' % (host, port, dbname),
            refresh=False)
    conn = pm.Connection(host=host, port=int(port))
    jobs = conn[dbname]['jobs']
    if key is None:
        query = {}
    else:
        query = {'exp_key': exp_key}
    query['state'] = hyperopt.JOB_STATE_ERROR
    retrieve = {'tid': 1, 'state': 1, 'result.status':1, 'misc.cmd': 1,
            'spec': int(spec)}
    for doc in conn[dbname]['jobs'].find(query, retrieve):
        print doc['_id'], doc['tid'], doc['book_time'], doc['error']
        if int(spec):
            print doc['spec']


def delete_all(dbname):
    # TODO: replace this with an input() y/n type thing
    y, n = 'y', 'n'
    db = 'mongo://localhost:44556/%s/jobs' % dbname
    print 'Are you sure you want to delete ALL trials from %s?' % db
    if input() != y:
        return
    trials = MongoTrials(db)
    B = main_lfw_driver(trials)
    B.delete_all()


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


