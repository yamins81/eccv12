"""
Usage:

e.g. to show all the errors in database with dbname="try2", type

$ fab list_errors:try2


"""
from fabric.api import run  # -- shutup pyflakes, we need this

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import copy
import cPickle
import hyperopt
from hyperopt.mongoexp import MongoTrials
from eccv12.eccv12 import main_lfw_driver

def lfw_suggest(dbname):
    """
    This class presents the entire LFW experiment as a BanditAlgo
    so that it can be started up with 
    
    hyperopt-mongo-search --exp_key='' eccv12.lfw.MultiBandit \
        eccv12.eccv12.WholeExperiment
    """
    trials = MongoTrials('mongo://localhost:44556/%s/jobs' % dbname)
    B = main_lfw_driver(trials)
    B.run()


def list_errors(dbname):
    trials = MongoTrials('mongo://localhost:44556/%s/jobs' % dbname,
                         refresh=False)
    for doc in trials.handle:
        if doc['state'] == hyperopt.JOB_STATE_ERROR:
            print doc['_id'], doc['tid'], doc['book_time'], doc['error']


def validate_from_tids(dbname):
    trials = MongoTrials('mongo://localhost:44556/%s/jobs' % dbname,
                         refresh=False)
    trials.refresh()
    tdict = dict([(t['tid'], t) for t in trials])
    print "TIDS", tdict.keys()

    for tid, t in tdict.items():
        assert t['misc']['tid'] == tid
        if 'from_tid' in t['misc']:
            if t['misc']['from_tid'] not in tdict:
                print 'WTF gave us', tid, t['misc']['from_tid']


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


def snapshot(dbname, tofile):
    print 'fetching trials'
    ofile = open(tofile, 'w')
    from_trials = MongoTrials('mongo://honeybadger.rowland.org:44556/%s/jobs' % dbname)
    to_trials = hyperopt.base.trials_from_docs(
            from_trials.trials)
    print 'saving to' , tofile
    cPickle.dump(to_trials, ofile, -1)


exp_keys = {
    'randomL': u'ek_randombandit:eccv12.lfw.MultiBandit_num_features:1280_bandit_algo:hyperopt.base.Random',
    'random': u'ek_randombandit:eccv12.lfw.MultiBandit_num_features:128_bandit_algo:hyperopt.base.Random',
    'tpe': u'ek_tpebandit:eccv12.lfw.MultiBandit_num_features:128_bandit_algo:hyperopt.tpe.TreeParzenEstimator',
    'tpe_asyncB': u"ek_tpebandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 200}",
    'random_asyncB':
    u"ek_randombandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.base.Random_meta_kwargs:{'round_len': 200}",
    }


def snapshot_history(tfile, key):
    trials = cPickle.load(open(tfile))
    docs = trials.trials
    kdocs = [d for d in docs if d['exp_key'] == exp_keys[key]]
    trials = hyperopt.base.trials_from_docs(kdocs)
    hyperopt.plotting.main_plot_history(trials)

def snapshot_tpe(tfile):
    import matplotlib.pyplot as plt
    trials = cPickle.load(open(tfile))
    m_docs = [d for d in trials.trials
            if d['exp_key'] == exp_keys['tpe'] and d['spec']['comparison'] == 'mult']
    s_docs = [d for d in trials.trials
            if d['exp_key'] == exp_keys['tpe'] and d['spec']['comparison'] != 'mult']

    plt.subplot(1, 2, 1)
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(m_docs),
            do_show=False)
    plt.ylim(.15, .4)
    plt.subplot(1, 2, 2)
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(s_docs),
            do_show=False)
    plt.ylim(.15, .4)
    plt.show()

def snapshot_histories(tfile):
    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.ylim(.15, .5)

    trials = cPickle.load(open(tfile))
    docs = trials.trials
    r_docs = [d for d in docs if d['exp_key'] == exp_keys['random']]
    t_docs = [d for d in docs if d['exp_key'] == exp_keys['tpe']]
    rB_docs = [d for d in docs if d['exp_key'] == exp_keys['random_asyncB']]
    tB_docs = [d for d in docs if d['exp_key'] == exp_keys['tpe_asyncB']]

    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(r_docs),
            do_show=False)

    plt.ylim(.15, .5)
    plt.subplot(2, 2, 2)

    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(t_docs),
            do_show=False)
    plt.ylim(.15, .5)
    plt.subplot(2, 2, 3)
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(rB_docs),
            do_show=False)
    plt.ylim(.15, .5)
    plt.subplot(2, 2, 4)
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(tB_docs),
            do_show=False)
    plt.ylim(.15, .5)
    plt.show()


if 0: # -- NOT SURE IF THIS IS CORRECT YET
  def main_fix_injected_tid_bug(dbname):
    trials = hyperopt.mongoexp.MongoTrials('mongo://localhost:44556/%s/jobs'
                                           % dbname)
    handle = trials.handle
    for doc in trials:
        misc = doc['misc']
        tid = misc['tid']
        fromtid = misc.get('from_tid', None)
        if fromtid is not None:
            idxs = misc['idxs']
            for nid, nidxs in idxs.items():
                assert len(nidxs) <= 1
                if nidxs:
                    assert nidxs[0] in (tid, fromtid)
                    if nidxs[0] == fromtid:
                        print 'fixing', tid, nid
                        nidxs[0] = tid
        #XXXX VERIFY THAT THIS ACTUALLY UPDATES JUST THE SUBDOCUMENT
        assert 0
        handle.coll.update(
            dict(_id=doc['_id']),
            {'misc': {'$set': doc['misc']}},
            safe=True,
            upsert=False,
            multi=False)
