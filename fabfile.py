"""
Usage:

e.g. to show all the errors in database with dbname="try2", type

$ fab list_errors:try2


"""
from fabric.api import run  # -- shutup pyflakes, we need this

import copy
import cPickle
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import numpy as np

import hyperopt
from hyperopt.mongoexp import MongoTrials
from eccv12.eccv12 import main_lfw_driver
from eccv12.lfw import get_view2_features
from eccv12.lfw import train_view2
from eccv12.lfw import MultiBandit
from eccv12.lfw import get_model_shape
from eccv12.experiments import SimpleMixture
from eccv12.experiments import AdaboostMixture
from eccv12.experiments import BoostHelper



exp_keys = {
    #randomL is for LARGE-output random jobs
    'randomL': u'ek_randombandit:eccv12.lfw.MultiBandit_num_features:1280_bandit_algo:hyperopt.base.Random',
    'random': u'ek_randombandit:eccv12.lfw.MultiBandit_num_features:128_bandit_algo:hyperopt.base.Random',
    'tpeL': u'ek_tpebandit:eccv12.lfw.MultiBandit_num_features:1280_bandit_algo:hyperopt.tpe.TreeParzenEstimator',
    'tpe': u'ek_tpebandit:eccv12.lfw.MultiBandit_num_features:128_bandit_algo:hyperopt.tpe.TreeParzenEstimator',
    'tpe_asyncB': u"ek_tpebandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 200}",
    'random_asyncB': u"ek_randombandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.base.Random_meta_kwargs:{'round_len': 200}",
    'tpe_asyncB_no_inj': "ek_tpeuse_injected:False_bandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 200}",
    'tpe_no_inj': "ek_tpeuse_injected:False_bandit:eccv12.lfw.MultiBandit_num_features:128_bandit_algo:hyperopt.tpe.TreeParzenEstimator",
    }

def _show_keys(docs):
    keys = set([d['exp_key'] for d in docs])
    ikeys = dict([(v, k) for k, v in exp_keys.items()])
    print 'Short Key Names:'
    for k in keys:
        print ikeys.get(k, k)


def lfw_suggest(dbname, port=44556, **kwargs):
    """
    This class presents the entire LFW experiment as a BanditAlgo
    so that it can be started up with

    hyperopt-mongo-search --exp_key='' eccv12.lfw.MultiBandit \
        eccv12.eccv12.WholeExperiment
        
    fab lfw_suggest:test_hyperopt,port=22334,random=.5,TPE=.5
    """
    port = int(port)
    if len(kwargs) > 0:
        priorities = {}
        for k in kwargs:
            priorities[k] = float(kwargs[k])
    else:
        priorities = None
    trials = MongoTrials('mongo://localhost:%d/%s/jobs' % (port, dbname))
    B = main_lfw_driver(trials)
    B.run(priorities=priorities)


def lfw_view2_randomL(host, dbname):
    trials = MongoTrials('mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    #B = main_lfw_driver(trials)
    #E = B.get_experiment(name=('random', 'foo'))
    mongo_trials = trials.view(exp_key=exp_keys['randomL'], refresh=True)

    docs = [d for d in mongo_trials.trials
            if d['result']['status'] == hyperopt.STATUS_OK]
    local_trials = hyperopt.trials_from_docs(docs)
    losses = local_trials.losses()
    best_doc = docs[np.argmin(losses)]

    #XXX: Potentially affected by the tid/injected jobs bug,
    #     but unlikely. Rerun just in case once dual svm solver is in.
    print best_doc['spec']
    namebase = '%s_randomL_%s' % (dbname, best_doc['tid'])

    get_view2_features(
            slm_desc=best_doc['spec']['model']['slm'],
            preproc=best_doc['spec']['model']['preproc'],
            comparison=best_doc['spec']['comparison'],
            namebase=namebase,
            basedir=os.getcwd(),
            )

    namebases = [namebase]
    basedirs = [os.getcwd()] * len(namebases)

    #train_view2(namebases=namebases, basedirs=basedirs)
    # running on the try2 database
    # finds id 1674
    #train err mean 0.0840740740741
    #test err mean 0.199666666667

    #running with libsvm:
    train_view2(namebases=namebases, basedirs=basedirs,
                use_libsvm={'kernel':'precomputed'})
    #train err mean 0.0
    #test err mean 0.183166666667


def lfw_view2_random_SimpleMixture(host, dbname, A):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            exp_key=exp_keys['random'],
            refresh=True)
    bandit = MultiBandit()
    mix = SimpleMixture(trials, bandit)
    specs, weights, tids = mix.mix_models(int(A), ret_tids=True)
    assert len(specs) == len(tids)
    namebases = []
    for spec, tid in zip(specs, tids):
        # -- allow this feature cache to be
        #    reused by AdaboostMixture and
        #    SimpleMixtures of different 
        #    sizes

        #XXX: Potentially affected by the tid/injected jobs bug,
        #     but unlikely. Rerun just in case once dual svm solver is in.
        namebase = '%s_%s' % (dbname, tid)
        namebases.append(namebase)

        get_view2_features(
                slm_desc=spec['model']['slm'],
                preproc=spec['model']['preproc'],
                comparison=spec['comparison'],
                namebase=namebase,
                basedir=os.getcwd(),
                )

    basedirs = [os.getcwd()] * len(namebases)

    train_view2(namebases=namebases, basedirs=basedirs)
    # running on the try2 database
    # finds id 1674
    # train err mean 0.049037037037
    # test err mean 0.1565


def lfw_view2_random_AdaboostMixture(host, dbname, A):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            exp_key=exp_keys['random'],
            refresh=True)
    bandit = MultiBandit()
    mix = AdaboostMixture(trials, bandit, test_mask=True)
    # XXX: Should the weights be used? I don't think so, we're basically
    #      doing LPBoost at this point
    specs, weights, tids = mix.mix_models(int(A), ret_tids=True)
    assert len(specs) == len(tids)
    namebases = []
    for spec, tid in zip(specs, tids):
        # -- allow this feature cache to be
        #    reused by AdaboostMixture and
        #    SimpleMixtures of different 
        #    sizes

        #XXX: Potentially affected by the tid/injected jobs bug,
        #     but unlikely. Rerun just in case once dual svm solver is in.
        namebase = '%s_%s' % (dbname, tid)
        namebases.append(namebase)

        get_view2_features(
                slm_desc=spec['model']['slm'],
                preproc=spec['model']['preproc'],
                comparison=spec['comparison'],
                namebase=namebase,
                basedir=os.getcwd(),
                )

    basedirs = [os.getcwd()] * len(namebases)

    train_view2(namebases=namebases, basedirs=basedirs)


def lfw_view2_random_AsyncB(host, dbname, A):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            exp_key=exp_keys['random_asyncB'],
            refresh=True)
    helper = BoostHelper(trials.trials)
    # XXX: Should the weights be used? I don't think so, we're basically
    #      doing LPBoost at this point
    members = helper.ensemble_members(MultiBandit())[:int(A)]
    for ii, dd in enumerate(members):
        ccc = helper.continues(dd)
        print ii, dd['_id'], dd['tid'], dd['result']['loss'],
        print (ccc['_id'] if ccc else None)
    namebases = []
    for doc in members:
        namebase = '%s_%s' % (dbname, doc['_id'])
        namebases.append(namebase)

        get_view2_features(
                slm_desc=doc['spec']['model']['slm'],
                preproc=doc['spec']['model']['preproc'],
                comparison=doc['spec']['comparison'],
                namebase=namebase,
                basedir=os.getcwd(),
                )

    basedirs = [os.getcwd()] * len(namebases)

    train_view2(namebases=namebases, basedirs=basedirs)


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


def snapshot(dbname):
    print 'fetching trials'
    from_trials = MongoTrials('mongo://honeybadger.rowland.org:44556/%s/jobs' % dbname)
    to_trials = hyperopt.base.trials_from_docs(
            from_trials.trials)
    ofile = open(dbname+'.snapshot.pkl', 'w')
    cPickle.dump(to_trials, ofile, -1)


def snapshot_history(dbname, key):
    trials = cPickle.load(open(dbname+'.snapshot.pkl'))
    docs = trials.trials
    _show_keys(docs)
    kdocs = [d for d in docs if d['exp_key'] == exp_keys[key]]
    hyperopt.plotting.main_plot_history(hyperopt.base.trials_from_docs(kdocs))


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


import pymongo as pm

def insert_consolidated_feature_shapes(trials=None):
    conn = pm.Connection('honeybadger.rowland.org',44556)
    Jobs = conn['final_random']['jobs']
    if trials is None:
        triallist = enumerate(Jobs.find(fields=['spec','result.num_features'],
                                        timeout=False).sort('_id'))
    else:
        triallist = enumerate(trials)
    for (_ind, j) in triallist:
        print (_ind, j['_id'])
        if 'num_features' not in j.get('result', {}):
            shp = list(get_model_shape(j['spec']['model']))
            num_features = int(np.prod(shp))
            Jobs.update({'_id': j['_id']}, 
                        {'$set':{'result.shape': shp,
                                 'result.num_features': num_features}},
                        upsert=False, safe=True, multi=False)
                        
                        
def lfw_view2_final_get_mix(host='honeybadger.rowland.org',
                            dbname='final_random',
                            A=100):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            exp_key=exp_keys['random'],
            refresh=True)
    return trials
    bandit = MultiBandit()
    simple_mix = SimpleMixture(trials, bandit)
    simple_mix_trials =  simple_mix.mix_trials(int(A))
    ada_mix = AdaboostMixture(trials, bandit)
    ada_mix_trials =  ada_mix.mix_trials(int(A))    
    
    return simple_mix_trials, ada_mix_trials
