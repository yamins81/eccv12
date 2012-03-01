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
from eccv12 import toyproblem
from eccv12 import utils
from eccv12.eccv12 import main_lfw_driver
from eccv12.eccv12 import BudgetExperiment
from eccv12.eccv12 import NestedExperiment
from eccv12.lfw import get_view2_features
from eccv12.lfw import train_view2
from eccv12.lfw import verification_pairs
from eccv12.lfw import MainBandit
from eccv12.lfw import MultiBandit
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
    trials = MongoTrials('mongo://localhost:%d/%s/jobs' % (port, dbname),
                        refresh=False)
    B = main_lfw_driver(trials)
    algo = B.interleaved_algo(priorities=priorities)
    exp = hyperopt.Experiment(B.trials, algo, max_queue_len=2)
    # -- the interleaving algo will break out of this
    exp.run(sys.maxint, block_until_done=True)


def lfw_suggest_parallel_tpe():
    dbname = 'feb29_par_tpe'
    port = 44556
    port = int(port)
    trials = MongoTrials('mongo://localhost:%d/%s/jobs' % (port, dbname),
                        refresh=False)
    #### trials.handle.delete_all()
    def add_exps(bandit_algo_class, exp_prefix):
        B = BudgetExperiment(ntrials=500, save=False, trials=trials,
                num_features=128 * 10,
                ensemble_sizes=[10],
                bandit_algo_class=bandit_algo_class,
                exp_prefix=exp_prefix,
                run_parallel=False)
        return B
    N = NestedExperiment(trials=trials, ntrials=500, save=False)
    priorities = {}
    for i in range(10):
        N.add_exp(add_exps(hyperopt.TreeParzenEstimator, 'ek_tpe%i' % i),
                'TPE%i' % i)
        priorities['TPE%i.fixed_features_10.AsyncBoostB' % i] = 1
    algo = N.interleaved_algo(priorities=priorities)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    # -- the interleaving algo will break out of this
    exp.run(sys.maxint, block_until_done=True)



def lfw_view2_randomL(host, dbname):
    trials = MongoTrials('mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    #B = main_lfw_driver(trials)
    #E = B.get_experiment(name=('random', 'foo'))
    mongo_trials = trials.view(exp_key=exp_keys['randomL'], refresh=True)

    docs = [d for d in mongo_trials.trials
            if d['result']['status'] == hyperopt.STATUS_OK]
    local_trials = hyperopt.trials_from_docs(docs, validate=False)
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


def lfw_view2_random_SimpleMixture(host, dbname, key, A):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            exp_key=exp_keys[key],
            refresh=True)
    bandit = MultiBandit()
    mix = SimpleMixture(trials, bandit)
    specs, weights, tids = mix.mix_models(int(A), ret_tids=True)
    assert len(specs) == len(tids)
    print 'SimpleMixture Members:'
    print '======================'
    for ii, tid in enumerate(tids):
        dd = [d for d in trials.trials if d['tid'] == tid][0]
        print ii, dd['_id'], dd['tid'], dd['result']['loss']
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

    #running with libsvm:
    train_view2(namebases=namebases, basedirs=basedirs,
                use_libsvm={'kernel':'precomputed'})
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


def lfw_view2_by_id(host, dbname, _id):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            #exp_key=exp_keys['random_asyncB'],
            refresh=True)
    doc = [t for t in trials.trials if str(t['_id']) == _id][0]
    namebase = '%s_%s' % (dbname, doc['_id'])

    get_view2_features(
            slm_desc=doc['spec']['model']['slm'],
            preproc=doc['spec']['model']['preproc'],
            comparison=doc['spec']['comparison'],
            namebase=namebase,
            basedir=os.getcwd(),
            )
    train_view2(namebases=[namebase], basedirs=[os.getcwd()],
                use_libsvm={'kernel':'precomputed'},
                )

def lfw_view1_by_id(host, dbname, _id):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            #exp_key=exp_keys['random_asyncB'],
            refresh=True)
    doc = [t for t in trials.trials if str(t['_id']) == _id][0]
    namebase = '%s_%s' % (dbname, doc['_id'])

    bandit = MainBandit()
    r = bandit.evaluate(doc['spec'], None)
    print 'loss', r['loss']
    print 'margin', r['margin']
    import pdb; pdb.set_trace()


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

    train_view2(namebases=namebases, basedirs=basedirs,
                use_libsvm={'kernel':'precomputed'})


def top_results(host, dbname, key, N):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            exp_key=exp_keys[key],
            refresh=False)
    # XXX: Does not use bandit.loss
    docs = list(trials.handle.jobs.find(
        {'exp_key': exp_keys[key], 'result.status': hyperopt.STATUS_OK},
        {'_id': 1, 'result.loss': 1}))
    losses_ids = [(d['result']['loss'], d['_id']) for d in docs]
    losses_ids.sort()
    for l, i in losses_ids[:int(N)]:
        print l, i


def lfw_view2_fold_kernels_by_id(host, dbname, _id):
    import bson
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    doc = trials.handle.jobs.find_one({'_id': bson.objectid.ObjectId(_id)})
    print 'TRIAL:', doc['_id']
    print 'SPEC :', doc['spec']
    print 'LOSS :', doc['result']['loss']
    namebase = '%s_%s' % (dbname, doc['_id'])
    image_features, pair_features = get_view2_features(
            slm_desc=doc['spec']['model']['slm'],
            preproc=doc['spec']['model']['preproc'],
            comparison=doc['spec']['comparison'],
            namebase=namebase,
            basedir=os.getcwd(),
            )

    split_data = [verification_pairs('fold_%d' % fold, subset=None)
            for fold in range(10)]

    for test_fold in range(10):
        print ('FOLD %i' % test_fold)
        test_X = pair_features[test_fold][:]
        test_y = split_data[test_fold][2]

        train_inds = [_ind for _ind in range(10) if _ind != test_fold]
        train_X = np.vstack([pair_features[ii][:] for ii in train_inds])
        print train_X.shape
        train_y = np.concatenate([split_data[_ind][2] for _ind in train_inds])

        train_Xyd_n, test_Xyd_n = toyproblem.normalize_Xcols(
            (train_X, train_y, None,),
            (test_X, test_y, None,))

        print ('Computing training kernel ...')
        (_Xtrain, _ytrain, _dtrain) = train_Xyd_n
        Ktrain = utils.linear_kernel(_Xtrain, _Xtrain, use_theano=True)
        print ('... computed training kernel of shape', Ktrain.shape)

        print ('Computing testtrain kernel ...')
        (_Xtest, _ytest, _dtest) = test_Xyd_n
        Ktest = utils.linear_kernel(_Xtest, _Xtrain, use_theano=True)
        print ('... computed testtrain kernel of shape', Ktest.shape)

        np.save(namebase + '_fold_%i_Ktrain.npy' % test_fold, Ktrain)
        np.save(namebase + '_fold_%i_Ktest.npy' % test_fold, Ktest)


def asdf(array_id = None):
    import cPickle
    # -- load index from PBS_ARRAYID
    if array_id is None:
        array_id = os.getenv('PBS_ARRAYID')
    filename = '/home/dyamins/eccv12/boosted_hyperopt_eccv12/Temp/simple_mix_id_nf.pkl'
    array_ids = cPickle.load(open(filename))
    print array_ids



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
            from_trials.trials,
            # -- effing slow, but required for un-pickling??
            validate=True)
    ofile = open(dbname+'.snapshot.pkl', 'w')
    cPickle.dump(to_trials, ofile, -1)


def history(host, dbname, key=None):
    exp_key = None if key is None else exp_keys[key]
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    # XXX: Does not use bandit.loss
    query = {'result.status': hyperopt.STATUS_OK}
    if exp_key is not None:
        query['exp_key'] = exp_key
    docs = list(trials.handle.jobs.find( query,
        {'tid': 1, 'result.loss': 1}))
    tids_losses = [(d['tid'], d['result']['loss']) for d in docs]
    tids_losses.sort()
    losses = [tl[1] for tl in tids_losses]
    print 'min', min(losses)
    import matplotlib.pyplot as plt
    plt.scatter(range(len(losses)), losses)
    plt.show()

def history_par_tpe(host, dbname):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    # XXX: Does not use bandit.loss
    query = {'result.status': hyperopt.STATUS_OK}
    docs = list(trials.handle.jobs.find( query,
        {'tid': 1, 'result.loss': 1, 'exp_key': 1}))
    tdocs = [(d['tid'], d) for d in docs]
    tdocs.sort()
    by_key = {}
    for d in docs:
        by_key.setdefault(d['exp_key'], []).append(d['result']['loss'])

    import matplotlib.pyplot as plt
    iii = 1
    for i, (k, losses) in enumerate(by_key.items()):
        if len(losses) < 50:
            continue
        print k, 'min', min(losses)
        plt.subplot(2, 5, iii)
        plt.scatter(range(len(losses)), losses)
        plt.ylim(.15, .55)
        iii += 1
    plt.show()


def hist(host, dbname, key=None):
    # XXX REFACTOR WITH ABOVE
    exp_key = None if key is None else exp_keys[key]
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    # XXX: Does not use bandit.loss
    query = {'result.status': hyperopt.STATUS_OK}
    if exp_key is not None:
        query['exp_key'] = exp_key
    docs = list(trials.handle.jobs.find( query,
        {'tid': 1, 'result.loss': 1}))
    tids_losses = [(d['tid'], d['result']['loss']) for d in docs]
    tids_losses.sort()
    losses = [tl[1] for tl in tids_losses]
    print 'min', min(losses)
    import matplotlib.pyplot as plt
    plt.hist(losses)
    plt.show()


def snapshot_history(dbname, key):
    trials = cPickle.load(open(dbname+'.snapshot.pkl'))
    docs = trials.trials
    _show_keys(docs)
    kdocs = [d for d in docs if d['exp_key'] == exp_keys[key]]
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(kdocs, validate=False))


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
            hyperopt.base.trials_from_docs(r_docs, validate=False),
            do_show=False)

    plt.ylim(.15, .5)
    plt.subplot(2, 2, 2)

    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(t_docs, validate=False),
            do_show=False)
    plt.ylim(.15, .5)
    plt.subplot(2, 2, 3)
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(rB_docs, validate=False),
            do_show=False)
    plt.ylim(.15, .5)
    plt.subplot(2, 2, 4)
    hyperopt.plotting.main_plot_history(
            hyperopt.base.trials_from_docs(tB_docs, validate=False),
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


def consolidate_random_jobs():
    final = hyperopt.mongoexp.MongoTrials(
            'mongo://localhost:44556/final_random/jobs',
            refresh=False)

    print 'FINAL RAW COUNT', final.handle.jobs.count()
    if final.handle.jobs.count() > 0:
        raise NotImplementedError()

    all_docs = dict()
    all_len = 0
    for othername in 'try2', 'feb28_1', 'march1_1':
        other = hyperopt.mongoexp.MongoTrials(
                'mongo://localhost:44556/%s/jobs' % othername,
                exp_key=exp_keys['random'],
                refresh=True)
        other_oks = [d for d in other
                     if d['result']['status'] == hyperopt.STATUS_OK]
        all_len += len(other_oks)
        print 'OTHER COUNT', len(other_oks)
        for d in other_oks:
            assert d["_id"] not in all_docs
            all_docs[d['_id']] = d
            d['misc']['consolidate_src'] = (othername, d['_id'])
            del d['_id'] # --need to remove it to re-insert

    assert len(all_docs) == all_len
    print 'Inserting %i jobs' % len(all_docs)
    for old_id, doc in all_docs.items():
        final.handle.jobs.insert(doc, safe=True)


