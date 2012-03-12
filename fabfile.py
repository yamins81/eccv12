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
import matplotlib
#matplotlib.use('gtkAgg')
import matplotlib.pyplot as plt

import pyll

import hyperopt
from hyperopt.mongoexp import MongoTrials
from eccv12 import toyproblem
from eccv12 import utils
from eccv12 import model_params
from eccv12.eccv12 import main_lfw_driver
from eccv12.eccv12 import BudgetExperiment
from eccv12.eccv12 import NestedExperiment
from eccv12.lfw import get_view2_features
from eccv12.lfw import train_view2
from eccv12.lfw import verification_pairs
from eccv12.lfw import MainBandit
from eccv12.lfw import MultiBandit
from eccv12.lfw import get_model_shape
from eccv12.lfw import FG11Bandit
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
    'tpe0':"ek_tpe0bandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 500}",
    'tpe1':"ek_tpe1bandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 500}",
    'tpe2':"ek_tpe2bandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 500}",
    'tpe3':"ek_tpe3bandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 500}",
    'tpe4':"ek_tpe4bandit:eccv12.lfw.MultiBandit_num_features:128_meta_algo:eccv12.experiments.AsyncBoostingAlgoB_bandit_algo:hyperopt.tpe.TreeParzenEstimator_meta_kwargs:{'round_len': 500}",
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


def lfw_suggest_parallel_tpe(dbname, ENUM, n_exps):
    port = 44556
    port = int(port)
    trials = MongoTrials('mongo://localhost:%d/%s/jobs' % (port, dbname),
                        refresh=False)
    #### trials.handle.delete_all()
    ntrials = 500
    def add_exps(bandit_algo_class, exp_prefix):
        B = BudgetExperiment(ntrials=ntrials, save=False, trials=trials,
                num_features=128 * 10,
                ensemble_sizes=[10],
                bandit_algo_class=bandit_algo_class,
                exp_prefix=exp_prefix,
                run_parallel=False)
        return B
    N = NestedExperiment(trials=trials, ntrials=ntrials, save=False)
    priorities = {}
    for i in range(int(ENUM),int(ENUM)+int(n_exps)):
        N.add_exp(add_exps(hyperopt.TreeParzenEstimator, 'ek_tpe%i' % i),
                'TPE%i' % i)
        priorities['TPE%i.fixed_features_10.AsyncBoostB' % i] = 1
    algo = N.interleaved_algo(priorities=priorities)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    # -- the interleaving algo will break out of this
    exp.run(sys.maxint, block_until_done=True)


def lfw_suggest_l3(dbname, port=44556):
    from eccv12.lfw import MultiBanditL3
    from eccv12.experiments import InterleaveAlgo
    from hyperopt import TreeParzenEstimator
    port = int(port)
    trials = MongoTrials('mongo://localhost:%d/%s/jobs' % (port, dbname),
                        refresh=False)
    algos = []
    keys = []
    for i in range(5):
        algos.append(
            TreeParzenEstimator(
                MultiBanditL3(),
                cmd=('bandit_json evaluate', 'eccv12.lfw.MultiBanditL3'),
                gamma=.20,
                n_EI_candidates=64,
                n_startup_jobs=2,
                linear_forgetting=50,
                ))
        keys.append('tpe_l3_lf50_ns2_g20_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
    exp.run(sys.maxint, block_until_done=True)


def lfw_suggest_tpe_fg11(dbname, port=44556):
    from eccv12.experiments import InterleaveAlgo
    from hyperopt import TreeParzenEstimator
    port = int(port)
    trials = MongoTrials('mongo://localhost:%d/%s/jobs' % (port, dbname),
                        refresh=False)
    algos = []
    keys = []
    for i in range(3):
        algos.append(TreeParzenEstimator(
            FG11Bandit(),
            cmd=('bandit_json evaluate', 'eccv12.lfw.FG11Bandit')))
        keys.append('tpe_fg11_%i' % i)
    algo = InterleaveAlgo(algos, keys)
    exp = hyperopt.Experiment(trials, algo, poll_interval_secs=.1)
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


def top_results(host, dbname, N, port=44556):
    port = int(port)
    trials = MongoTrials(
            'mongo://%s:%s/%s/jobs' % (host, port, dbname),
            refresh=False)
    conn = pm.Connection(host=host, port=port)
    K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
    for exp_key in K:
        # N.B. Does not use bandit.loss, because that requires loading the
        # entire document.
        docs = list(trials.handle.jobs.find(
            {'exp_key': exp_key, 'result.status': hyperopt.STATUS_OK},
            {'_id': 1, 'result.loss': 1, 'spec': 1}))
        losses_ids = [(d['result']['loss'], d) for d in docs]
        losses_ids.sort()
        for l, d in losses_ids[:int(N)]:
            print l, d['_id'],
            print 'comp:', d['spec']['comparison'],
            print 'preproc', d['spec']['model']['preproc'],
            print 'layers', len(d['spec']['model']['slm'])


def Ktrain_name(dbname, _id, fold):
    namebase = '%s_%s' % (dbname, _id)
    return namebase + '_fold_%i_Ktrain.npy' % fold


def Ktest_name(dbname, _id, fold):
    namebase = '%s_%s' % (dbname, _id)
    return namebase + '_fold_%i_Ktest.npy' % fold


def lfw_view2_fold_kernels_fg11():
    L = FG11Bandit()
    config = pyll.stochastic.sample(L.template, np.random.RandomState(0))
    config['decisions'] = None
    config['slm'] = pyll.stochastic.sample(pyll.as_apply(model_params.fg11_top),
            np.random.RandomState(0))

    lfw_view2_fold_kernels_by_spec(config, 'fakedbFG11', 'best0comp4',
            comparison=['mult', 'sqrtabsdiff', 'absdiff', 'sqdiff'])


def lfw_view2_fold_kernels_by_spec(doc_spec_model, dbname, _id,
        comparison=['mult', 'sqrtabsdiff']):
    namebase = '%s_%s' % (dbname, _id)
    image_features, pair_features_by_comp = get_view2_features(
            slm_desc=doc_spec_model['slm'],
            preproc=doc_spec_model['preproc'],
            namebase=namebase,
            comparison=comparison,
            basedir=os.getcwd(),
            )

    split_data = [verification_pairs('fold_%d' % fold, subset=None)
            for fold in range(10)]

    for test_fold in range(10):
        try:
            open(Ktest_name(dbname, _id, test_fold)).close()
            continue
        except IOError:
            pass
        print ('FOLD %i' % test_fold)
        blend_train = blend_test = None
        for comp, pf in pair_features_by_comp.items():
            test_X = pf[test_fold][:]
            test_y = split_data[test_fold][2]

            train_inds = [ii for ii in range(10) if ii != test_fold]
            train_X = np.vstack([pf[ii][:] for ii in train_inds])
            train_y = np.concatenate([split_data[_ind][2] for _ind in train_inds])

            train_Xyd_n, test_Xyd_n = toyproblem.normalize_Xcols(
                (train_X, train_y, None,),
                (test_X, test_y, None,))

            print ('Computing training kernel. n_features=%i' %
                    train_X.shape[1])
            (_Xtrain, _ytrain, _dtrain) = train_Xyd_n
            Ktrain = utils.linear_kernel(_Xtrain, _Xtrain, use_theano=True)

            print ('Computing testtrain kernel ...')
            (_Xtest, _ytest, _dtest) = test_Xyd_n
            Ktest = utils.linear_kernel(_Xtest, _Xtrain, use_theano=True)

            if blend_train is None:
                blend_train = Ktrain
                blend_test = Ktest
            else:
                blend_train += Ktrain
                blend_test += Ktest

        np.save(Ktrain_name(dbname, _id, test_fold), blend_train)
        np.save(Ktest_name(dbname, _id, test_fold), blend_test)


def lfw_view2_fold_kernels_by_id(host, dbname, _id, port=44556):
    import bson
    trials = MongoTrials(
            'mongo://%s:%s/%s/jobs' % (host, port, dbname),
            refresh=False)
    doc = trials.handle.jobs.find_one({'_id': bson.objectid.ObjectId(_id)})
    print 'TRIAL:', doc['_id']
    print 'SPEC :', doc['spec']
    print 'LOSS :', doc['result']['loss']
    return lfw_view2_fold_kernels_by_spec(
            doc['spec']['model'], dbname, doc['_id'])


def blend_N(N, dbname, out_template, dryrun, *_ids):
    return blend_top_N(int(N), dbname, _ids, out_template, int(dryrun))

def run_each(dbname, out_template, dryrun, C, *_ids):
    for _id in _ids:
        blend_top_N(1, dbname, [_id], out_template, int(dryrun), C=float(C))


def blend_top_N(N, dbname, _ids, out_template, dryrun=False, C=100):
    import  eccv12.classifier
    # allocate the gram matrix for each fold
    # This will be incremented as we loop over the top models
    Ktrains = [np.zeros((5400, 5400), dtype='float')
               for i in range(10)]
    Ktests  = [np.zeros((600, 5400), dtype='float32')
               for i in range(10)]

    split_data = [verification_pairs('fold_%d' % fold, subset=None)
            for fold in range(10)]

    test_errs = {}

    # loop over top N `_id`s incrementing train_X by Ktrain
    for member_position in range(N):
        _id = _ids[member_position]
        print 'Working through model', _id

        for test_fold in range(10):

            if dryrun:
                try:
                    open(Ktrain_name(dbname, _id, test_fold)).close()
                    open(Ktest_name(dbname, _id, test_fold)).close()
                except IOError:
                    print '---> Missing', member_position, _id
                continue

            Ktrain_n_fold = np.load(Ktrain_name(dbname, _id, test_fold))
            Ktest_n_fold = np.load(Ktest_name(dbname, _id, test_fold))
            Ktrains[test_fold] += Ktrain_n_fold
            Ktests[test_fold] += Ktest_n_fold

            train_y = np.concatenate([split_data[_ind][2]
                    for _ind in range(10) if _ind != test_fold])
            test_y = split_data[test_fold][2]

            svm, _ = eccv12.classifier.train_scikits(
                    (Ktrains[test_fold], train_y, None),
                    labelset=[-1, 1],
                    model_type='svm.SVC',
                    model_kwargs={'kernel': 'precomputed', 'C': C},
                    normalization=False
                    )
            test_predictions = svm.predict(Ktests[test_fold])
            test_err = (test_predictions != test_y).mean()
            print member_position, test_fold, test_err
            test_errs[(member_position, test_fold)] = test_err

        if not dryrun:
            print 'Mean', member_position, np.mean(
                [test_errs[(member_position, ii)] for ii in range(10)])

            if out_template:
                cPickle.dump(test_errs,
                         open(out_template % member_position,'w'))

simplemix_top_N_filename = 'simple_mix_id_nf.pkl'

def simplemix_top_N(N, trace_normalize='False'):
    dbname = 'final_random'
    _ids = cPickle.load(open(simplemix_top_N_filename))['_id']
    return blend_top_N(int(N), dbname, _ids, 'simpleMix_top_%i.pkl')


adamix_top_N_filename = 'ada_mix_id_nf.pkl'

def adamix_top_N(N, trace_normalize='False', dryrun=False):
    dbname = 'final_random'
    filename=adamix_top_N_filename
    _ids = cPickle.load(open(filename))['_id']
    return blend_top_N(int(N), dbname, _ids, 'adaMix_top_%i.pkl', bool(dryrun))


def extract_kernel(ids_filename, position=None):
    # -- load index from PBS_ARRAYID

    if position is None:
        position = os.getenv('PBS_ARRAYID')
    array_ids = cPickle.load(open(ids_filename))
    assert position is not None
    position = int(position)
    print len(array_ids)
    print array_ids[position]
    _id = array_ids['_id'][position]
    return lfw_view2_fold_kernels_by_id('honeybadger', 'final_random', _id)


def simple_vs_ada_curves():
    sm_recarray = cPickle.load(open(simplemix_top_N_filename))
    am_recarray = cPickle.load(open(adamix_top_N_filename))

    sm_top_20 = cPickle.load(open('simpleMix_top_20.pkl'))
    am_top_49 = cPickle.load(open('adaMix_top_49.pkl'))
    pt_top_9 = cPickle.load(open('par_tpe_top_9.pkl'))

    sm20_means = [np.mean([sm_top_20[(i, j)] for j in range(10)]) for i in range(20)]
    am49_means = [np.mean([am_top_49[(i, j)] for j in range(10)]) for i in
            range(49)]
    pt9_means =[np.mean([pt_top_9[(i, j)] for j in range(10)]) for i in range(10)]

    sm20_feats = np.cumsum(sm_recarray['num_features'][:20])
    am49_feats = np.cumsum(am_recarray['num_features'][:49])

    plt.plot(sm20_feats, sm20_means)
    plt.plot(am49_feats, am49_means)
    # XXX: TODO: GET ACTUAL COUNTS
    plt.plot(sm20_feats[:10], pt9_means)
    plt.show()


def list_errors(dbname,host='localhost'):
    trials = MongoTrials(
            'mongo://%s:44556/%s/jobs' % (host, dbname),
            refresh=False)
    for doc in trials.handle:
        if doc['state'] == hyperopt.JOB_STATE_ERROR:
            print doc['_id'], doc['tid'], doc['book_time'], doc['error']
            print doc['spec']


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
    #tdocs = [(d['tid'], d) for d in docs if d['exp_key'].startswith('tpe_l3')]
    tdocs = [(d['tid'], d) for d in docs]
    tdocs.sort()
    by_key = {}
    for tid, d in tdocs:
        by_key.setdefault(d['exp_key'], []).append(d['result']['loss'])

    print len(by_key)
    kl_items = by_key.items()
    kl_items.sort()

    ROWS = int(np.ceil(len(kl_items) / 5.0))

    iii = 1
    for i, (k, losses) in enumerate(kl_items):
        minloss = min(losses)
        print k, 'min', minloss, [d['_id']
                for t, d in tdocs if d['result']['loss'] == minloss]
        plt.subplot(ROWS, 5, iii)
        plt.title(k)
        plt.scatter(range(len(losses)), losses)
        plt.ylim(.10, .55)
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

import pymongo as pm

def insert_consolidated_feature_shapes(dbname,
                                       trials=None,
                                       host='honeybadger.rowland.org',
                                       port=44556):
    conn = pm.Connection(host=host, port=port)
    Jobs = conn[dbname]['jobs']
    if trials is None:
        triallist = enumerate(Jobs.find({'result.status': hyperopt.STATUS_OK},
                                        fields=['spec','result.num_features'],
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


def get_top_tpe(N=1, dbname='feb29_par_tpe', host='honeybadger.rowland.org', port=44556):
    conn = pm.Connection(host=host, port=port)
    J = conn[dbname]['jobs']
    K = [k for k  in conn['feb29_par_tpe']['jobs'].distinct('exp_key') if 'Async' in k]
    R = [np.rec.array([(x['_id'], x['result']['loss'], x['result'].get('num_features',-1)) for x in J.find({'exp_key': k,
                                                      'misc.boosting.round': 0,
                                                      'result.status': hyperopt.STATUS_OK},
                                                     fields=['result.loss','result.num_features'])],
                      names = ['_id', 'loss', 'num_features'])
         for k in K]
    for r in R:
        r.sort(order=['loss'])
    R = [r[:N] for r in R]
    return R


def get_continues(tid, coll):
    assert coll.find({'tid': tid, 'result.status': hyperopt.STATUS_OK}).count() == 1
    new_misc = coll.find_one({'tid': tid})['misc']
    if 'boosting' in new_misc:
        new_tid = new_misc['boosting']['continues']
    else:
        assert 'from_tid' in new_misc
        new_tid = coll.find_one({'tid': new_misc['from_tid']})['misc']['boosting']['continues']
    
    if new_tid is not None:
        return [tid] + get_continues(new_tid, coll)
    else:
        return [tid]


def get_tpe_chain(k, dbname='feb29_par_tpe', host='honeybadger.rowland.org', port=44556):
    conn = pm.Connection(host=host, port=port)
    J = conn[dbname]['jobs']
    R = np.rec.array([(x['_id'], x['tid'], x['result']['loss']) for x in J.find({'exp_key':k, 'result.status': hyperopt.STATUS_OK},
                                                                                fields=['result.loss','tid'])],
                  names=['_id','tid','loss'])
    top_tid = int(R['tid'][R['loss'].argmin()])
    all_tids = get_continues(top_tid, J)
    R = np.rec.array([(x['_id'], x['tid'], x['result']['loss'], x['result'].get('num_features',-1))
                      for x in J.find({'tid': {'$in': all_tids}}, fields=['result.loss', 'tid', 'result.num_features'])],
                     names=['_id','tid','loss','num_features'])
    assert len(R) == len(all_tids)
    return R

def get_top_tpe_chains(dbname='feb29_par_tpe', host='honeybadger.rowland.org', port=44556):
    conn = pm.Connection(host=host, port=port)
    J = conn[dbname]['jobs']
    K = [k for k  in J.distinct('exp_key') if 'Async' in k]
    return [get_tpe_chain(k, dbname=dbname, host=host, port=port) for k in K]


par_tpe_top_N_filename = 'par_tpe_mix_id_nf_list.pkl'

def save_par_tpe_top_N():
    array_ids_list = get_top_tpe_chains()
    _ids = []
    for ail in array_ids_list:
        _ids.extend(ail['_id'])
    print 'Saving', len(_ids), 'IDs'
    print _ids
    cPickle.dump(array_ids_list, open(par_tpe_top_N_filename, 'w'))


def par_tpe_top_N(N, trace_normalize='False', dryrun=False):
    dbname='feb29_par_tpe'
    array_ids_list = cPickle.load(open(par_tpe_top_N_filename))
    _ids = []
    for ail in array_ids_list:
        _ids.extend(ail['_id'])
    return blend_top_N(int(N), dbname, _ids, 'par_tpe_top_%i.pkl', bool(dryrun))


def extract_kernel_par_tpe(position):
    position = int(position)
    array_ids_list = cPickle.load(open(par_tpe_top_N_filename))
    _ids = []
    for ail in array_ids_list:
        _ids.extend(ail['_id'])
    return lfw_view2_fold_kernels_by_id('honeybadger', 'feb29_par_tpe', _ids[position])


def show_vars(key=None, dbname='march1_1', host='honeybadger.rowland.org', port=44556):
    conn = pm.Connection(host=host, port=port)
    J = conn[dbname]['jobs']
    K = [k for k  in conn[dbname]['jobs'].distinct('exp_key')]
    for k in K:
        print k
    if key is None:
        raise NotImplementedError()
    else:
        exp_key = exp_keys.get(key, key)
        docs = list(
                conn[dbname]['jobs'].find(
                    {'exp_key': exp_key},
                    {
                        'tid': 1,
                        'state': 1,
                        'result.loss':1,
                        'result.status':1,
                        'spec':1,
                        'misc.tid':1,
                        'misc.idxs':1,
                        'misc.vals': 1,
                    }))
    import eccv12.lfw
    trials = hyperopt.trials_from_docs(docs, validate=False)
    hyperopt.plotting.main_plot_vars(trials, bandit=eccv12.lfw.MultiBanditL3())


