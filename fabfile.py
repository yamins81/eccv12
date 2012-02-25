"""
Usage:

e.g. to show all the errors in database with dbname="try2", type

$ fab list_errors:try2


"""
from fabric.api import run  # -- shutup pyflakes, we need this

import copy
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
    assert 0, "go and disable the assertion if you really want to delete all"
    trials = MongoTrials('mongo://localhost:44556/%s/jobs' % dbname)
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
