"""
Experiment generator classes for easily creating the kinds of search-strategy
comparisons we care about for the eccv paper.   Classes also contain
conventions for saving out results and then (soon enough) generating figures.

This stuff is still in the process of being tested.

Entry point is to call something like

* `run_random_experiment()`

* `run_tpe_experiment()`

"""
import copy
import cPickle
import logging
import sys

logger = logging.getLogger(__name__)
import numpy as np

try:
    from collections import OrderedDict
except ImportError:
    print "Python 2.7+ OrderedDict collection not available"
    try:
        from ordereddict import OrderedDict
        logger.warn("Using backported OrderedDict implementation")
    except ImportError:
        raise ImportError("Backported OrderedDict implementation "
                          "not available. To install it: "
                          "'pip install -vUI ordereddict'")


import hyperopt
import hyperopt.plotting
from hyperopt import STATUS_RUNNING, STATUS_NEW, StopExperiment
from hyperopt.mongoexp import MongoTrials, as_mongo_str

from .lfw import MultiBandit

from .experiments import SyncBoostingAlgo
from .experiments import AsyncBoostingAlgoA
from .experiments import AsyncBoostingAlgoB
from .experiments import AdaboostMixture
from .experiments import SimpleMixture
from .experiments import InterleaveAlgo


def cname(cls):
    return cls.__class__.__module__ + '.' + cls.__class__.__name__


# -- keep tests running
class LFWBandit(MultiBandit): pass



class SearchExp(object):
    """
    Basic control experiment against which to compare to other approaches.

    num_features - the experiment will search a bandit that is configured to
            deliver this many features.

    """
    def __init__(self, num_features, bandit_func, bandit_algo_class, exp_prefix,
            trials=None, mongo_opts=None, walltime_cutoff=np.inf,
            ntrials=None):
        # if trials is None, then mongo_opts is used to create a MongoTrials,
        # otherwise it is ignored.
        #
        self.num_features = num_features
        self.bandit_algo_class = bandit_algo_class
        self.bandit = bandit_func(num_features)
        self.init_bandit_algo()
        self.exp_prefix = exp_prefix
        self.walltime_cutoff = walltime_cutoff
        assert ntrials is not None
        self.ntrials = ntrials

        if trials is None:
            trials = MongoTrials(as_mongo_str(mongo_opts) + '/jobs',
                                      exp_key=self.get_exp_key())
            #trials = Trials()

        self.trials = trials
        self.exp_key = self.trials._exp_key

    def init_bandit_algo(self):
        self.bandit_algo = self.bandit_algo_class(self.bandit,
                                     cmd=('driver_attachment', 'bandit_data'))

    def get_info(self):
        """
        return a dictionary containing identifying information about the experiment
        """
        return OrderedDict(
            num_features=self.num_features,
            bandit=cname(self.bandit),
            bandit_algo=cname(self.bandit_algo)
            )

    def get_exp_key(self):
        """
        turn identifying information into a mongo experiment key
        """
        info = self.get_info()
        tag = '_'.join([k + ':' + str(v) for (k, v) in info.items()])
        return self.exp_prefix + tag

    def get_filename(self, ntrials):
        """
        turn indentifying information into a filename for results saveout
        """
        info = self.get_info()
        tag = '_'.join([k + ':' + str(v) for (k, v) in info.items()])
        return self.exp_prefix + tag + ('_%d' % ntrials) + '.pkl'

    def get_result(self):
        trial_info = self.get_info()
        trial_info['trials'] = self.trials.view(exp_key=self.get_exp_key())
        return trial_info

    def prepare_trials(self):
        bandit_name = self.get_info()['bandit']
        bandit_args = (self.num_features,)
        bandit_kwargs = {}
        blob = cPickle.dumps((bandit_name, bandit_args, bandit_kwargs))
        self.trials.attachments['bandit_data'] = blob

    def get_bandit_algo(self):
        rval = NtrialsBanditAlgo(self.bandit_algo, ntrials=self.ntrials,
                walltime_cutoff=self.walltime_cutoff)
        return rval

    def save(self):
        """
        assemble results and save them out to a pkl file
        """
        result = self.get_result()
        info = self.get_info()
        self.trials.refresh()
        ntrials = len(self.trials.results)
        cPickle.dump(result, open(self.get_filename(ntrials), 'w'))

    def delete_all(self):
        self.trials.delete_all()

    def run(self):
        """
        Keeps suggesting jobs from each SearchExp until they are all done.
        """
        self.prepare_trials()
        algo = self.get_bandit_algo()
        exp = hyperopt.Experiment(self.trials, algo)
        exp.run(sys.maxint, block_until_done=True)


class NtrialsBanditAlgo(hyperopt.BanditAlgo):
    def __init__(self, base_bandit_algo, ntrials, walltime_cutoff, **kwargs):
        hyperopt.BanditAlgo.__init__(self, base_bandit_algo.bandit, **kwargs)
        self.base_bandit_algo = base_bandit_algo
        self.ntrials = ntrials
        self.walltime_cutoff = walltime_cutoff

    def __str__(self):
        return 'NtrialsBanditAlgo{%i, %s}' % (self.ntrials, self.base_bandit_algo)

    def filter_oks(self, trials):
        OKs = [t for t in trials
                          if t['result']['status'] == hyperopt.STATUS_OK]
        FAILs = [t for t in trials
                        if t['result']['status'] == hyperopt.STATUS_FAIL]
        for t in FAILs:
            wall_time = (t['refresh_time'] - t['book_time']).total_seconds()
            if wall_time > self.walltime_cutoff:
                OKs.append(t)
        return OKs

    def suggest(self, new_ids, trials):
        OKs = self.filter_oks(trials)
        if len(OKs) >= self.ntrials:
            return StopExperiment()
        else:
            UNFINISHED = [t for t in trials 
                      if t['result']['status'] in [STATUS_RUNNING, STATUS_NEW]]
            new_ids = new_ids[: self.ntrials - len(OKs) - len(UNFINISHED)]
            if not new_ids:
                return []
            else:
                return self.base_bandit_algo.suggest(new_ids, trials)


class MixtureExp(SearchExp):
    """
    Mixture version of the class.  (just basically adds mixture info to
    the identifying information)
    """
    def __init__(self, mixture_class, mixture_kwargs, ensemble_size, *args, **kwargs):
        self.mixture_class = mixture_class
        self.mixture_kwargs = mixture_kwargs
        self.ensemble_size = ensemble_size
        SearchExp.__init__(self, *args, **kwargs)
        self.mixture = self.mixture_class(self.trials, self.bandit,
                                          **mixture_kwargs)

    def get_info(self):
        info = SearchExp.get_info(self)
        info['mixture'] = cname(self.mixture_class(None, None, **self.mixture_kwargs))
        info['mixture_kwargs'] = self.mixture_kwargs
        info['ensemble_size'] = self.ensemble_size
        return info

    def get_result(self):
        trial_info = SearchExp.get_result(self)
        inds, weights = self.mixture.mix_inds(self.ensemble_size)
        trial_info['mixture_inds'] = inds
        trial_info['mixture_weights'] = weights
        return trial_info


class MetaExp(SearchExp):
    """
    Version for having meta-bandit-algos, e.g. boosting, parallel.
    """
    def __init__(self, meta_algo_class, meta_kwargs, *args, **kwargs):
        self.meta_algo_class = meta_algo_class
        self.meta_kwargs = meta_kwargs
        SearchExp.__init__(self, *args, **kwargs)

    def init_bandit_algo(self):
        """
        wrap the original bandit algo in the meta bandit algo
        """
        self.base_bandit_algo = self.bandit_algo_class(self.bandit,
                                      cmd=('driver_attachment', 'bandit_data'))
        self.bandit_algo = self.meta_algo_class(self.base_bandit_algo,
                                                **self.meta_kwargs)

    def get_info(self):
        info = SearchExp.get_info(self)
        info['meta_algo'] = info.pop('bandit_algo')
        info['bandit_algo'] = cname(self.base_bandit_algo)
        info['meta_kwargs'] = self.meta_kwargs
        return info


class NestedExperiment(object):
    """
    Basic class for nested experiments.  The purpose of this class is to make
    it possible to run nested-style experiments in whatever order one wants
    and to obtain information about them easily.

    Derived classes must implement add_experiments method.
    N.B. These methods take name that is a *tuple*.
         The `name` is a list/tuple of strings... these index into a hierarchy
         of nested experiments.
    """
    def __init__(self, trials, ntrials, save, *args, **kwargs):
        self.trials = trials
        self.experiments = OrderedDict([])
        self.ntrials = ntrials
        self.save = save
        if args or kwargs:
            self.add_experiments(*args, **kwargs)

    def add_exp(self, exp, tag):
        self.experiments[tag] = exp

    def get_experiment(self, name):
        if len(name) == 0:
            return self
        else:
            e = self.experiments[name[0]]
            if isinstance(e, NestedExperiment):
                return e.get_experiment(name[1:])
            else:
                return e

    def flatten(self):
        rval = []
        for exp in self.experiments.values():
            if hasattr(exp, 'flatten'):
                rval.extend(exp.flatten())
            else:
                rval.append(exp)
        return rval

    def interleaved_algo(self):
        search_exps = self.flatten()
        # XXX assert that all search_exps have same bandit
        search_exps[0].prepare_trials()
        algos = [se.get_bandit_algo() for se in search_exps]
        keys = [se.get_exp_key() for se in search_exps]
        rval = InterleaveAlgo(algos, keys)
        return rval

    def run(self):
        """
        Keeps suggesting jobs from each SearchExp until they are all done.
        """
        rval = self.interleaved_algo()
        exp = hyperopt.Experiment(self.trials, rval)
        # -- the interleaving algo will break out of this
        exp.run(sys.maxint, block_until_done=True)

    def delete_all(self, name=()):
        exp = self.get_experiment(name)
        if isinstance(exp, NestedExperiment):
            for exp0 in exp.experiments.values():
                exp0.delete_all()
        else:
            exp.delete_all()

    def save(self, name=()):
        exp = self.get_experiment(name)
        if isinstance(exp, NestedExperiment):
            for exp0 in exp.experiments.vales():
                exp0.save()
        else:
            exp.save()

    def get_info(self, name=()):
        exp = self.get_experiment(name)
        if isinstance(exp, NestedExperiment):
            res = {}
            for exp0_name in exp.experiments:
                exp0 = exp.experiments[exp0_name]
                res[exp0_name] = exp0.get_info()
            return res
        else:
            return exp.get_info()

    def pretty_info(self, name=(), indent=0):
        exp = self.get_experiment(name)
        if isinstance(exp, NestedExperiment):
            for exp0_name in exp.experiments:
                exp0 = exp.experiments[exp0_name]
                print ' ' * indent + exp0_name
                if isinstance(exp0, NestedExperiment):
                    exp0.pretty_info(indent=indent+2)
                else:
                    print ' ' * (indent+2) + str(exp0.get_info())
        else:
            print ' ' * indent + str(exp.get_info())

    def get_result(self, name=()):
        exp = self.get_experiment(name)
        if isinstance(exp, NestedExperiment):
            res = {}
            for exp0_name in exp.experiments:
                exp0 = exp.experiments[exp0_name]
                res[exp0_name] = exp0.get_result()
            return res
        else:
            return exp.get_result()


    #####plotting code goes here also


class ComparisonExperiment(NestedExperiment):
    """Compare various approaches to ensemble construction.
    """
    def add_experiments(self, num_features, round_len, ensemble_size,
                 bandit_func, bandit_algo_class, exp_prefix,
                 run_parallel, adamix_kwargs):

        std_kwargs = dict(
                num_features=num_features,
                bandit_func=bandit_func,
                bandit_algo_class=bandit_algo_class,
                exp_prefix=exp_prefix,
                ntrials=self.ntrials,
                trials=self.trials)

        if 1:
            basic_exp = SearchExp(**std_kwargs)
            self.add_exp(basic_exp, 'basic')

        if 0:
            simple_mix = MixtureExp(
                    mixture_class=SimpleMixture,
                    mixture_kwargs={},
                    ensemble_size=ensemble_size,
                    **std_kwargs)
            self.add_exp(simple_mix, 'simple_mix')

        if 0:
            ada_mix = MixtureExp(
                    mixture_class=AdaboostMixture,
                    mixture_kwargs=adamix_kwargs,
                    ensemble_size=ensemble_size,
                    **std_kwargs)
            self.add_exp(ada_mix, 'ada_mix')

        if 0:
            syncboost_exp = MetaExp(
                    meta_algo_class=SyncBoostingAlgo,
                    meta_kwargs={"round_len": round_len},
                    **std_kwargs)
            self.add_exp(syncboost_exp, 'SyncBoost')

        if 0:
            asyncboostA_exp = MetaExp(
                    meta_algo_class=AsyncBoostingAlgoA,
                    meta_kwargs={"round_len": round_len},
                    **std_kwargs)
            self.add_exp(asyncboostA_exp, 'AsyncBoostA')

        if 1:
            asyncboostB_exp = MetaExp(
                    meta_algo_class=AsyncBoostingAlgoB,
                    meta_kwargs={"round_len": round_len},
                    **std_kwargs)
            self.add_exp(asyncboostB_exp, 'AsyncBoostB')

        if run_parallel:
            parallel_exp = MetaExp(
                    # XXX: Dan where is this class defined?
                    meta_algo_class=ParallelBoostingAlgo,
                    meta_kwargs={"num_procs": ensemble_size},
                    **std_kwargs)
            self.add_exp(parallel_exp, 'parallel')


class BudgetExperiment(NestedExperiment):
    """
    For a given budget, explore comparisons in various ways for various
    sizes of ensembles.


    self.ntrials * num_features is the total budget for feature evaluation
    during training.

    num_features is the total budget for features in the final model.

    This function sets up several experiments, that partition num_features
    into various numbers (ensemble_sizes[i]) of feature sets.

    N.B. that in the context of LFW and pythor-style feature extraction in
    particular, the num features will always be multiplied by the output
    feature-map size... which is a somewhat complicated function of many
    parameters in the search space.  This introduces noise into the process of
    trying to equalize experiment sizes, but anyway the equality was never
    really quite there so no huge loss.

    """
    def add_experiments(self, num_features,
                   ensemble_sizes,
                   bandit_func,
                   bandit_algo_class,
                   exp_prefix,
                   trials=None,
                   run_parallel=False):

        if trials is None:
            trials = self.trials
        ntrials = self.ntrials
        save = self.save
        # -- search models sampled from `bandit_func(num_features)`
        #    using search algorithm `bandit_algo_class`
        if 0: # XXX bring back later, too slow...
            control_exp = SearchExp(num_features=num_features,
                      bandit_func=bandit_func,
                      bandit_algo_class=bandit_algo_class,
                      exp_prefix=exp_prefix,
                      ntrials=ntrials,
                      trials=trials)
            self.add_exp(control_exp, 'control')

        for es in ensemble_sizes:
            #trade off ensemble size for more trials, fixed final feature size
            assert num_features % es == 0
            _C = ComparisonExperiment(trials=trials,
                               ntrials=ntrials * es,
                               num_features=num_features / es,
                               round_len=ntrials,
                               save=save,
                               ensemble_size=es,
                               bandit_func=bandit_func,
                               bandit_algo_class=bandit_algo_class,
                               exp_prefix=exp_prefix,
                               run_parallel=run_parallel,
                               adamix_kwargs={'test_mask':True})
            self.add_exp(_C, 'fixed_features_%d' % es)

            if 0: # -- bring in later
                # trade off ensemble size for more features,
                # fixed number of trials
                _C = ComparisonExperiment(trials=trials,
                               ntrials=ntrials,
                               save=save,
                               num_features=num_features,
                               round_len=ntrials / es,
                               ensemble_size=es,
                               bandit_func=bandit_func,
                               bandit_algo_class=bandit_algo_class,
                               exp_prefix=exp_prefix,
                               run_parallel=run_parallel,
                               adamix_kwargs={'test_mask':True})
                self.add_exp(_C, 'fixed_trials_%d' % es)


def main_lfw_driver(trials):
    def add_exps(bandit_algo_class, exp_prefix):
        B = BudgetExperiment(ntrials=200, save=False, trials=trials,
                num_features=128 * 10,
                ensemble_sizes=[10],
                bandit_func=MultiBandit,
                bandit_algo_class=bandit_algo_class,
                exp_prefix=exp_prefix,
                run_parallel=False) # XXX?
        return B
    N = NestedExperiment(trials=trials, ntrials=200, save=False)
    N.add_exp(add_exps(hyperopt.Random, 'ek_random'), 'random')
    N.add_exp(add_exps(hyperopt.TreeParzenEstimator, 'ek_tpe'), 'TPE')
    return N

# The driver code for these classes is in scripts/main.py
#
#

