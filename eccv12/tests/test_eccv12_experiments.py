import copy
import numpy as np
import pyll
import hyperopt
import eccv12.eccv12 as exps
import eccv12.experiments as experiments
from eccv12.bandits import BaseBandit

try:
    from collections import OrderedDict
except ImportError:
    print "Python 2.7+ OrderedDict collection not available"
    try:
        from ordereddict import OrderedDict
        warn("Using backported OrderedDict implementation")
    except ImportError:
        raise ImportError("Backported OrderedDict implementation "
                          "not available. To install it: "
                          "'pip install -vUI ordereddict'")


def test_mixture_initializes():
    S = exps.MixtureExp(experiments.AdaboostMixture,
                        5,
                        10,
                        exps.num_features_lfw,
                        hyperopt.Random,
                        "localhost:22334/test_hyperopt",
                        "test_stuff")

    assert S.get_info() == OrderedDict([('bandit', 'eccv12.eccv12.LFWBandit'),
                 ('num_features', 10),
                 ('bandit_algo', 'hyperopt.base.Random'),
                 ('mixture', 'eccv12.experiments.AdaboostMixture'),
                 ('ensemble_size', 5)])
    
    assert S.get_exp_key() == 'test_stuffbandit:eccv12.eccv12.LFWBandit_num_features:10_bandit_algo:hyperopt.base.Random_mixture:eccv12.experiments.AdaboostMixture_ensemble_size:5'
                        
                        
def test_meta_initializes():
    S = exps.MetaExp(experiments.AsyncBoostingAlgo,
                    {"round_len":5, "look_back":1},
                    10,
                    exps.num_features_lfw,
                    hyperopt.Random,
                    "localhost:22334/test_hyperopt",
                    "test_stuff")
                    
    assert S.get_info() == OrderedDict([('bandit', 'eccv12.eccv12.LFWBandit'),
                 ('num_features', 10),
                 ('meta_algo', 'eccv12.experiments.AsyncBoostingAlgo'),
                 ('bandit_algo', 'hyperopt.base.Random'),
                 ('meta_kwargs', {'look_back': 1, 'round_len': 5})])

                    
def test_search_initializes():
    S = exps.SearchExp(10,
                       exps.num_features_lfw,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
                       
    assert S.get_info() == OrderedDict([('bandit', 'eccv12.eccv12.LFWBandit'),
               ('num_features', 10), ('bandit_algo', 'hyperopt.base.Random')])
               

class DummyDecisionsBandit(BaseBandit):
    param_gen = dict(
            seed=pyll.scope.randint(1000),
            decisions=None)
            
    def __init__(self, n_features):
        BaseBandit.__init__(self)
        self.n_features = n_features

    def performance_func(self, config, ctrl):
        r34 = np.random.RandomState(34)
        y = np.sign(r34.randn(self.n_features))
        rs = np.random.RandomState(int(config['seed']))
        yhat = rs.normal(loc=0, scale=1, size=(1, self.n_features))
        decisions = config['decisions']
        if decisions is None:
            decisions = np.zeros((1, self.n_features))     
        else:
            decisions = np.array(decisions)
        new_dec = yhat + decisions
        is_test = np.ones(decisions.shape)
        result = dict(
                loss=np.mean(y != np.sign(new_dec)),
                labels=y.tolist(),
                decisions=new_dec.tolist(),
                is_test=is_test.tolist())
        return result


def test_search_dummy():
    S = exps.SearchExp(10,
                       DummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
    S.trials.delete_all()
    S.run(10)
    assert len(S.trials.results) == 10
    assert 1 > np.mean([x['loss'] for x in S.trials.results]) > 0
    T = copy.deepcopy(S.trials.results)
    S.run(20)
    assert all([t == s for t, s in zip(T, S.trials.results[:10])])
    
    
def test_mix_dummy():
    S = exps.MixtureExp(experiments.AdaboostMixture,
                        5,
                        10,
                       DummyDecisionsBandit,
                       hyperopt.Random,
                       "localhost:22334/test_hyperopt",
                       "test_stuff")
    S.trials.delete_all()
    S.run(20)
    res = S.get_result()
    assert len(res['mixture_inds']) == 5
    assert res['mixture_weights'].shape == (5, 1)
    

def test_meta_dummy():
    """
    THIS TEST IS NOT YET COMPLETE:  most of the time it works
    but then for reasons I don't yet know i sometimes see:
       331                 last_best = losses.argmin() 
       TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
       This problem appears to go away if you re-run the test ... 
       This has to be investiaged further.
       
    """
    S = exps.MetaExp(experiments.SyncBoostingAlgo,
                    {"round_len": 5},
                    10,
                   DummyDecisionsBandit,
                   hyperopt.Random,
                   "localhost:22334/test_hyperopt",
                   "test_stuff")
    S.trials.delete_all()
    S.run(10)
    selected = S.bandit_algo.best_by_round(list(S.trials))
    assert len(selected) == 2
    T = copy.deepcopy(S.trials.results)
    S.run(20)
    assert all([t == s for t, s in zip(T, S.trials.results[:10])])
    selected = S.bandit_algo.best_by_round(list(S.trials))
    assert len(selected) == 4


def test_budget_experiment():
    S = exps.BudgetExperiment(ntrials=10, 
                       save=False,
                       num_features=10,
                       ensemble_sizes=[2, 5],
                       bandit_func=DummyDecisionsBandit,
                       bandit_algo_class=hyperopt.Random,
                       exp_prefix='test_stuff',
                       mongo_opts='localhost:22334/test_hyperopt',
                       look_back=1,
                       run_parallel=False)
    S.delete_all()
    S.run()
                      