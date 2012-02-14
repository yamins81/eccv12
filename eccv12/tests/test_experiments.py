"""
testing experiment classes

as of now, these tests are mostly for demonstrating the api -- they don't really
test anything of importance, since they're not really good for unit tests, being
too slow.  We'll have to figure out exactly what to test.   There's also testing
for error and exception and re-starting handling, which is hard but important.
"""

import hyperopt
import hyperopt.experiments

import eccv12.experiments as experiments
import eccv12.lfw


def test_boosted_serial():
    exp = experiments.BoostedSerialExperiment(hyperopt.Random,
                                              eccv12.lfw.TestBandit,
                                              2, 1)
    exp.run()
    assert len(exp.results) == 2
    

def test_boosted_mongo():
    exp = experiments.BoostedMongoExperiment('hyperopt.Random',
                                             'eccv12.lfw.TestBandit',
                                             2,
                                             1,
                                             mongo_opts='localhost:27017/testdb')
    exp.run()
    assert len(exp.results) == 2
        

def test_mixtures():
    M = 5
    N = 2
    
    bandit_algo = hyperopt.Random(eccv12.lfw.TestBandit())
    exp = hyperopt.experiments.SerialExperiment(bandit_algo)
    exp.run(M)
    simple = experiments.SimpleMixture(exp)
    inds, weights = simple.mix_inds(N)
    losses = np.array([_r['loss'] for _r in exp.results])
    s = losses.argsort()
    assert (inds == s[:N]).all()
    
    ada = experiments.AdaboostMixture(exp)
    ada_inds, ada_weights = ada.mix_inds(N)
    #I'm not 100 sure exactly what to test here ...
    

def test_parallel_mongo():
    """
        I've only testing the class experiments.ParallelMongoExperiment
        informally ...
    """
    pass
