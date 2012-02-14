"""
Testing experiment classes that use mongo
"""

import hyperopt
import hyperopt.experiments

import eccv12.experiments as experiments
from eccv12.toyproblem import BoostableDigits


def test_boosted_mongo():
    exp = experiments.BoostedMongoExperiment('hyperopt.Random',
                                             'eccv12.lfw.TestBandit',
                                             2,
                                             1,
                                             mongo_opts='localhost:27017/testdb')
    exp.run()
    assert len(exp.results) == 2


def test_parallel_mongo():
    """
        I've only testing the class experiments.ParallelMongoExperiment
        informally ...
    """
    raise NotImplementedError()

