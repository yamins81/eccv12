import os
import time
import subprocess
import sys

import numpy as np
import pymongo as pm
import bson

import pyll
import hyperopt
from hyperopt import TreeParzenEstimator
from hyperopt.mongoexp import MongoTrials

import eccv12.lfw2 as lfw2
from eccv12.experiments import InterleaveAlgo

def lfw2_bandit_sample(rseed=1):
    bandit = lfw2.lfw_bandit(n_train=10, n_test=10, n_view2_per_split=10)
    #print 'EXPR'
    #print bandit.expr
    result = pyll.stochastic.sample(bandit.expr,
            np.random.RandomState(int(rseed)),
            print_trace=False)
    # attachments are big and binary,
    # don't print them explicitly
    attachments = result.pop('attachments', {})
    print 'RESULT', result
    if attachments:
        print "RESULT attachments keys"
        for k, v in attachments.items():
            print '  name=%s, size=%i' % (k, len(v))

if __name__ == '__main__':
    lfw2_bandit_sample(sys.argv[1])
