import cPickle
import string
import numpy as np
import unittest
import scipy.io

import skdata.lfw
from hyperopt.genson_helpers import choice

import eccv12.plugins
from eccv12.plugins import (slm_memmap,
                            pairs_memmap,
                            verification_pairs,
                            get_images,
                            pairs_cleanup,
                            delete_memmap)
                        


class Bandit(eccv12.plugins.Bandit):
    desc  = [[('lnorm',{'kwargs':{'inker_shape': (9, 9),
                         'outker_shape': (9, 9),
                         'stretch':10,
                         'threshold': 1}})], 
             [('fbcorr', {'initialize': {'filter_shape': (3, 3),
                                         'n_filters': 64,
                                         'generate': ('random:uniform', 
                                                      {'rseed': choice(range(5))})},
                          'kwargs':{'min_out': 0,
                                    'max_out': None}}),
              ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                    'order': 1,
                                    'stride': 2}}),
              ('lnorm', {'kwargs': {'inker_shape': (5, 5),
                                    'outker_shape': (5, 5),
                                    'stretch': 0.1,
                                    'threshold': 1}})],
             [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                         'n_filters': 128,
                                         'generate': ('random:uniform',
                                                      {'rseed': choice(range(5))})},
                          'kwargs': {'min_out': 0,
                                     'max_out': None}}),
              ('lpool', {'kwargs': {'ker_shape': (5, 5),
                                    'order': 1,
                                    'stride': 2}}),
              ('lnorm', {'kwargs': {'inker_shape': (7, 7),
                                    'outker_shape': (7, 7),
                                    'stretch': 1,
                                    'threshold': 1}})],
             [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                         'n_filters': 256,
                                         'generate': ('random:uniform',
                                                     {'rseed': choice(range(5))})},
                          'kwargs': {'min_out': 0,
                                     'max_out': None}}),
               ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                     'order': 10,
                                     'stride': 2}}),
               ('lnorm', {'kwargs': {'inker_shape': (3, 3),
                                     'outker_shape': (3, 3),
                                     'stretch': 10,
                                     'threshold': 1}})]]

    param_gen = dict(
            slm=desc,
            comparison='mult',
            )
            
    def evaluate(self, config, ctrl):
        
        preprocs = [('prenorm', {'global_normalize': True}),
                    ('noprenorm', {'global_normalize': False})]
        
        result = {}
        for (lbl, preproc) in preprocs:
            prog = screening_program(
                    slm_desc=config['slm'],
                    comparison=config['comparison'],
                    preproc=preproc,
                    namebase='memmap_')['rval']
    
            scope = dict(
                    ctrl=ctrl,
                    decisions={},
                    )
            # XXX: hard-codes self.train_decisions to be DevTrain - what happens
            # in view 2?
            if self.train_decisions is None:
                scope['decisions']['DevTrain'] = np.zeros(2200)
            else:
                scope['decisions']['DevTrain'] = self.train_decisions
    
            if self.test_decisions is None:
                scope['decisions']['DevTest'] = np.zeros(1000)
            else:
                scope['decision']['DevTest'] = self.test_decisions
    
            fson_eval(prog, scope=scope)
    
            result[lbl] = scope['result']
            
        result['loss'] = np.mean([v['loss'] for v in result.values()])
        
        return result
        
        


        
                
        
        