import cPickle
import string
import numpy as np
import unittest
import scipy.io

import skdata.lfw

import model_params
import plugins
from plugins import (slm_memmap,
                     pairs_memmap,
                     verification_pairs,
                     get_images,
                     pairs_cleanup,
                     delete_memmap,
                     screening_program,
                     fson_eval)
                        


class Bandit(plugins.Bandit):

    param_gen = dict(
            slm=model_params.lfwtop,
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
        
        


        
                
        
        
