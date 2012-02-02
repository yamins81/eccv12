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
                        


class CVPRTopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.cvpr_top,
            comparison='mult',
            )
             
             
class FG11TopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.fg11_top,
            comparison='mult',
            )
        


        
                
        
        
