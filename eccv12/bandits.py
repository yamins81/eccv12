"""
put bandits here
"""
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh

import lfw
import model_params


class BaseBandit(gb.GensonBandit):
    param_gen = model_params.l3_params 

    def __init__(self, training_weights):
        super(BaseBandit, self).__init__(source_string=gh.string(self.param_gen))
        self.training_weights = training_weights

    def evaluate(self, config, ctrl):
        return self.performance_func(config, self.training_weights)
        

class LFWBase(object):
    def performance_func(config, ctrl, training_weights):
        return lfw.get_performance(config, training_weights=training_weights)
    
    
class LFWBandit(BaseBandit, LFWBase):
    pass
