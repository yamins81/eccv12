"""
put bandits here
"""
import cPickle

import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh

import lfw
import model_params


class BaseBandit(gb.GensonBandit):
    param_gen = model_params.l3_params 

    def __init__(self, train_decisions, test_decisions, attach_weights):
        super(BaseBandit, self).__init__(source_string=gh.string(self.param_gen))
        self.train_decisions = train_decisions
        self.test_decisions = test_decisions
        self.attach_weights = attach_weights

    def evaluate(self, config, ctrl):
        result = self.performance_func(config, 
                                       self.train_decisions,
                                       self.test_decisions)
        assert 'train_decisions' in result
        assert 'test_decisions' in result
        
        if self.attach_weights:
            model_data = {'weights': result.pop('weights'),
                          'bias': result.pop('bias')}                   
            model_blob = cPickle.dumps(model_data)
            ctrl.set_attachment(model_blob, 'model_data')
            
        return result

    
class LFWBase(object):
    def performance_func(self, config, train_decisions, test_decisions):
        return lfw.get_performance(config,
                                   train_decisions,
                                   test_decisions)
    
    
class LFWBandit(BaseBandit, LFWBase):
    pass
