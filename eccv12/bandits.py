"""
put bandits here
"""
import cPickle

import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh

import lfw


class BaseBandit(gb.GensonBandit):
    # Required: self.param_gen

    def __init__(self, decisions = None):
        super(BaseBandit, self).__init__(source_string=gh.string(self.param_gen)) 
        self.decisions = decisions
        
    def status(self, result, config=None):
        try:
            return result.get('status', 'ok')
        except:
            print result.keys()
            raise

    def evaluate(self, config, ctrl):
        result = self.performance_func(config, self.decisions)
        assert 'decisions' in result
        model_data = {'weights': result.pop('weights'),
                      'bias': result.pop('bias')}
        model_blob = cPickle.dumps(model_data)
        ctrl.attachment['model_data'] = model_blob
        return result


class LFWBase(object):
    """
    config is a dictionary with keys:
    - slm - config for SLM model (e.g. TheanoSLM),
    - comparison - name from ".comparison" module

    """
    ###this does not work right now, since lfw_get_performance expects
    ###both train_dcisiosn and test_decisions ... the exact format needs to be 
    ###worked out
    def performance_func(self, config, decisions):
        return lfw.get_performance(config['slm'],
                                   decisions,
                                   config['comparison'])


class LFWBandit(BaseBandit, LFWBase):
    pass

