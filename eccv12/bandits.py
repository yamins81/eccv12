"""
put bandits here
"""
import cPickle

import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh

import lfw


class BaseBandit(gb.GensonBandit):
    # Required: self.param_gen

    def __init__(self, attach_weights=False):
        super(BaseBandit, self).__init__(source_string=gh.string(self.param_gen))
        self.attach_weights = attach_weights

    def status(self, result, config=None):
        try:
            return result.get('status', 'ok')
        except:
            print result.keys()
            raise

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
    """
    config is a dictionary with keys:
    - slm - config for SLM model (e.g. TheanoSLM),
    - comparison - name from ".comparison" module

    """
    def performance_func(self, config, train_decisions, test_decisions):
        return lfw.get_performance(config['slm'],
                                   train_decisions,
                                   test_decisions,
                                   config['comparison'])


class LFWBandit(BaseBandit, LFWBase):
    pass

