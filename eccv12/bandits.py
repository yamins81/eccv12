"""
put bandits here
"""
import cPickle

import hyperopt


def validate_result(result):
    decs = np.asarray(result['decisions'])
    assert decs.ndim == 2
    labs = np.asarray(result['labels'])
    assert labs.ndim == 1
    assert decs.shape[0] == len(labs)
    
    
def validate_config(config):
    decs = np.asarray(config['decisions'])
    assert decs.ndim == 2


class BaseBandit(hyperopt.Bandit):
    # Required: self.param_gen

    def __init__(self):
        super(BaseBandit, self).__init__(self.param_gen)

    def status(self, result, config=None):
        return result.get('status', 'ok')

    def evaluate(self, config, ctrl):
        validate_config(config)
        result = self.performance_func(config, ctrl)
        validate_result(result)
        return result
