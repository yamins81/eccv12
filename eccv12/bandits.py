"""
put bandits here
"""
import cPickle

import hyperopt


def validate_result(result):
    assert set(result['decisions'].keys()) == set(result['labels'].keys())
    decs = result['decisions']
    assert decs.ndim == 2
    labs = result['labels']
    assert labs.ndim == 1
    assert decs.shape[0] = len(labs)
    

class BaseBandit(hyperopt.Bandit):
    # Required: self.param_gen

    def __init__(self, decisions=None):
        super(BaseBandit, self).__init__(self.param_gen)
        self.decisions = decisions

    def status(self, result, config=None):
        return result.get('status', 'ok')

    def evaluate(self, config, ctrl):
        if not self.decisions is None:
            ctrl.attachments['decisions'] = cPickle.dumps(self.decisions)
        result = self.performance_func(config, ctrl)
        validate_result(result)
        return result
