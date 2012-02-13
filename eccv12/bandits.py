"""
put bandits here
"""
import cPickle

import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh


def validate_result(result):
    assert set(result['decisions'].keys()) == set(result['labels'].keys())
    decs = result['decisions']
    labs = result['labels']
    assert all([len(decs[k]) == len(labs[k]) for k in decs])


class BaseBandit(gb.GensonBandit):
    # Required: self.param_gen

    def __init__(self, decisions=None):
        super(BaseBandit, self).__init__(source_string=gh.string(self.param_gen)) 
        self.decisions = decisions
        
    def status(self, result, config=None):
        try:
            return result.get('status', 'ok')
        except:
            print result.keys()
            raise

    def evaluate(self, config, ctrl):
        if not self.decisions is None:
            ctrl.attachments['decisions'] = cPickle.dumps(self.decisions)
        result = self.performance_func(config, ctrl)
        validate_result(result)
        return result
