import copy
from hyperopt.genson_helpers import null
#from hyperopt.genson_helpers import false
#from hyperopt.genson_helpers import true
from hyperopt.genson_helpers import choice
from hyperopt.genson_helpers import uniform
#from hyperopt.genson_helpers import gaussian
#from hyperopt.genson_helpers import lognormal
#from hyperopt.genson_helpers import qlognormal
from hyperopt.genson_helpers import ref


lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : uniform(0,10),
         'threshold' : choice([.1,1,10])
         }}

lpool = {'kwargs': {'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1, 2, 10])
         }}

lpool_sub2 = {'kwargs': {'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1, 2, 10]),
          'stride': 2,
         }}

rescale = {'kwargs': {'stride' : 2}}

activ =  {'kwargs': {'min_out' : choice([null, 0]),
                     'max_out' : choice([1, null])}}

filter1 = dict(
         initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16, 32, 64]),
            generate=(
                'random:uniform',
                {'rseed': choice(range(5))})),
         kwargs={})

filter2 = copy.deepcopy(filter1)
filter2['initialize']['n_filters'] = choice([16, 32, 64, 128])
filter2['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(5,10))})

filter3 = copy.deepcopy(filter1)
filter3['initialize']['n_filters'] = choice([16, 32, 64, 128, 256])
filter3['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(10,15))})


fg11_desc = [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('lpool', lpool_sub2),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('lpool', lpool_sub2),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('lpool', lpool_sub2),
             ('lnorm', lnorm)],
           ]
