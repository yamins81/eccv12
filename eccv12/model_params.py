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


lfwtop = [[('lnorm',{'kwargs':{'inker_shape': (9, 9),
                     'outker_shape': (9, 9),
                     'stretch':10,
                     'threshold': 1}})], 
         [('fbcorr', {'initialize': {'filter_shape': (3, 3),
                                     'n_filters': 64,
                                     'generate': ('random:uniform', 
                                                  {'rseed': choice(range(5))})},
                      'kwargs':{'min_out': 0,
                                'max_out': null}}),
          ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                'order': 1,
                                'stride': 2}}),
          ('lnorm', {'kwargs': {'inker_shape': (5, 5),
                                'outker_shape': (5, 5),
                                'stretch': 0.1,
                                'threshold': 1}})],
         [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                     'n_filters': 128,
                                     'generate': ('random:uniform',
                                                  {'rseed': choice(range(5))})},
                      'kwargs': {'min_out': 0,
                                 'max_out': null}}),
          ('lpool', {'kwargs': {'ker_shape': (5, 5),
                                'order': 1,
                                'stride': 2}}),
          ('lnorm', {'kwargs': {'inker_shape': (7, 7),
                                'outker_shape': (7, 7),
                                'stretch': 1,
                                'threshold': 1}})],
         [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                     'n_filters': 256,
                                     'generate': ('random:uniform',
                                                 {'rseed': choice(range(5))})},
                      'kwargs': {'min_out': 0,
                                 'max_out': null}}),
           ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                 'order': 10,
                                 'stride': 2}}),
           ('lnorm', {'kwargs': {'inker_shape': (3, 3),
                                 'outker_shape': (3, 3),
                                 'stretch': 10,
                                 'threshold': 1}})]]

fg11_top = lfwtop

cvpr_top = [[('lnorm',
       {'kwargs': {'inker_shape': [5, 5],
         'outker_shape': [5, 5],
         'remove_mean': 0,
         'stretch': 0.1,
         'threshold': 1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [5, 5],
         'generate': ['random:uniform', {'rseed': 12}],
         'n_filters': 64},
        'kwargs': {'max_out': null, 'min_out': 0}}),
      ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 10, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [5, 5],
         'outker_shape': [5, 5],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 10}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [7, 7],
         'generate': ['random:uniform', {'rseed': 24}],
         'n_filters': 64},
        'kwargs': {'max_out': null, 'min_out': 0}}),
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 1, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 1,
         'threshold': 0.1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [3, 3],
         'generate': ['random:uniform', {'rseed': 32}],
         'n_filters': 256},
        'kwargs': {'max_out': null, 'min_out': null}}),
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 2, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 1}})]]