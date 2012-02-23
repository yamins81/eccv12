import copy
import numpy as np

import pyll
choice = pyll.scope.choice
uniform = pyll.scope.uniform
one_of = pyll.scope.one_of    # -- just one_of(a, b) means choice([a, b])
uniform = pyll.scope.uniform
quniform = pyll.scope.quniform
loguniform = pyll.scope.loguniform
qloguniform = pyll.scope.qloguniform

norm_shape_choice = choice([(3,3),(5,5),(7,7),(9,9)])

lnorm = {'kwargs':{'inker_shape' : norm_shape_choice,
         'outker_shape' : norm_shape_choice,
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

activ =  {'kwargs': {'min_out' : choice([None, 0]),
                     'max_out' : choice([1, None])}}

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
                                'max_out': None}}),
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
                                 'max_out': None}}),
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
                                 'max_out': None}}),
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
        'kwargs': {'max_out': None, 'min_out': 0}}),
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
        'kwargs': {'max_out': None, 'min_out': 0}}),
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
        'kwargs': {'max_out': None, 'min_out': None}}),
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 2, 'stride': 2}}),
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 1}})]]


crop_choice = choice([[0, 0, 250, 250],
                      [25, 25, 175, 175],
                      [88, 63, 163, 188]])

l3_params = {'slm': [[('lnorm', lnorm)],
                     [('fbcorr', filter1),
                      ('lpool', lpool_sub2),
                     ('lnorm', lnorm)],
                     [('fbcorr', filter2),
                      ('lpool', lpool_sub2),
                      ('lnorm', lnorm)],
                     [('fbcorr', filter3),
                      ('lpool', lpool_sub2),
                      ('lnorm', lnorm)],
                    ],
             'preproc': {'global_normalize': 0,
                         'crop': crop_choice,
                         'size': [200, 200]}} 

l2_params = {'slm': [[('lnorm', lnorm)],
                     [('fbcorr', filter1),
                      ('lpool', lpool_sub2),
                     ('lnorm', lnorm)],
                     [('fbcorr', filter2),
                      ('lpool', lpool_sub2),
                      ('lnorm', lnorm)]
                    ],
             'preproc': {'global_normalize': 0,
                         'crop': crop_choice,
                         'size': [100, 100]}}

main_params = choice([l3_params, l2_params])

test_params = {'slm': [[('lnorm', lnorm)]],
                          'preproc': {'global_normalize': 0,
                                      'crop': crop_choice,
                                      'size': [20, 20]}}


def main_param_func(nf):
    v3 = l3_params
    v3['slm'][-1][0][1]['initialize']['n_filters'] = nf
    v2 = l2_params
    v2['slm'][-1][0][1]['initialize']['n_filters'] = nf
    return choice([v2, v3])


def pyll_param_func(nf=None):
    """
    Return a template for lfw.MainBandit that describes a hyperopt-friendly
    description of the search space.

    The goal here is to approximately match the FG11 sampling distribution,
    while smoothing out the search space by using quantized ranges and
    continuous variables where appropriate.
    """

    def rfilter_size(smin, smax, q=1):
        """Return an integer size from smin to smax inclusive with equal prob
        """
        return quniform(smin - q + 1e-5, smax, q)

    def cont_pt1_10():
        """Return a continuous replacement for one_of(.1, 1, 10)"""
        s = np.sqrt(10)
        return loguniform(np.log(.1 / s), np.log(10 * s))


    # N.B. that each layer is constructed with distinct objects
    # we don't want to use the same norm_shape_size at every layer.
    def lnorm():
        size = rfilter_size(2, 10)

        return ('lnorm', {'kwargs':
                    {'inker_shape' : (size, size),
                     'outker_shape' : (size, size),
                     'remove_mean' : one_of(0, 1),
                     'stretch' : cont_pt1_10(),
                     'threshold' : cont_pt1_10(),
                 }})

    def lpool(stride=2):
        # XXX test that bincount on a big sample of these guys comes out about right
        size = rfilter_size(2, 10)
        # XXX are fractional powers ok here?
        return ('lpool', {
            'kwargs': {
                'ker_shape': (size, size),
                'order': loguniform(np.log(1), np.log(10)),
                'stride': stride,
                }})

    def fbcorr(max_filters, iseed, n_filters=None):
        if n_filters is None:
            n_filters = qloguniform(
                    np.log(1e-5),
                    np.log(max_filters),
                    q=16)
        size = rfilter_size(2, 10)
        return ('fbcorr', {
            'initialize': {
                'filter_shape': (size, size),
                'n_filters': n_filters,
                'generate': (
                    'random:uniform',
                    {'rseed': choice(range(iseed, iseed + 5))}
                    ),
                },
            'kwargs': {},
            })

    preproc_l3 = dict(
            global_normalize=0,
            crop=one_of(
                [0, 0, 250, 250],
                [25, 25, 175, 175],
                [88, 63, 163, 188]),
            size=[200, 200],
            )

    # -- N.B. shallow copy keeps the same crop object
    preproc_l2 = dict(preproc_l3, size=[100, 100])

    l0 = [lnorm()]
    l1 = [fbcorr(64 + 32, 1), lpool(), lnorm()]
    l2 = [fbcorr(128 + 64, 11), lpool(), lnorm()]
    l3 = [fbcorr(256 + 128, 111), lpool(), lnorm()]
    l2_clone = pyll.clone(pyll.as_apply(l2))

    # -- the re-use of l0 and l1 will make both slm models accumulate evidence
    #    for what works and what doesn't
    # -- the cloning of l2 will do the opposite -- what works for l2 in l2_slm
    #    will not be connected to what's good for l2 in l3_slm.
    l3_slm = dict(slm=[l0, l1, l2, l3], preproc=preproc_l3)
    l2_slm = dict(slm=[l0, l1, l2_clone], preproc=preproc_l2)

    return one_of(l3_slm, l2_slm)

