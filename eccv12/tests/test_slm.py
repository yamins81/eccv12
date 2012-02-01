import cPickle
import string
import numpy as np
import unittest
import scipy.io

import skdata.lfw

from eccv12.plugins import (slm_memmap,
                            pairs_memmap,
                            verification_pairs,
                            get_images,
                            pairs_cleanup,
                            delete_memmap)


class BestVsSavedKernels(unittest.TestCase): 
    namebase = 'test'
    test_pair_inds = [0, 1]
    comparison = 'mult'
    #XXX: get this file from the S3 account (use boto?)
    matfile = 'lfw_view1.csv.kernel.ht1_1_l3__150fd767e9d5d6822e414b6ae20d7da6ce9469fa_gray_gray.mul.mat'
    desc  = [[('lnorm',{'kwargs':{'inker_shape': (9, 9),
                         'outker_shape': (9, 9),
                         'stretch':10,
                         'threshold': 1}})], 
             [('fbcorr', {'initialize': {'filter_shape': (3, 3),
                                         'n_filters': 64,
                                         'generate': ('random:uniform', 
                                                      {'rseed': 42})},
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
                                                      {'rseed': 42})},
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
                                                     {'rseed': 42})},
                          'kwargs': {'min_out': 0,
                                     'max_out': None}}),
               ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                     'order': 10,
                                     'stride': 2}}),
               ('lnorm', {'kwargs': {'inker_shape': (3, 3),
                                     'outker_shape': (3, 3),
                                     'stretch': 10,
                                     'threshold': 1}})]]

    def test(self):
                                            
        all_test_pair_inds = self.test_pair_inds
        
        x = scipy.io.loadmat(self.matfile)
        x_train_fnames_0 = map(string.strip, map(str, x['train_fnames'][::2]))
        x_train_fnames_1 = map(string.strip, map(str, x['train_fnames'][1::2]))
        x_train_labels = map(int, x['train_labels'])
        
        
        dataset = skdata.lfw.Aligned()
        train_fnames_0, train_fnames_1, train_labels = dataset.raw_verification_task(split='DevTrain')
        train_fnames_0 = ['/'.join(_f.split('/')[-2:]) for _f in map(str, train_fnames_0)]
        train_fnames_1 = ['/'.join(_f.split('/')[-2:]) for _f in map(str, train_fnames_1)]
        fnames, _l = dataset.raw_classification_task()
        fnames = map(str, ['/'.join(_f.split('/')[-2:]) for _f in fnames])
        pairs = verification_pairs(split='DevTrain')
        
        for _ind in all_test_pair_inds:
            assert train_fnames_0[_ind] == x_train_fnames_0[_ind]
            assert train_fnames_1[_ind] == x_train_fnames_1[_ind]
            assert fnames[pairs[0][_ind]] == train_fnames_0[_ind] 
            assert fnames[pairs[1][_ind]] == train_fnames_1[_ind] 
            assert pairs[2][_ind] == train_labels[_ind] == x_train_labels[_ind]
            
        pairs = (pairs[0][all_test_pair_inds],
                 pairs[1][all_test_pair_inds],
                 pairs[2][all_test_pair_inds])
        
        x_kern = x['kernel_traintrain'][all_test_pair_inds][:, all_test_pair_inds]
        
        namebase = self.namebase
        image_features = slm_memmap(self.desc,
                                    get_images('float32'),
                                    namebase + '_img_feat')
        #print np.asarray(image_features[:4])
        pf_cache, matches = pairs_memmap(pairs,
                                              image_features,
                                              comparison_name=self.comparison,
                                              name=namebase + '_pairs_DevTrain')

        
        pair_features = np.asarray(pf_cache)
        delete_memmap(image_features)
        pairs_cleanup((pf_cache, None))
        
        assert (pairs[2] == matches).all()
        
        #XXX: normalize the pair_features here in some way? do we have to compute
        #all the features?
        kern = np.dot(pair_features, pair_features.T)
        
        absdiff = abs(kern - x_kern)
        absdiffmax = absdiff.max()
    
        if absdiffmax > .001:
            print 'kern', kern
            print 'x_kern', x_kern
            assert 0, ('too much error: %s' % absdiffmax)
            

        
                
        
        