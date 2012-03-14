import time
import numpy as np

import hyperopt
import pyll
from pyll import scope
import pyll_slm

pyll.scope.import_(globals(),
    'pyll_theano_batched_lmap',
    'cifar10_img_classification_task',
    'linsvm_train_test',
    'hyperopt_set_loss',
    'np_transpose',
    'partial',
    'callpipe1',
    # /begin distributions... that hyperopt can tune
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'one_of',
    'choice',
    # /end distributions...
    **{
    # NEW NAME:  ORIG NAME
    'lnorm': 'slm_lnorm',
    'lpool': 'slm_lpool',
    'fbcorr': 'slm_fbcorr',
    's_int': 'int',
    'HP': 'hyperopt_param',
    })


def rfilter_size(smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(quniform(smin - q / 2.0 + 1e-5, smax + q / 2.0, q))


def logu_range(lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return loguniform(np.log(lower), np.log(upper))


class Cifar10Bandit1(pyll_slm.HPBandit):
    def __init__(self, namebase, n_train, n_valid, n_test,
            use_l3='no',  # can be yes, no, maybe
            batchsize=10,
            C=1.0):

        lnorm0 = partial(lnorm,
                ker_size=HP('n0_size', rfilter_size(2, 6)),
                remove_mean=HP('n0_remove_mean', one_of(0, 1)),
                stretch=HP('n0_stretch', logu_range(.1/3, 10.*3)),
                threshold=HP('n0_thresh', logu_range(.1/3, 10.*3)))

        fbcorr1 = partial(fbcorr,
                ker_size=HP('f1_size', rfilter_size(2, 6)),
                n_filters=s_int(
                    HP('f1_nfilt', qloguniform(
                        np.log(16 / 2.0) + 1e-5,
                        np.log(64),
                        q=16))),
                generate=('random:uniform',
                    {'rseed': HP('f1_seed', choice(range(10, 15)))}))

        lpool1 = partial(lpool,
                ker_size=HP('p1_size', rfilter_size(2, 5)),
                order=HP('p1_order', loguniform(np.log(1), np.log(10))),
                stride=HP('p1_stride', 1))

        lnorm1 = partial(lnorm,
                ker_size = HP('n1_size', rfilter_size(2, 5)),
                remove_mean=HP('n1_nomean', one_of(0, 1)),
                stretch=HP('n1_stretch', logu_range(.1/3, 10.*3)),
                threshold=HP('n1_thresh', logu_range(.1/3, 10.*3)))

        pipeline = [lnorm0, fbcorr1, lpool1, lnorm1]

        assert n_train + n_valid < 50000
        assert n_test < 10000

        # -- map cifar10 through the pipeline
        all_imgs, all_labels = cifar10_img_classification_task()
        all_imgs = np_transpose(all_imgs, (0, 3, 1, 2))

        screen_features = pyll_theano_batched_lmap(
                partial(callpipe1, pipeline),
                all_imgs[:n_train + n_valid],
                batchsize=batchsize)

        test_features = pyll_theano_batched_lmap(
                partial(callpipe1, pipeline),
                all_imgs[50000:50000 + n_test],
                batchsize=batchsize)

        self.result = linsvm_train_test(
                train=(screen_features[:n_train], all_labels[:n_train]),
                test_sets = dict(
                    valid=(
                        screen_features[n_train:n_train + n_valid],
                        all_labels[n_train:n_train + n_valid]
                        ),
                    test=(
                        test_features,
                        all_labels[50000:50000 + n_test],
                        ),
                    ),
                C=C,
                normalize_cols=True,
                allow_inplace=True,
                report={
                    'col_affine': 0,
                    'svm': 0,
                    'train.error_rate': 1,
                    'train.fit_time': 1,
                    'test.valid.error_rate': 1,
                    'test.valid.accuracy': 1,
                    'test.test.error_rate': 1,
                    'test.test.accuracy': 1,
                    'train.sgd_step_size0': 1,
                    })

        self.result_w_loss = hyperopt_set_loss(self.result, 'test.valid.error_rate')

        pyll_slm.HPBandit.__init__(self, self.result_w_loss)


