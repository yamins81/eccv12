import time
import numpy as np

import hyperopt
import pyll
from pyll import scope
import pyll_slm

pyll.scope.import_(globals(),
    'pyll_theano_batched_lmap',
    'cifar10_img_classification_task',
    'fit_linear_svm',
    'model_predict',
    'error_rate',
    'np_transpose',
    'flatten_elems',
    'partial',
    'callpipe1',
    'mean_and_std',
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
    'HR': 'hyperopt_result',
    })


def rfilter_size(smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(quniform(smin - q / 2.0 + 1e-5, smax + q / 2.0, q))


def logu_range(lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return loguniform(np.log(lower), np.log(upper))


class Cifar10Bandit1(pyll_slm.HPBandit):
    def __init__(self, n_train=40000, n_valid=10000, n_test=10000,
            nfilt_ubounds=[64],
            batchsize=10,
            l2_regularization=1e-7):

        lnorm0 = partial(lnorm,
                ker_size=HP('n0_size', rfilter_size(2, 6)),
                remove_mean=HP('n0_remove_mean', one_of(0, 1)),
                stretch=HP('n0_stretch', logu_range(.1/3, 10.*3)),
                threshold=HP('n0_thresh', logu_range(.1/3, 10.*3)))
        #print lnorm0

        pipeline = [lnorm0]
        for ii, nfu in enumerate(nfilt_ubounds):
            ll = ii + 1
            fbcorr1 = partial(fbcorr,
                    ker_size=HP('f%i_size' % ll, rfilter_size(2, 6)),
                    n_filters=s_int(
                        HP('f%i_nfilt' % ll, qloguniform(
                            np.log(16 / 2.0) + 1e-5,
                            np.log(nfu),
                            q=16))),
                    generate=('random:uniform',
                        {'rseed': HP('f%i_seed' % ll, choice(range(10, 15)))}))

            lpool1 = partial(lpool,
                    ker_size=HP('p%i_size' % ll, rfilter_size(2, 5)),
                    order=HP('p%i_order' % ll, loguniform(np.log(1), np.log(10))),
                    stride=HP('p%i_stride' % ll, 1))

            lnorm1 = partial(lnorm,
                    ker_size = HP('n%i_size' % ll, rfilter_size(2, 5)),
                    remove_mean=HP('n%i_nomean' % ll, one_of(0, 1)),
                    stretch=HP('n%i_stretch' % ll, logu_range(.1/3, 10.*3)),
                    threshold=HP('n%i_thresh' % ll, logu_range(.1/3, 10.*3)))

            pipeline.extend([fbcorr1, lpool1, lnorm1])

        #print pipeline
        pipeline = pyll.as_apply(pipeline)

        assert n_train + n_valid <= 50000
        assert n_test <= 10000

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

        xmean, xstd = mean_and_std(
                flatten_elems(screen_features[:n_train]),
                remove_std0=True)
        xmean = HR('xmean', xmean)
        xstd = HR('xstd', xstd)

        trn_xy=(
            (flatten_elems(screen_features[:n_train]) - xmean) / xstd,
            all_labels[:n_train])

        val_xy = (
            (flatten_elems(screen_features[n_train:n_train + n_valid])
                - xmean) / xstd,
            all_labels[n_train:n_train + n_valid])

        tst_xy = (
            (flatten_elems(test_features[:]) - xmean) / xstd,
            all_labels[50000:50000 + n_test])

        # TODO: choose l2_regularization by optimization
        svm = fit_linear_svm(trn_xy, l2_regularization=l2_regularization)

        outputs = []
        for name, xy in ('trn', trn_xy), ('val', val_xy), ('tst', tst_xy):
            erate = error_rate(model_predict(svm, xy[0]), xy[1])
            if name == 'val':
                erate = HR('loss', erate)
            outputs.append(HR(name + "_erate", erate))
        pyll_slm.HPBandit.__init__(self, pyll.as_apply(outputs))


def Cifar10Bandit2():
    return Cifar10Bandit1(nfilt_ubounds=[64, 128])


def Cifar10Bandit3():
    return Cifar10Bandit1(nfilt_ubounds=[64, 128, 256])


def Cifar10Bandit1Small():
    return Cifar10Bandit1(nfilt_ubounds=[64],
                          n_train=10000, n_valid=10000, n_test=1000)


def Cifar10Bandit2Small():
    return Cifar10Bandit1(nfilt_ubounds=[64, 128],
                          n_train=10000, n_valid=10000, n_test=1000)


def Cifar10Bandit3Small():
    return Cifar10Bandit1(nfilt_ubounds=[64, 128, 256],
                          n_train=10000, n_valid=10000, n_test=1000)

