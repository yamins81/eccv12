"""
A library file for the design approach of cifar10.py

It includes the slm components as well as a few other things that should
migrate upstream.

"""
import copy

import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv

import pyll
import hyperopt

from skdata import larray
from skdata.cifar10 import CIFAR10


class InvalidDescription(Exception):
    """Model description was invalid"""


def alloc_filterbank(n_filters, height, width, channels, dtype,
        method_name, method_kwargs, normalize=True, SLMP=None):
    """
    Generate the same weights as are generated by pythor3
    """
    if height != width:
        raise ValueError('filters must be square')
    if channels is None:
        filter_shape = [n_filters, height, width]
    else:
        filter_shape = [n_filters, height, width, channels]

    if method_name == 'random:uniform':
        rseed = method_kwargs.get('rseed', None)
        np.random.seed(rseed)
        fb_data = np.random.uniform(size=filter_shape)
    elif method_name == 'gabor2d:grid':
        # allocate a filterbank spanning a grid of frequencies, phases,
        # orientations
        raise NotImplementedError()

    else:
        raise ValueError(
            "method to generate filterbank '%s' not understood"
            % method_name)

    # normalize each filter in the bank if needed
    if normalize:
        # TODO: vectorize these computations, do all at once.
        for fidx, filt in enumerate(fb_data):
            # normalization here means zero-mean, unit-L2norm
            filt -= filt.mean()
            filt_norm = np.sqrt((filt * filt).sum())
            assert filt_norm != 0
            filt /= filt_norm
            fb_data[fidx] = filt

    return fb_data.astype(dtype)


@pyll.scope.define_info(o_len=2)
def boxconv((x, x_shp), kershp, channels=False):
    """
    channels: sum over channels (T/F)
    """
    kershp = tuple(kershp)
    if channels:
        rshp = (   x_shp[0],
                    1,
                    x_shp[2] - kershp[0] + 1,
                    x_shp[3] - kershp[1] + 1)
        kerns = np.ones((1, x_shp[1]) + kershp, dtype=x.dtype)
    else:
        rshp = (   x_shp[0],
                    x_shp[1],
                    x_shp[2] - kershp[0] + 1,
                    x_shp[3] - kershp[1] + 1)
        kerns = np.ones((1, 1) + kershp, dtype=x.dtype)
        x_shp = (x_shp[0]*x_shp[1], 1, x_shp[2], x_shp[3])
        x = x.reshape(x_shp)
    try:
        rval = tensor.reshape(
                conv.conv2d(x,
                    kerns,
                    image_shape=x_shp,
                    filter_shape=kerns.shape,
                    border_mode='valid'),
                rshp)
    except Exception, e:
        if "Bad size for the output shape" in str(e):
            raise InvalidDescription()
        else:
            raise
    return rval, rshp


@pyll.scope.define_info(o_len=2)
def slm_fbcorr((x, x_shp), n_filters, ker_size,
        min_out=None,
        max_out=None,
        stride=1,
        mode='valid',
        generate=None):
    assert x.dtype == 'float32'
    filter_shape = (ker_size, ker_size)
    # Reference implementation:
    # ../pythor3/pythor3/operation/fbcorr_/plugins/scipy_naive/scipy_naive.py
    if stride != 1:
        raise NotImplementedError('stride is not used in reference impl.')
    kerns = alloc_filterbank(n_filters=n_filters,
            height=filter_shape[0],
            width=filter_shape[1],
            channels=x_shp[1],
            dtype=x.dtype,
            method_name=generate[0],
            method_kwargs=generate[1])
    kerns = kerns.transpose(0, 3, 1, 2).copy()[:,:,::-1,::-1]
    x = conv.conv2d(
            x,
            kerns,
            image_shape=x_shp,
            filter_shape=kerns.shape,
            border_mode=mode)
    if mode == 'valid':
        x_shp = (x_shp[0], n_filters,
                x_shp[2] - filter_shape[0] + 1,
                x_shp[3] - filter_shape[1] + 1)
    elif mode == 'full':
        x_shp = (x_shp[0], n_filters,
                x_shp[2] + filter_shape[0] - 1,
                x_shp[3] + filter_shape[1] - 1)
    else:
        raise NotImplementedError('fbcorr mode', mode)

    if min_out is None and max_out is None:
        return x, x_shp
    elif min_out is None:
        return tensor.minimum(x, max_out), x_shp
    elif max_out is None:
        return tensor.maximum(x, min_out), x_shp
    else:
        return tensor.clip(x, min_out, max_out), x_shp


@pyll.scope.define_info(o_len=2)
def slm_lpool((x, x_shp),
        ker_size=3,
        order=1,
        stride=1,
        mode='valid'):
    assert x.dtype == 'float32'
    order=float(order)

    ker_shape = (ker_size, ker_size)
    if hasattr(order, '__iter__'):
        o1 = (order == 1).all()
        o2 = (order == order.astype(np.int)).all()
    else:
        o1 = order == 1
        o2 = (order == int(order))

    if o1:
        r, r_shp = boxconv((x, x_shp), ker_shape)
    elif o2:
        r, r_shp = boxconv((x ** order, x_shp), ker_shape)
        r = tensor.maximum(r, 0) ** (1.0 / order)
    else:
        r, r_shp = boxconv((abs(x) ** order, x_shp), ker_shape)
        r = tensor.maximum(r, 0) ** (1.0 / order)

    if stride > 1:
        r = r[:, :, ::stride, ::stride]
        # intdiv is tricky... so just use numpy
        r_shp = np.empty(r_shp)[:, :, ::stride, ::stride].shape
    return r, r_shp


@pyll.scope.define_info(o_len=2)
def slm_lnorm((x, x_shp),
        ker_size=3,
        remove_mean= False,
        div_method='euclidean',
        threshold=0.0,
        stretch=1.0,
        mode='valid',
        EPSILON=1e-4,
        ):
    # Reference implementation:
    # ../pythor3/pythor3/operation/lnorm_/plugins/scipy_naive/scipy_naive.py
    assert x.dtype == 'float32'
    inker_shape=(ker_size, ker_size)
    outker_shape=(ker_size, ker_size)  # (3, 3)
    if mode != 'valid':
        raise NotImplementedError('lnorm requires mode=valid', mode)

    threshold = float(threshold)
    stretch = float(stretch)

    if outker_shape == inker_shape:
        size = np.asarray(x_shp[1] * inker_shape[0] * inker_shape[1],
                dtype=x.dtype)
        ssq, ssqshp = boxconv((x ** 2, x_shp), inker_shape,
                channels=True)
        xs = inker_shape[0] // 2
        ys = inker_shape[1] // 2
        # --local contrast normalization in regions that are not symmetric
        #   about the pixel being normalized feels weird, but we're
        #   allowing it here.
        xs_inc = (inker_shape[0] + 1) % 2
        ys_inc = (inker_shape[1] + 1) % 2
        if div_method == 'euclidean':
            if remove_mean:
                arr_sum, _shp = boxconv((x, x_shp), inker_shape,
                        channels=True)
                arr_num = (x[:, :, xs-xs_inc:-xs, ys-ys_inc:-ys]
                        - arr_sum / size)
                arr_div = EPSILON + tensor.sqrt(
                        tensor.maximum(0,
                            ssq - (arr_sum ** 2) / size))
            else:
                arr_num = x[:, :, xs-xs_inc:-xs, ys-ys_inc:-ys]
                arr_div = EPSILON + tensor.sqrt(ssq)
        else:
            raise NotImplementedError('div_method', div_method)
    else:
        raise NotImplementedError('outker_shape != inker_shape',outker_shape, inker_shape)
    if (hasattr(stretch, '__iter__') and (stretch != 1).any()) or stretch != 1:
        arr_num = arr_num * stretch
        arr_div = arr_div * stretch
    arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

    r = arr_num / arr_div
    r_shp = x_shp[0], x_shp[1], ssqshp[2], ssqshp[3]
    return r, r_shp


@pyll.scope.define
def pyll_theano_batched_lmap(pipeline, seq, batchsize,
        _debug_call_counts=None, print_progress=False):
    """
    This function returns a skdata.larray.lmap object whose function
    is defined by a theano expression.

    The theano expression will be built and compiled specifically for the
    dimensions of the given `seq`. Therefore, in_rows, and out_rows should
    actually be a *pyll* graph, that evaluates to a theano graph.
    """

    in_shp = (batchsize,) + seq.shape[1:]
    batch = np.zeros(in_shp, dtype='float32')
    s_ibatch = theano.shared(batch)
    s_xi = (s_ibatch * 1).type() # get a TensorType w right ndim
    s_N = s_xi.shape[0]
    s_X = theano.tensor.set_subtensor(s_ibatch[:s_N], s_xi)
    #print 'PIPELINE', pipeline
    thing = pipeline((s_X, in_shp))
    #print 'THING'
    #print thing
    #print '==='
    s_obatch, oshp = pyll.rec_eval(thing)
    assert oshp[0] == batchsize

    # Compile a function that takes a variable number of elements in,
    # returns the same number of processed elements out,
    # but does all internal computations using a fixed number of elements,
    # because convolutions are fastest when they're hard-coded to a certain
    # size.
    fn = theano.function([s_xi], s_obatch[:s_N],
            updates={
                s_ibatch: s_X, # this allows the inc_subtensor to be in-place
                })

    def fn_1(x):
        if _debug_call_counts:
            _debug_call_counts['fn_1'] += 1
        return fn(x[None, :, :, :])[0]

    attrs = {
            'shape': oshp[1:],
            'ndim': len(oshp) -1,
            'dtype': s_obatch.dtype }
    def rval_getattr(attr, objs):
        # -- objs don't matter to the structure of the return value
        try:
            return attrs[attr]
        except KeyError:
            raise AttributeError(attr)

    fn_1.rval_getattr = rval_getattr

    def f_map(X):
        if _debug_call_counts:
            _debug_call_counts['f_map'] += 1
        rval = np.zeros((len(X),) + oshp[1:], dtype=s_obatch.dtype)
        offset = 0
        while offset < len(X):
            if print_progress:
                print 'pyll_theano_batched_lmap.f_map', offset, len(X)
            xi = X[offset: offset + batchsize]
            rval[offset:offset + len(xi)] = fn(xi)
            offset += len(xi)
        return rval

    return larray.lmap(fn_1, seq, f_map=f_map)


@pyll.scope.define
def linsvm_train_test(features, labels, n_train, n_test,
        C=1.0, allow_inplace=False, normalize_cols=True, shuffle=False,
        **kwargs):
    return {'result': 'awesome'}


@pyll.scope.define_info(o_len=2)
def cifar10_img_classification_task(dtype='float32'):
    imgs, labels = CIFAR10().img_classification_task(dtype='float32')
    return imgs, labels


@pyll.scope.define
def hyperopt_set_loss(dct, key):
    #TODO: move this logic to general nested-key-setting pyll fn
    #N.B. this function modifies dct in-place, but doesn't change any of the
    #     existing keys... so it's probably safe not to copy dct.
    keys = key.split('.')
    obj = dct
    for key in keys:
        obj = obj[key]
    dct.setdefault('loss', obj)
    if obj != dct['loss']:
        raise KeyError('loss already set')
    return dct


@pyll.scope.define
def np_transpose(obj, arg):
    return obj.transpose(*arg)


@pyll.scope.define
def hyperopt_param(label, obj):
    return obj


class HPBandit(hyperopt.Bandit):
    """ Create a hyperopt.Bandit from a pyll program that has been annotated
    with hyperopt_param (HP) expressions.
    """

    def __init__(self, result_expr):
        # -- need a template: which is a dictionary with all the HP nodes
        template = {}
        for node in pyll.dfs(result_expr):
            if node.name == 'hyperopt_param':
                template[node.arg['label'].obj] = node.arg['obj']
        hyperopt.Bandit.__init__(self, template)
        self.result_expr = result_expr

    def evaluate(self, config, ctrl):
        # -- config is a dict with values for all the HP nodes
        memo = {}
        for node in pyll.dfs(self.result_expr):
            if node.name == 'hyperopt_param':
                label = node.arg['label'].obj
                memo[node] = config[label]
        assert len(memo) == len(config)
        return pyll.rec_eval(self.result_expr, memo=memo)

