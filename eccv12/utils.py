import numpy as np
import Image


class ImgLoaderResizer(object):
    """ Load 250x250 greyscale images, return normalized 200x200 float32 ones.
    """
    def __init__(self, shape=None, ndim=None, dtype='float32', normalize=True,
                 crop=None):
        assert len(shape) == 2
        shape = tuple(shape)
        assert len(crop) == 4
        crop = tuple(crop)
        l, t, r, b = crop
        assert 0 <= l < r <= 250
        assert 0 <= t < b <= 250 
        self._crop = crop   
        assert dtype == 'float32'
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.normalize = normalize

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        assert im.size == (250, 250)
        if self._crop != (0, 0, 250, 250):
            im = im.crop(self._crop)
        l, t, r, b = self._crop
        assert im.size == (r - l, b - t)
        if im.size != self._shape:
            im = im.resize(self._shape, Image.ANTIALIAS)
        assert im.size == self._shape
        rval = np.asarray(im, 'float32')
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
        assert rval.shape == self._shape
        return rval

