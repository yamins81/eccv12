import numpy as np
from eccv12.utils import linear_kernel

def test_linear_kernel():
    def foo(xshp, yshp, use_theano):
        X = np.random.randn(*xshp)
        Y = np.random.randn(*yshp)
        A = linear_kernel(X, Y, use_theano=use_theano)
        B = np.dot(X, Y.T)

        assert A.shape == B.shape
        assert np.allclose(A, B)

        # there is special code for symmetric case in linear_kernel
        A = linear_kernel(X, X, use_theano=use_theano)
        B = np.dot(X, X.T)

        assert A.shape == B.shape
        assert np.allclose(A, B)

    foo((10, 5), (10, 5), True)
    foo((10, 5), (10, 5), False)

    foo((100, 5), (100, 5), True)
    foo((100, 5), (100, 5), False)

    foo((200, 5), (100, 5), True)
    foo((200, 5), (100, 5), False)
