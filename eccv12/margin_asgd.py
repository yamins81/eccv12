import copy
from itertools import izip

import numpy as np
from numpy import dot

from asgd import NaiveBinaryASGD

class MarginBinaryASGD(NaiveBinaryASGD):

    def fit_converged(self):
        train_means = self.train_means
        if len(train_means) > 2:
            midpt = len(train_means) // 2
            thresh = (1 - self.fit_tolerance) * train_means[midpt] - 5e-3
            return train_means[-1] > thresh
        return False

    def partial_fit(self, X, y, previous_decisions):

        assert np.all(y**2 == 1)

        assert len(X) == len(y) == len(previous_decisions)

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        train_means = self.train_means
        recent_train_costs = self.recent_train_costs
        min_observations = self.min_observations

        for obs, label, pdec in izip(X, y, previous_decisions):

            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias + pdec)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            if margin < 1:

                sgd_weights += sgd_step_size * label * obs
                sgd_bias += sgd_step_size * label
                recent_train_costs.append(1 - float(margin))
            else:
                recent_train_costs.append(0)

            # -- update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # 4.1 update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

            if len(recent_train_costs) == self.fit_n_partial:
                train_means.append(np.mean(recent_train_costs)
                        + l2_regularization * np.dot(
                            self.asgd_weights, self.asgd_weights))
                self.recent_train_costs = recent_train_costs = []

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        return self

    def fit(self, X, y, previous_decisions):

        assert X.ndim == 2
        assert y.ndim == 1

        if previous_decisions is None:
             previous_decisions = np.zeros(len(y), dtype=X.dtype)

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_points_remaining = self.max_observations - self.n_observations

        while n_points_remaining > 0:

            # -- every iteration will train from n_partial observations and
            # then check for convergence
            fit_n_partial = min(n_points_remaining, self.fit_n_partial)

            idx = self.rstate.permutation(n_points)
            Xb = X[idx[:fit_n_partial]]
            yb = y[idx[:fit_n_partial]]
            db = previous_decisions[idx[:fit_n_partial]]
            self.partial_fit(Xb, yb, db)

            if self.feedback:
                raise NotImplementedError(
                    'partial_fit logic requires memory to be distinct')
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

            if (self.n_observations >= self.min_observations
                    and self.fit_converged()):
                break

            n_points_remaining -= len(Xb)

        return self


import copy
import numpy as np
from scipy import optimize

DEFAULT_INITIAL_RANGE = 0.25, 0.5
DEFAULT_MAX_EXAMPLES = 1000
DEFAULT_TOLERANCE = 0.5
DEFAULT_BRENT_OUTPUT = False


def find_sgd_step_size0(
    model, partial_fit_args,
    initial_range=DEFAULT_INITIAL_RANGE,
    tolerance=DEFAULT_TOLERANCE, brent_output=DEFAULT_BRENT_OUTPUT):
    """Use a Brent line search to find the best step size

    Parameters
    ----------
    model: BinaryASGD
        Instance of a BinaryASGD

    partial_fit_args - tuple of arguments for model.partial_fit.
        This tuple must start with X, y, ...

    initial_range: tuple of float
        Initial range for the sgd_step_size0 search (low, high)

    max_iterations:
        Maximum number of interations

    Returns
    -------
    best_sgd_step_size0: float
        Optimal sgd_step_size0 given `X` and `y`.
    """
    # -- stupid scipy calls some sizes twice!?
    _cache = {}

    def eval_size0(log2_size0):
        try:
            return _cache[log2_size0]
        except KeyError:
            pass
        other = copy.deepcopy(model)
        current_step_size = 2 ** log2_size0
        other.sgd_step_size0 = current_step_size
        other.sgd_step_size = current_step_size
        other.partial_fit(*partial_fit_args)
        # Hack: asgd is lower variance than sgd, but it's tuned to work
        # well asymptotically, not after just a few examples
        weights = .5 * (other.asgd_weights + other.sgd_weights)
        bias = .5 * (other.asgd_bias + other.sgd_bias)

        X, y = partial_fit_args[:2]

        margin = y * (np.dot(X, weights) + bias)
        l2_cost = other.l2_regularization * (weights ** 2).sum()
        rval = np.maximum(0, 1 - margin).mean() + l2_cost
        _cache[log2_size0] = rval
        return rval

    best_sgd_step_size0 = optimize.brent(
        eval_size0, brack=np.log2(initial_range), tol=tolerance)

    return best_sgd_step_size0


def binary_fit(
    model, fit_args,
    max_examples=DEFAULT_MAX_EXAMPLES,
    **find_sgd_step_size0_kwargs):
    """Returns a model with automatically-selected sgd_step_size0

    Parameters
    ----------
    model: BinaryASGD
        Instance of the model to be fitted.

    fit_args - tuple of args to model.fit
        This method assumes they are all length-of-dataset ndarrays.

    max_examples: int
        Maximum number of examples to use from `X` and `y` to find an
        estimate of the best sgd_step_size0. N.B. That the entirety of X and y
        is used for the final fit() call after the best step size has been found.

    Returns
    -------
    model: BinaryASGD
        Instances of the model, fitted with an estimate of the best
        sgd_step_size0
    """

    if max_examples <= 0:
        # negative number would actually work for indexing and do something
        # weird.
        raise ValueError('max_examples must be positive')

    # randomly choose up to max_examples uniformly without replacement from
    # across the whole set of training data.
    idxs = model.rstate.permutation(len(fit_args[0]))[:max_examples]

    # Find the best learning rate for that subset
    best = find_sgd_step_size0(
        model, [a[idxs] for a in fit_args], **find_sgd_step_size0_kwargs)

    # Heuristic: take the best stepsize according to the first max_examples,
    # and go half that fast for the full run.
    best_estimate = 2. ** (best - 1.0)
    model.sgd_step_size0 = best_estimate
    model.sgd_step_size = best_estimate
    model.fit(*fit_args)

    return model

