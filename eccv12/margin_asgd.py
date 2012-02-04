import copy
from itertools import izip

import numpy as np
from numpy import dot

from asgd import NaiveBinaryASGD

class MarginBinaryASGD(NaiveBinaryASGD):

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
                new_train_mean = (np.mean(recent_train_costs)
                        + l2_regularization * np.dot(
                            self.asgd_weights, self.asgd_weights))

                train_means.append(new_train_mean)
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

