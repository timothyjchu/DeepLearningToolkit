
"""
-- Wrapper class for applying a selected regularizer on either
the weight matrix W or the jacobians (coming)
-- All methods expect symbolic or shared variables
"""

import theano.tensor as T


# TODO: Add Jacobian regularizers (contractive autoencoder)


class Regularizers():

    def __init__(self, reg_op):

        """
        -- Takes a theano tensor and returns an algebraic expression for the
        regularization
        -- In particular, expects a matrix argument
        -- Assumes rows are n-dimensional vectors
        """

        self._reg_op = reg_op
        self.reg_set = {'weight_decay_L1': self.weight_decay_L1,
                         'jacobian_L1': self.jacobian_L1,
                         'weight_decay_L2': self.weight_decay_L2,
                         'jacobian_L2': self.jacobian_L2}
        self._reg = None

    def weight_decay_L1(self, x):

        # mean of sum of absolute value of weight matrix
        l1_reg = T.mean(T.sum(abs(x), axis=1))

        return l1_reg

    def weight_decay_L2(self, x):

        # sum of square of the difference vectors
        x_2 = (x ) **2
        l2_reg = T.mean(T.sum(x_2, axis=1))

        return l2_reg

    def jacobian_L1(self, x):

        raise NotImplementedError('Jacobian regularizers not available yet.')

    def jacobian_L2(self, x):

        raise NotImplementedError('Jacobian regularizers not available yet.')

    def regularizer(self, x):

        reg_callable = self.reg_set[self._reg_op]
        self._reg = reg_callable(x)

        return self._reg
