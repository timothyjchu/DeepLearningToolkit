
import theano.tensor as T


"""
-- Wrapper class for applying a selected loss function to a provided
set of symbolic matrices (expected to be of the same dimension)
-- Symbolic output is a vector
"""


class LossConstructors():

    def __init__(self, loss_op):

        """
        -- Takes theano tensors and returns algebraic expressions for a
        specified loss vector (specified with loss_op)
        -- In particular, expects matrix arguments, as these loss functions
        are intended for use with mini-batch SGD
        -- Assumes rows are n-dimensional vectors, and returns a vector
        containing the losses between the i-th row in x and the i-th row y
        for each i from 1 to {n_rows}
        """

        self._loss_op = loss_op
        self.loss_set = {'squared_loss': self.squared_loss_constructor,
                         'L1_loss': self.absolute_loss_constructor,
                         'binary_crossentropy': self.binary_crossentropy_constructor}
        self._loss = None

    def squared_loss_constructor(self, x, y):

        # sum of square of the difference vectors
        diff_v = (x - y)**2
        l2_loss = T.sum(diff_v, axis=1)

        return l2_loss

    def absolute_loss_constructor(self, x, y):

        # sum of the absolute value of the difference vectors
        diff_v = x - y
        l1_loss = T.sum(abs(diff_v), axis=1)

        return l1_loss

    def binary_crossentropy_constructor(self, x, y):

        # average log loss over the dimensions of the rows
        return -T.mean(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)

    def loss(self, x, y):

        loss_callable = self.loss_set[self._loss_op]
        self._loss = loss_callable(x, y)

        return self._loss
