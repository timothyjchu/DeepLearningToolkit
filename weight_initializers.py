
import theano
import numpy as np


"""
-- Defines options for weight-matrix initializations
-- Currently only supports the recommended uniform weights 
for the tanh activation function
"""


class WeightInits():

    def __init__(self, n_visible, n_hidden, numpy_rng):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.numpy_rng = numpy_rng
        self._weight_init = None

        self.weight_set = {'uniform': self.uniform_weights}

    def uniform_weights(self):
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        tmp_W = np.asarray(
            self.numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (self.n_hidden + self.n_visible)),
                high=4 * np.sqrt(6. / (self.n_hidden + self.n_visible)),
                size=(self.n_visible, self.n_hidden)
            ),
            dtype=theano.config.floatX
        )
        return tmp_W

    def weight_init(self, weight_op):

        weight_callable = self.weight_set[weight_op]
        self._weight_init = weight_callable()

        return self._weight_init