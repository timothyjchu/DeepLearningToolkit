
import theano.tensor.nnet as nets
import theano.tensor as T

"""
-- Wrapper class for selecting activation functions and applying to 
a given symbolic tensor
-- Activations are not compiled (methods perform symbolic algebra)
-- Offers ReLU, which is not currently available in Theano 0.7.1
"""


# TODO: Add softmax_with_bias implementation

class ActivationConstructors():

    def __init__(self, activation_op):

        self.activation_op = activation_op
        self.activation_set = {"relu": self.relu,
                               "sigmoid": self.theano_sigmoid,
                               "hard_sigmoid": self.theano_hard_sigmoid,
                               "softmax": self.theano_softmax,
                               "softplus": self.theano_softplus,
                               "linear": self.linear}
        self._activation = None

    def relu(self, x):

        return T.switch(x < 0, 0, x)

    def theano_sigmoid(self, x):

        return nets.sigmoid(x)

    def theano_hard_sigmoid(self, x):

        return nets.hard_sigmoid(x)

    def theano_softmax(self, x):

        return nets.softmax(x)

    def theano_softplus(self, x):

        return nets.softplus(x)

    def linear(self, x):

        return x

    def activation(self, x):

        activation_callable = self.activation_set[self.activation_op]
        self._activation = activation_callable(x)

        return self._activation
