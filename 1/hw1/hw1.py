"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

>>> activation = Identity()
>>> activation(3)
3
>>> activation.forward(3)
3
"""

import numpy as np
import os


def LogSumExp(vector):
"""
https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
"""
	mx = np.amax(vector)
	shifted = vector - mx
	exp = np.exp(shifted)
	sumexp = np.sum(exp)
	result = mx + np.log(sumexp)
	return exp, sumexp, result

def load_mnist_data_file(path):
    raise NotImplementedError


class Activation(object):
    """ Interface for activation functions (non-linearities).

        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ Identity function (already implemented).
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Implement the sigmoid non-linearity """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Tanh(Activation):
    """ Implement the tanh non-linearity """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class ReLU(Activation):
    """ Implement the ReLU non-linearity """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


# CRITERION


class Criterion(object):
    """ Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
    	batch_result = []
    	self.memory = []
    	for b in range(x.shape[0]):
    		exp, sumexp, logsumexp = LogSumExp(x[b])
    		negsumproduct = -np.sum(x[b]*y[b])
    		t2 = np.sum(y[b]*logsumexp)
    		result_per_example = t2 + negsumproduct
    		batch_result.append(result_per_example)
    		self.memory.append(-y[b] + (exp/sumexp))
    	return(np.array(batch_result))

    def derivative(self):
    	return(np.array(self.memory))


class BatchNorm(object):
    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        raise NotImplemented

    def backward(self, delta):
        raise NotImplemented


def random_normal_weight_init(d0, d1):
	return np.random.randn(do,d1)

def zeros_bias_init(d):
	return np.zeros((1,d))

class MLP(object):
    """ A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        raise NotImplemented

    def zero_grads(self):
        raise NotImplemented

    def step(self):
        raise NotImplemented

    def backward(self, labels):
        raise NotImplemented

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    raise NotImplemented


