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

"""https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
"""

def load_mnist_data_file(path):
	train_data = np.load(path+"train_data.npy")
	val_data = np.load(path+"val_data.npy")
	test_data = np.load(path+"test_data.npy")
	train_labels = np.load(path+"train_labels.npy")
	val_labels = np.load(path+"val_labels.npy")
	test_labels = np.load(path+"test_labels.npy")
	return train_data, train_labels, val_data, val_labels, test_data, test_labels



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
		return self.state

	def derivative(self):
		return 1.0

	def d2(self):
		return np.ones(self.state.shape)



class Sigmoid(Activation):
	""" Implement the sigmoid non-linearity """

	def __init__(self):
		super(Sigmoid, self).__init__()

	def forward(self, x):
		self.state = 1/(1+np.exp(-x))
		return self.state

	def derivative(self):
		return self.state*(1-self.state) 

class Tanh(Activation):
	""" Implement the tanh non-linearity """

	def __init__(self):
		super(Tanh, self).__init__()

	def forward(self, x):
		self.state = np.tanh(x)
		return self.state

	def derivative(self):
		return 1-self.state**2


class ReLU(Activation):
	""" Implement the ReLU non-linearity """

	def __init__(self):
		super(ReLU, self).__init__()

	def forward(self, x):
		self.state = ( x*(x>0), x)
		return self.state[0]

	def derivative(self):
		x = self.state[1]
		return 1*(x>0)


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

def LogSumExp(x):
	mx = np.amax(x,1).reshape(x.shape[0],1)
	shifted = x - mx
	exp = np.exp(shifted)
	sumexp = np.sum(exp,1).reshape(x.shape[0],1)
	result = mx + np.log(sumexp)
	return exp, sumexp, result

class SoftmaxCrossEntropy(Criterion):
	def __init__(self):
		super(SoftmaxCrossEntropy, self).__init__()
		self.sm = None

	def forward(self, x, y):
		self.state = []
		exp, sumexp, logsumexp = LogSumExp(x)
		self.state = exp/sumexp-y
		out = -np.sum(y*(x-logsumexp),1)
		return np.array(out)

	def derivative(self):
		return np.array(self.state)

class BatchNorm(object):
	def __init__(self, fan_in, alpha=0.9):
		self.alpha = alpha
		self.eps = 1e-8
		self.x = None
		self.norm = None
		self.out = None
		self.count = 0

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
		self.x = x
		self.mean = np.average(x,0)
		self.running_mean = (self.running_mean*self.count + self.mean)/(self.count+1)
		self.shiftedX = x-self.mean
		self.var = np.average(np.square(self.shiftedX),0)
		self.running_var = (self.running_var*self.count + self.var)/(self.count+1)
		self.stable_var = self.var+self.eps
		self.sqrt_stable_var = 1/np.sqrt(self.stable_var)
		self.normalizedX = self.shiftedX*self.sqrt_stable_var
		out = self.normalizedX*self.gamma + self.beta
		self.count+=1
		return np.array(out)

	def backward(self, delta): #need gradient.T
		self.delta=delta
		batch_size = delta.shape[0]
		dnormalizedX = delta*self.gamma
		self.dbeta = np.sum(delta,0)
		self.dgamma = np.sum(delta*self.normalizedX,0)
		dvar = np.sum(dnormalizedX*self.shiftedX*-0.5*(self.stable_var**-1.5),0)  #float/pow/** ?
		dmean = np.sum(dnormalizedX*-self.sqrt_stable_var,0) + dvar*np.sum(-2*self.shiftedX,0)/batch_size
		dX = dnormalizedX*self.sqrt_stable_var + dvar*2*self.shiftedX/batch_size + dmean/batch_size
		return np.array(dX)                      #return .T?


def random_normal_weight_init(d0, d1):
	return np.random.randn(d0,d1)

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
		for a in self.activations:
			if type(a)==Identity:
				a.derivative=a.d2
		self.criterion = criterion
		self.lr = lr
		self.momentum = momentum
		# <---------------------

		# Don't change the name of the following class attributes,
		# the autograder will check against these attributes
		self.lsizes = [input_size] + hiddens + [output_size]
		self.W = [weight_init_fn(m,n) for m,n in zip(self.lsizes[:-1],self.lsizes[1:])]
		self.b = [bias_init_fn(m) for m in self.lsizes[1:]]
		self.db = [np.zeros_like(b) for b in self.b]

		# if batch norm, add batch norm parameters
#		if self.bn:
		self.bn_layers = [BatchNorm(self.lsizes[i]) for i in range(self.num_bn_layers)]
		# Feel free to add any other attributes useful to your implementation (input, output, ...)
		self.loss = SoftmaxCrossEntropy()

	def forward(self, x):
		self.state = [[x,]] + [[]]*len(self.W)
		for i in range(len(self.W)):	
			if i<self.num_bn_layers:
				self.state[i][0] = self.bn_layers[i].forward(self.state[i][0])
			self.state[i+1] = ( self.activations[i].forward(self.state[i][0].dot(self.W[i]) + self.b[i]), self.activations[i].derivative() )
		return self.state[-1][0]

	def zero_grads(self):
		self.gradients = [[] for l in self.lsizes]

	def step(self)  :
		self.W = [self.W[i] - self.lr*self.dW[i] for i in range(len(self.W))]
		self.b = [self.b[i] - self.lr*self.db[i] for i in range(len(self.b))]
		for bn in self.bn_layers:
			bn.gamma -= bn.dgamma*bn.alpha
			bn.beta -= bn.dbeta*bn.alpha

	def backward(self, labels):
		loss_value = self.loss.forward(self.state[-1][0], labels)
		dloss = self.loss.derivative()
		self.zero_grads()
		self.gradients[-1] = dloss.T
		self.db[-1]=self.gradients[-1].T*self.state[-1][1]
		g=len(self.gradients)-2
		while g>=0:
			self.gradients[g] = self.W[g].dot(self.db[g].T)
			if g<self.num_bn_layers:
				self.gradients[g] = self.bn_layers[g].backward(self.gradients[g].T).T
			if (g-1)>=0:
				self.db[g-1]=self.gradients[g].T*self.state[g][1]
			g-=1	
		self.dW = [self.state[i][0].T.dot(self.db[i]) for i in range(len(self.b))]
		self.db = [np.average(d,0) for d in self.db]
		self.dW = [d/self.state[-1][-1].shape[0] for d in self.dW]


	def __call__(self, x):
		return self.forward(x)

	def train(self):	
		self.train_mode = True


	def eval(self):
		self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
	for e in range(nepochs):
		pass

