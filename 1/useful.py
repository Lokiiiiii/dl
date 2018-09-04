import numpy as np

def LogSumExp(vector):
"""
https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
"""
	mx = np.amax(vector)
	shifted = vector - mx
	result = mx + np.log(np.sum(np.exp(shifted)))
	return result
