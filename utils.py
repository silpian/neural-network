import numpy as np

def tanh_derivative(z):
	return 1.0 - np.tanh(z)**2

def relu(z):
	return z * (z > 0)

def relu_derivative(z):
	return 1.0 * (z > 0)

def logistic(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_derivative(z):
    return logistic(z)*(1-logistic(z))