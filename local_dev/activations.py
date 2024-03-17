"""
This python file contains all the activations functions and its
derivatives.

"""

import numpy as np

def sigmoid_activation(x , derivative = False):
    if derivative:
        return np.exp(-x)/np.square(1+np.exp(-x))
    else:
        return 1.0/(1.0 + np.exp(-x)) # 1.0 is for floating for accuracy

def tanh_activation(x,derivative = False):
    actvn = np.tanh(x)
    if derivative:
        return 1 - np.square(actvn)
    else:
        return actvn
def relu_activation(x , derivative = False):
    if derivative:
        return np.heaviside(x,0.0)
    else:
        return np.maximum(0.0,x)

def softmax(x):
    # returns the output probabilities
    exps = np.exp(x)
    return exps/np.sum(exps)