"""
This python file contains all the activations functions and its
derivatives.

"""

import numpy as np

def sigmoid_activation(x , derivative = False):
    if derivative:
        return np.exp(-x)/np.square(1+np.exp(-x))
    else:
        return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    
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

def compute_cross_entropy_loss(y_truth,y_pred):
    """
    Since we are doing classification , we need to compute the 
    cross entropy loss
    
    For the case of one -hot encodings
    
    L(t,p) = -tklog(pk) = -log(pk)

    """
    epsilon = 1e-9 
    max_index = np.argmax(y_truth)
    p_k = y_pred[max_index]
    if p_k <= 0:
        p_k += epsilon
    return -np.log(p_k)
    
def compute_square_error_loss(y_truth,y_pred):
    return 0.5*np.sum(np.square(y_truth-y_pred))