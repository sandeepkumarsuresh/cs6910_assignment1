"""
This python file contains all the activations functions and its
derivatives.

"""

import numpy as np

def sigmoid_activation(x , derivative = False):

    """
        Compute the sigmoid activation function or its derivative.

        Parameters:
        - x (numpy.ndarray): Input to the sigmoid function.
        - derivative (bool, optional): Flag to compute the derivative of the sigmoid function.
                                    Default is False.

        Returns:
        - numpy.ndarray: Output of the sigmoid function or its derivative.

        If derivative is False, computes the sigmoid activation function element-wise for the input array x.
        If derivative is True, computes the derivative of the sigmoid activation function element-wise for the input array x.

    
    """

    if derivative:
        return np.exp(-x)/np.square(1+np.exp(-x))
    else:
        return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    
def tanh_activation(x,derivative = False):

    """
    Compute the hyperbolic tangent (tanh) activation function or its derivative.

    Parameters:
    - x (numpy.ndarray): Input to the tanh function.
    - derivative (bool, optional): Flag to compute the derivative of the tanh function.
                                   Default is False.

    Returns:
    - numpy.ndarray: Output of the tanh function or its derivative.

    If derivative is False, computes the hyperbolic tangent (tanh) activation function element-wise for the input array x.
    If derivative is True, computes the derivative of the hyperbolic tangent (tanh) activation function element-wise for the input array x.

    
    """
    actvn = np.tanh(x)
    if derivative:
        return 1 - np.square(actvn)
    else:
        return actvn
    

def relu_activation(x , derivative = False):

    """
   Compute the Rectified Linear Unit (ReLU) activation function or its derivative.

    Parameters:
    - x (numpy.ndarray): Input to the ReLU function.
    - derivative (bool, optional): Flag to compute the derivative of the ReLU function.
                                   Default is False.

    Returns:
    - numpy.ndarray: Output of the ReLU function or its derivative.

    If derivative is False, computes the Rectified Linear Unit (ReLU) activation function element-wise for the input array x.
    If derivative is True, computes the derivative of the Rectified Linear Unit (ReLU) activation function element-wise for the input array x.
    

    """

    if derivative:
        return np.heaviside(x,0.0)
    else:
        return np.maximum(0.0,x)

def softmax(x):

    """
       Compute the softmax activation function.

    Parameters:
    - x (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: Output probabilities computed using the softmax function.

    This function computes the softmax activation function element-wise for the input array x.
    Softmax function is applied to convert input values into probabilities, ensuring they sum up to 1.
 
    """

    # returns the output probabilities
    exps = np.exp(x)
    return exps/np.sum(exps)

def compute_cross_entropy_loss(y_truth,y_pred):
    """
    Compute the cross-entropy loss between ground truth and predicted probabilities.

    Parameters:
    - y_truth (numpy.ndarray): Ground truth labels in one-hot encoded format.
    - y_pred (numpy.ndarray): Predicted probabilities.

    Returns:
    - float: Cross-entropy loss computed based on the ground truth and predicted probabilities.

    """
    epsilon = 1e-9 
    max_index = np.argmax(y_truth)
    p_k = y_pred[max_index]
    if p_k <= 0:
        p_k += epsilon
    return -np.log(p_k)
    
def compute_square_error_loss(y_truth,y_pred):
    """
    Compute the mean squared error (MSE) loss between ground truth and predicted values.

    Parameters:
    - y_truth (numpy.ndarray): Ground truth values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - float: Mean squared error loss computed based on the ground truth and predicted values.

    
    """
    return 0.5*np.sum(np.square(y_truth-y_pred))