import numpy as np
from tqdm import tqdm
from activations_and_lossfunctions import *
from sklearn.metrics import accuracy_score
import random




np.random.seed(42)

class NN():

    """
    Neural Network model for classification tasks.

    Parameters:
    - size_of_network (int): Number of layers in the network including input and output layers.
    - n_hidden_layers (int): Number of hidden layers.
    - s_hidden_layer (list): List containing the number of neurons in each hidden layer.
    - lr (float, optional): Learning rate for gradient descent optimization. Default is 1e-4.
    - mini_batch_size (int, optional): Size of mini-batches for stochastic gradient descent. Default is 64.
    - optimiser (str, optional): Optimization algorithm to use. Default is 'sgd' (Stochastic Gradient Descent).
    - epochs (int, optional): Number of epochs to train the network. Default is 3.
    - weight_init_params (str, optional): Parameter for weight initialization. Default is 'Xavier'.
    - activation (str, optional): Activation function to use in hidden layers. Default is 'sigmoid'.
    - weight_decay (float, optional): Weight decay (L2 regularization) parameter. Default is 0.1.
    - loss_function (str, optional): Loss function to use. Default is 'cre' (Cross-Entropy Loss).
    - regularization (bool, optional): Whether to apply regularization. Default is False.
    - momentum (float, optional): Momentum parameter for optimization algorithms. Default is 0.9.
    - beta (float, optional): Beta parameter for optimization algorithms. Default is 0.9.
    - beta1 (float, optional): Beta1 parameter for optimization algorithms. Default is 0.9.
    - beta2 (float, optional): Beta2 parameter for optimization algorithms. Default is 0.999.
    - epsilon (float, optional): Epsilon parameter for optimization algorithms. Default is 1e-8.

    Methods:
    - Initialize_Params(): Initialize weights and biases for the network.
    - Initialize_gradients_to_zeros(): Initialize gradients to zeros for optimization.
    - update_weights_and_bias(grads): Update weights and biases after backpropagation.
    - forward_pass(x): Perform forward pass through the network.
    - back_propagation(y, activation_A, activation_H): Perform backpropagation to compute gradients.
    - compute_acc_and_loss(train_X, train_Y): Compute accuracy and loss on training data.
    - fit(train_X, train_Y, val_X, val_Y): Train the neural network using specified optimization algorithm.
    """
   

   

    def __init__(self,size_of_network,n_hidden_layers,s_hidden_layer,lr=1e-4,mini_batch_size=64,optimiser = 'sgd',epochs = 3,weight_init_params = 'Xavier',activation = 'sigmoid',weight_decay = 0.1,loss_function = 'cre',regularization = False,momentum=0.9,beta=0.9
                 ,beta1 =0.9,beta2 =0.999,epsilon=1e-8):
        # Initializing the Constructor
        self.weight_decay = weight_decay
        self.activation = activation
        self.weight_init_params = weight_init_params
        self.size_of_network = size_of_network
        self.n_hidden_layers = n_hidden_layers # Number of hidden layers
        self.s_hidden_layer = s_hidden_layer # Size of the hidden layers
        self.lr = lr # Making the learning rate to some value
        self.params = self.Initialize_Params() # Initalizing the weights and biases
        self.mini_batch_size = mini_batch_size # Mini Batch Size
        self.optimiser = optimiser
        self.epochs = epochs
        self.loss_function = loss_function
        self.regularization = regularization
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def Initialize_Params(self):

        """
        Initialize weights and biases for the neural network.

        Returns:
        - dict: Dictionary containing initialized weights and biases for each layer.

        """
        intialize_weights_and_bias = {}

        for i in range(1,self.size_of_network):
            if self.weight_init_params == 'random':
                intialize_weights_and_bias["W"+str(i)] = np.random.randn(self.s_hidden_layer[i],self.s_hidden_layer[i-1])*0.01
            elif self.weight_init_params == 'Xavier':
                intialize_weights_and_bias["W"+str(i)] = np.random.randn(self.s_hidden_layer[i],self.s_hidden_layer[i-1]) * np.sqrt(2/(self.s_hidden_layer[i-1] + self.s_hidden_layer[i]))
            intialize_weights_and_bias["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))
        
        return intialize_weights_and_bias
    
    def Initialize_gradients_to_zeros(self):
        """
        Initialize weights and biases for the neural network to zero.

        Returns:
        - dict: Dictionary containing initialized weights and biases to zero.

        """
        grads_to_zero = {}
        for i in range(1,self.size_of_network):
            grads_to_zero["W"+str(i)] = np.zeros((self.s_hidden_layer[i],self.s_hidden_layer[i-1]))
            grads_to_zero["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))
        return grads_to_zero
        


    
    def update_weights_and_bias(self,grads):
        """
        Update the weights and biases after backpropagation.

        Parameters:
        - grads (dict): Dictionary containing parameters updated

        """
        # for i in range(1,self.n_hidden_layers+2):
        #     self.params['W'+str(i)] -= self.lr * grads['W'+str(i)]
        #     self.params['b'+str(i)] -= self.lr * grads['b'+str(i)]
        for key in self.params:
            self.params[key] -= (self.lr * grads[key]) - (self.lr*self.weight_decay*grads[key])





        
    def forward_pass(self,x):
        """
        Perform the forward pass through the neural network.

        Parameters:
        - x (numpy.ndarray): Input data for the forward pass.

        Returns:
        - tuple: Tuple containing dictionaries of activation values for each layer (activation_A) 
                and activation outputs for each layer (activation_H).
        """
        activation_A , activation_H = {} , {}
        activation_H['h'+str(0)] = x.reshape(x.shape[0],1)
        # print(activation_H['h0'].shape)
        for i in range(1 , self.size_of_network-1):

            activation_A['a'+str(i)] = (self.params['W'+str(i)] @ activation_H['h'+str(i-1)] ) + self.params["B"+str(i)]  # o = W_1*x + B_1

            if self.activation == 'sigmoid':
                activation_H['h'+str(i)] = sigmoid_activation( activation_A['a'+str(i)]) # y = sigmoid(W_1*x + B_1) 
            elif self.activation == 'tanh':
                activation_H['h'+str(i)] = tanh_activation( activation_A['a'+str(i)]) 
            else:
                activation_H['h'+str(i)] = relu_activation( activation_A['a'+str(i)]) 

        activation_A['a'+str(self.size_of_network-1)] = self.params["W"+str(self.size_of_network-1)] @ activation_H['h'+str(self.size_of_network -2)]+ self.params["B"+str(self.size_of_network-1)]
        activation_H['h'+str(self.size_of_network-1)] = softmax(activation_A['a'+str(self.size_of_network-1)])

        # y_hat = activation_H['h'+str(self.n_hidden_layers-1)] # Final output pred

        return activation_A , activation_H 
    
    def back_propagation(self,y,activation_A ,activation_H ) :

        # print("####################### Inside Backprop ####################### ")

        """
        Perform the back-propagation through the neural network.

        Parameters:
        - x (numpy.ndarray): Input data for the forward pass.

        Returns:
        - grad (dict): Dictionary containing gradients during the backpropagation

        """
        grad = {}
        # Here we are calculating the squared error loss
        y_pred = activation_H['h'+str(self.size_of_network-1)] # Final ouput during forward pass
        y_truth = y.reshape(-1 , 1 ) # reshape to column 1 with any number of rows


        if self.loss_function == 'mse':
            grad['a'+str(self.size_of_network - 1 )] = (y_pred - y_truth) * y_pred * (1 - y_pred)
        else:
            grad['a'+str(self.size_of_network - 1 )] = (y_pred - y_truth) #* y_pred * (1 - y_pred)

        
        # Callculating gradient from L to 1
        for i in range(self.size_of_network-1 , 0 , -1):
            
            
            # Calculating wrt Weights and B

            # grad['W'+str(i)] = np.outer(grad['a'+str(i)],activation_H['h'+str(i-1)])
            grad['W'+str(i)] = np.outer(grad['a'+str(i)],activation_H['h'+str(i-1)])
            # print('a shape',grad['a'+str(i)].shape,'h.shape',activation_H['h'+str(i-1)].shape)
            grad['B'+str(i)] = grad['a'+str(i)]

            # Calculating wrt hidden layers

            grad['h'+ str(i-1)] = np.dot(self.params['W'+str(i)].T,grad['a'+str(i)])
            
            if i > 1:
                if self.activation == 'sigmoid':
                    tmp = np.exp(-activation_A['a'+str(i-1)])
                    # print('tmp',tmp.shape)
                    grad['a'+ str(i-1)] = grad['h'+ str(i-1)] * (tmp/((tmp+1)**2))
                elif self.activation == 'relu':
                    grad['a'+str(i-1)] = grad['h'+str(i-1)] * relu_activation(activation_A['a'+str(i-1)],derivative=True)
                elif self.activation == 'tanh':
                    grad['a'+str(i-1)] = grad['h'+str(i-1)] * tanh_activation(activation_A['a'+str(i-1)],derivative=True)

        return grad




                

    def compute_acc_and_loss(self,train_X,train_Y):
        """
        Compute accuracy and loss on the data.

        Parameters:
        - train_X (numpy.ndarray): Input features 
        - train_Y (numpy.ndarray): Target labels 

        Returns:
        - tuple: Tuple containing accuracy and loss values.
        """

        pred_labels , truth_labels ,cumullative_loss = [] , [] , []
        # predictions = []
        
        for x,y in tqdm(zip(train_X,train_Y)):
            # doing the forward pass for all test data
            _ , activations_H = self.forward_pass(x)
            
            # output for the last layer
            # pred = np.argmax(activations_H['h'+ str(self.n_hidden_layers -1 )])
            pred = (activations_H['h'+ str(self.size_of_network -1 )])

            y_truth = np.argmax(y.reshape(len(y),1))

            # print('y_truth',y_truth)
            # print('pred',pred)
            
            pred_labels.append(np.argmax(pred))
            truth_labels.append(y_truth)
            # predictions.append(pred == y_truth)
            if self.loss_function == 'cre':
                cumullative_loss.append(compute_cross_entropy_loss(y,pred))
            else:
                cumullative_loss.append(compute_square_error_loss(y,pred))



        
        # print((np.sum(predictions)*100)/len(predictions))
        accuracy = accuracy_score(pred_labels,truth_labels)
        loss = np.sum(cumullative_loss)/len(cumullative_loss)

        return accuracy,loss


    def fit(self,train_X ,train_Y,val_X,val_Y):

        """
        Train the neural network using the specified optimization algorithm.

        Parameters:
        - train_X (numpy.ndarray): Input features for the training data.
        - train_Y (numpy.ndarray): Target labels for the training data.
        - val_X (numpy.ndarray): Input features for the validation data.
        - val_Y (numpy.ndarray): Target labels for the validation data.

        Returns:
        - None
        """

        if self.optimiser == 'vanilla_GD':
            self.vanilla_GD(train_X,train_Y,val_X,val_Y)
        elif self.optimiser == 'sgd':
            self.sgd(train_X,train_Y,val_X,val_Y)
        elif self.optimiser == 'mgd':
            self.mgd(train_X,train_Y,val_X,val_Y)
        elif self.optimiser == 'nag':
            self.nag(train_X,train_Y,val_X,val_Y)
        elif self.optimiser == 'rms_prop':
            self.rms_prop(train_X,train_Y,val_X,val_Y)
        elif self.optimiser == 'adam':
            self.adam(train_X,train_Y,val_X,val_Y)
        elif self.optimiser == 'nadam':
            self.nadam(train_X,train_Y,val_X,val_Y)
        else:
            return "Error in fit function. Optimiser Value must be specified"