import numpy as np

"""
Here the 
    Input is 784 
    Output is 10 classes

All the other parameters are initialised by the user/hardcoded

"""

class NN():
    """
    Initializing the number and the size of each hidden layers
    """
    def __init__(self,n_hidden_layers,s_hidden_layer):
        self.n_hidden_layers = n_hidden_layers # Number of hidden layers
        self.s_hidden_layer = s_hidden_layer # Size of the hidden layers
    
    def sigmoid_activation(self,x):
        return 1.0/(1.0 + np.exp(-x)) # 1.0 is for floating for accuracy
    
    def softmax(self,x):
        # returns the output probabilities
        exps = np.exp(x)
        return exps/np.sum(exps)

    def forward_pass(self,x):
        """
        Here we have to initialize weights for each layer 
        """
        intialize_weights_and_bias = {}
        for i in range(1,self.n_hidden_layers):
            intialize_weights_and_bias["W"+str(i)] = np.random.randn(self.s_hidden_layer[i],self.s_hidden_layer[i-1])
            intialize_weights_and_bias["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))

        """ Now we have the weights and biases , we need todo forward passing 
         for layer 1 to N-1 .

         And for the last N^th layer , we need to apply a activation function
        """
        activation_A , activation_H = {} , {}
        for i in range(1 , self.n_hidden_layers-1):
            activation_A['a'+str(i)] = activation_H['h'+str(i-1)] @ intialize_weights_and_bias['W'+str(i)] + intialize_weights_and_bias["B"+str(i)]  # o = W_1*x + B_1
            activation_H['h'+str(i)] = self.sigmoid_activation( activation_A['a'+str(i)]) # y = sigmoid(W_1*x + B_1) 

        activation_A['a'+str(self.n_hidden_layers-1)] = activation_H['h'+str(self.n_hidden_layers -2)]@intialize_weights_and_bias["W"+str(self.n_hidden_layers-1)] + intialize_weights_and_bias["B"+str(self.n_hidden_layers-1)]
        activation_H['h'+str(self.n_hidden_layers-1)] = self.softmax(activation_A['a'+str(self.n_hidden_layers-1)])

        return activation_A , activation_H  , intialize_weights_and_bias 

    def back_propagation(self,y,activation_A ,activation_H):

        """
        First we need to compute the output layer gradient and then backpropagat
        through each layer to the initial input layer to find the error 
        """
        grad = {}
        # Here we are calculating the squared error loss
        y_pred = self.softmax(activation_H['h'+str(self.n_hidden_layers - 1)])
        y_truth = y.reshape(-1 , 1 ) # reshape to column 1 with any number of rows
        
        grad['a'+str(self.n_hidden_layers - 1 )] = (y_pred - y_truth) * y_pred * (1 - y_pred)

        for i in range(self.n_hidden_layers-1 , 0 , -1):
            
            # Calculating wrt parameters
            grad['W'+str(i)] = np.outer(grad['a'+str(i)],activation_H['h'+str(i-1)])
            grad['B'+str(i)] = grad['a'+str(i)]

            # Calculating wrt hidden layers

            grad['h'+ str(i-1)] = np.dot(self.intialize_weights_and_bias['W'+str(i)],activation_H['h',str(i-1)])
            
            if i > 1:
                tmp = np.exp(-activation_A['a'+str(i-1)])
                grad['a'+ str(i-1)] = grad['h'+ str(i-1)] * (tmp/((tmp+1)**2))
        
        return grad





        