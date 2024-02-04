import numpy as np
from tqdm import tqdm

"""
Author : Sandeep Kumar Suresh
            EE23S059
"""

"""
Here the 
    Input is n - dim vector
    Output is 10 classes

    The network has L-1 hidden layers
        I/p = 0th Layer
        O/P = Lth Layer

        Wi = nxn and B = n for layers 1 to L-1

        W_l = kxn and b_l = k for the last hidden layer


"""

## To Do
# clear Confusions regarding the parameter update and the number of hidden
# layer neurons

class NN():
    """
    Initializing the number and the size of each hidden layers
    """
    def __init__(self,n_hidden_layers,s_hidden_layer,lr=0.001):
        self.n_hidden_layers = n_hidden_layers # Number of hidden layers
        self.s_hidden_layer = s_hidden_layer # Size of the hidden layers
        self.lr = lr # Making the learning rate to some value
        self.params = self.Initialize_Params() # Initalizing the weights and biases
    def sigmoid_activation(self,x):
        return 1.0/(1.0 + np.exp(-x)) # 1.0 is for floating for accuracy
    
    def softmax(self,x):
        # returns the output probabilities
        exps = np.exp(x)
        return exps/np.sum(exps)

    def Initialize_Params(self):
        """
        We need to update the weights and bias as a whole.
        This can only be done if the weights and bias are defined like this
        """
        intialize_weights_and_bias = {}
        for i in range(1,self.n_hidden_layers):
            intialize_weights_and_bias["W"+str(i)] = np.random.randn(self.s_hidden_layer[i],self.s_hidden_layer[i-1])
            intialize_weights_and_bias["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))
        return intialize_weights_and_bias
    
    def update_weights_and_bias(self,grads):
        """
        To update the weights and bias after backpropagation
        """
        # for i in range(1,self.n_hidden_layers+2):
        #     self.params['W'+str(i)] -= self.lr * grads['W'+str(i)]
        #     self.params['b'+str(i)] -= self.lr * grads['b'+str(i)]
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

        
    def forward_pass(self,x):
        """
        For Layer 1 to L-1 we find the the a_k and h_k

        Final Layer (output) we find the activation and then the softmax

        """
        activation_A , activation_H = {} , {}
        activation_H['h0'] = x.reshape(x.shape[0],1)
        # print(activation_H['h0'].shape)
        for i in range(1 , self.n_hidden_layers-1):
            activation_A['a'+str(i)] = (self.params['W'+str(i)] @ activation_H['h'+str(i-1)] ) + self.params["B"+str(i)]  # o = W_1*x + B_1
            activation_H['h'+str(i)] = self.sigmoid_activation( activation_A['a'+str(i)]) # y = sigmoid(W_1*x + B_1) 

        activation_A['a'+str(self.n_hidden_layers-1)] = self.params["W"+str(self.n_hidden_layers-1)] @ activation_H['h'+str(self.n_hidden_layers -2)]+ self.params["B"+str(self.n_hidden_layers-1)]
        activation_H['h'+str(self.n_hidden_layers-1)] = self.softmax(activation_A['a'+str(self.n_hidden_layers-1)])

        # y_hat = activation_H['h'+str(self.n_hidden_layers-1)] # Final output pred

        return activation_A , activation_H 
    def back_propagation(self,y,activation_A ,activation_H ) :

        # print("####################### Inside Backprop ####################### ")

        """
    
        During Back Propagation:
            - Gradient wrt output layer
            - gradient wrt to hidden layer (a and h) from L to 1
            - gradient wrt weights and bias from L to 1


        """
        grad = {}
        # Here we are calculating the squared error loss
        y_pred = activation_H['h'+str(self.n_hidden_layers-1)] # Final ouput during forward pass
        y_truth = y.reshape(-1 , 1 ) # reshape to column 1 with any number of rows

        print('ypred',y_pred.shape)
        # print('ytruth',y_truth.shape)

        # gradient of L th Layer
        grad['a'+str(self.n_hidden_layers - 1 )] = (y_pred - y_truth) * y_pred * (1 - y_pred)
        
        # Callculating gradient from L to 1
        for i in range(self.n_hidden_layers-1 , 0 , -1):
            
            
            # Calculating wrt Weights and B

            # grad['W'+str(i)] = np.outer(grad['a'+str(i)],activation_H['h'+str(i-1)])
            grad['W'+str(i)] = np.dot(grad['a'+str(i)],activation_H['h'+str(i-1)].T)
            # print('a shape',grad['a'+str(i)].shape,'h.shape',activation_H['h'+str(i-1)].shape)
            grad['B'+str(i)] = grad['a'+str(i)]

            # Calculating wrt hidden layers

            grad['h'+ str(i-1)] = np.dot(self.params['W'+str(i)].T,grad['a'+str(i)])
            
            if i > 1:
                tmp = np.exp(-activation_A['a'+str(i-1)])
                # print('tmp',tmp.shape)
                grad['a'+ str(i-1)] = grad['h'+ str(i-1)] * (tmp/((tmp+1)**2))
        
        return grad

    
    def sgd(self,train_X ,train_Y):

        """
        Below is the implementation of the sthochastic gradient descent

            estimating the total gradient based on a single data point.
        """
        max_epoch = 3
        for i in range(max_epoch) :   
            for x ,y in zip(train_X,train_Y):
                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

                # Updating weight and biases in the same loop

                self.update_weights_and_bias(gradients)


    def mgd(self, train_X , train_Y):
        """
        Below is the implementation of the momentum based gradient descent
        
        def do_mgd(max_epochs):
            w,b,eta = -2,-2,1.0
            prev_uw,prev_ub,beta = 0,0,0.9
        
            for i in range(max_epochs):
                dw,db = 0,0        
                for x,y in zip(X,Y):
                    dw += grad_w(w,b,x,y)
                    db += grad_b(w,b,x,y)
                    
                uw = beta*prev_uw+eta*dw
                ub = beta*prev_ub+eta*db
                w = w - vw
                b = b - vb
                prev_uw = uw
                prev_ub = ub
            
        """
        pass

                

    def train(self,train_X,train_y):

        for i in tqdm(range(3)):
            for x , y in zip(train_X,train_y):
                
                # print('x shape' ,x.shape)
            
                activations_A , activations_H = self.forward_pass(x)

                # print('actA shape',activations_A)
                # print('actH shape',activations_H)
                # print('_',_.shape)


                gradients = self.back_propagation(y , activations_A ,activations_H )
                # print("gradients",gradients)
                # For gradient Update
                # grads  = self.update_weights_and_bias(gradients)
            break
            # print('gradient_update_after_each_epoch',grads)




        