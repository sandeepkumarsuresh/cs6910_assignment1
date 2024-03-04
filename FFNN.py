import numpy as np
from tqdm import tqdm
from activations import *
from sklearn.metrics import accuracy_score
import random
import wandb
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
# np.random.seed(42)
class NN():
    """
    Initializing the number and the size of each hidden layers
    """
    def __init__(self,n_hidden_layers,s_hidden_layer,lr=1e-4,mini_batch_size=64,optimiser = 'sgd',epochs = 3):
        # Initializing the Constructor
        self.n_hidden_layers = n_hidden_layers # Number of hidden layers
        self.s_hidden_layer = s_hidden_layer # Size of the hidden layers
        self.lr = lr # Making the learning rate to some value
        self.params = self.Initialize_Params() # Initalizing the weights and biases
        self.mini_batch_size = mini_batch_size # Mini Batch Size
        self.optimiser = optimiser
        self.epochs = epochs



    def Initialize_Params(self):
        """
        We need to update the weights and bias as a whole.
        This can only be done if the weights and bias are defined like this
        """
        
        intialize_weights_and_bias = {}
        for i in range(1,self.n_hidden_layers):

            intialize_weights_and_bias["W"+str(i)] = np.random.randn(self.s_hidden_layer[i],self.s_hidden_layer[i-1]) *0.1
            intialize_weights_and_bias["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))

            # intialize_weights_and_bias["W"+str(i)] = np.random.randn(self.s_hidden_layer[i],self.s_hidden_layer[i-1]) * np.sqrt(2/(self.s_hidden_layer[i-1] + self.s_hidden_layer[i]))
            # intialize_weights_and_bias["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))
        return intialize_weights_and_bias
    
    def Initialize_gradients_to_zeros(self):
        """
        This function is to initialize all the weights and biases
        to zero.

        Mainly used in MGD and other optimisation function to accumulate
        the gradients
        """
        grads_to_zero = {}
        for i in range(1,self.n_hidden_layers):
            grads_to_zero["W"+str(i)] = np.zeros((self.s_hidden_layer[i],self.s_hidden_layer[i-1]))
            grads_to_zero["B"+str(i)] = np.zeros((self.s_hidden_layer[i],1))
        return grads_to_zero
        


    
    def update_weights_and_bias(self,grads):
        """
        To update the weights and bias after backpropagation
        """
        # for i in range(1,self.n_hidden_layers+2):
        #     self.params['W'+str(i)] -= self.lr * grads['W'+str(i)]
        #     self.params['b'+str(i)] -= self.lr * grads['b'+str(i)]
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

    def compute_loss(self,y_truth,y_pred):
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
            activation_H['h'+str(i)] = sigmoid_activation( activation_A['a'+str(i)]) # y = sigmoid(W_1*x + B_1) 

        activation_A['a'+str(self.n_hidden_layers-1)] = self.params["W"+str(self.n_hidden_layers-1)] @ activation_H['h'+str(self.n_hidden_layers -2)]+ self.params["B"+str(self.n_hidden_layers-1)]
        activation_H['h'+str(self.n_hidden_layers-1)] = softmax(activation_A['a'+str(self.n_hidden_layers-1)])

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

        # print('ypred',y_pred.shape)
        # print('ytruth',y_truth.shape)

        # gradient of L th Layer
        # grad['a'+str(self.n_hidden_layers - 1 )] = (y_pred - y_truth) * y_pred * (1 - y_pred)
        grad['a'+str(self.n_hidden_layers - 1 )] = (y_pred - y_truth) #* y_pred * (1 - y_pred)

        
        # Callculating gradient from L to 1
        for i in range(self.n_hidden_layers-1 , 0 , -1):
            
            
            # Calculating wrt Weights and B

            # grad['W'+str(i)] = np.outer(grad['a'+str(i)],activation_H['h'+str(i-1)])
            grad['W'+str(i)] = np.outer(grad['a'+str(i)],activation_H['h'+str(i-1)])
            # print('a shape',grad['a'+str(i)].shape,'h.shape',activation_H['h'+str(i-1)].shape)
            grad['B'+str(i)] = grad['a'+str(i)]

            # Calculating wrt hidden layers

            grad['h'+ str(i-1)] = np.dot(self.params['W'+str(i)].T,grad['a'+str(i)])
            
            if i > 1:
                tmp = np.exp(-activation_A['a'+str(i-1)])
                # print('tmp',tmp.shape)
                grad['a'+ str(i-1)] = grad['h'+ str(i-1)] * (tmp/((tmp+1)**2))
        
        return grad



    def vanilla_GD(self,train_X,train_Y):

        for i in tqdm(range(self.epochs)):
            for x , y in zip(train_X,train_Y):
                
                # print('x shape' ,x.shape)
            
                activations_A , activations_H = self.forward_pass(x)

                # print('actA shape',activations_A)
                # print('actH shape',activations_H)
                # print('_',_.shape)


                gradients = self.back_propagation(y , activations_A ,activations_H )
            
                self.update_weights_and_bias(gradients)
            # print("gradients",gradients)
                # For gradient Update
                # grads  = self.update_weights_and_bias(gradients)
            # break
            print('gradient_update_after_each_epoch',gradients)
            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)

    
    def sgd(self,train_X ,train_Y):

        """
        Below is the implementation of the sthochastic gradient descent

            estimating the total gradient based on a single data point.
        """
        for i in range(self.epochs) :   
            for x ,y in zip(train_X,train_Y):
                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

                # Updating weight and biases in the same loop

                self.update_weights_and_bias(gradients)
            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)
            wandb.log({"loss":loss,
                       "Accuracy":acc})

    def mgd(self, train_X , train_Y):
        """
        Below is the implementation of the momentum based gradient descent
        
        def do_mgd(self.epochss):
            w,b,eta = -2,-2,1.0
            prev_uw,prev_ub,beta = 0,0,0.9
        
            for i in range(self.epochss):
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
        momentum = 0.9
        history = self.Initialize_gradients_to_zeros()

        for i in range(self.epochs):
            grads_wandb = self.Initialize_gradients_to_zeros()
            lookahead_wandb = self.Initialize_gradients_to_zeros()
            num_points_seen = 0
            for x ,y in zip(train_X,train_Y):
                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

                for key in grads_wandb:
                    grads_wandb[key]+=gradients[key]
                
                num_points_seen +=1

                if(num_points_seen  % self.mini_batch_size == 0):

                        # print(grads_wandb)
                    for key in lookahead_wandb:
                        lookahead_wandb[key] = momentum*history[key] + self.lr*grads_wandb[key]
                    
                    


                    for key in self.params:
                        self.params[key] = self.params[key] - lookahead_wandb[key]
                    
                    # self.update_weights_and_bias(lookahead_wandb)


                    for key in history:
                        history[key] = lookahead_wandb[key]
            
                

            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)
            wandb.log({"loss":loss,
                       "Accuracy":acc})
    def nag(self ,train_X , train_Y):
        """
        This is how the gradient of all the previous updates is added to the current update.

        Update rule for NAG:
            wt+1 = wt - updatet
            While calculating the updatet, We will include the look ahead gradient (∇wlook_ahead).
            updatet = gamma * update_t-1 + η∇wlook_ahead

            ∇wlook_ahead is calculated by:
            wlook_ahead = wt -  gamma*update_t-1

            This look-ahead gradient will be used in our update and will prevent overshooting.
        
        """
        history = self.Initialize_gradients_to_zeros()
        momentum = 0.9
        for i in range(self.epochs):
            grads_wandb = self.Initialize_gradients_to_zeros()
            lookaheads_wandb = self.Initialize_gradients_to_zeros()
            num_points_seen = 0
            # Computing beta*u_{t-1}
            for key in lookaheads_wandb:
                lookaheads_wandb[key] = momentum*history[key]
            # Computing w_t - beta*u_{t-1}
            for key in self.params:
                self.params[key]-= lookaheads_wandb[key]
                        
            for x,y in zip(train_X,train_Y):
                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

                for key in grads_wandb:
                    grads_wandb[key] += gradients[key]
                num_points_seen +=1

                if(num_points_seen  % self.mini_batch_size == 0):

                    # Now moving further in the direction of that gradients
                
                    # Calaculating Lookaheads
                    for key in lookaheads_wandb:
                        lookaheads_wandb[key] = momentum*history[key] + self.lr*grads_wandb[key]
                    
                    # Caluclating the update
                        
                    for key in self.params:
                        self.params[key] -= lookaheads_wandb[key]
                    
                    # updating the lookaheads to global history
                    
                    for key in history:
                        history[key] = lookaheads_wandb[key]
                    
                    grads_wandb = self.Initialize_gradients_to_zeros()

            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)


    
    def rms_prop(self,train_X , train_Y):
        """
        Depends on the intial learning rate 
            parameters that worked are
                epsilon = 1e-8
                lr = 0.1
        """
        epsilon = 1e-8 # This is for mathematical stability 
        beta = 0.9

        history = self.Initialize_gradients_to_zeros()

        for epoch in range(self.epochs):
            grads_wandb = self.Initialize_gradients_to_zeros()

            num_points_seen = 0
            # Computing the gradients
            for x,y in zip(train_X,train_Y):
                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

            
            for key in grads_wandb:
                grads_wandb[key] += gradients[key]

            num_points_seen+=1

            if (num_points_seen % self.mini_batch_size == 0):

                # Computing the history of updates
                for key in history:
                    history[key] = beta* history[key] + ((1- beta)*np.square(grads_wandb[key]))
                
                # Updating the parameters
                for key in self.params:
                    self.params[key] = self.params[key] - (self.lr/(np.sqrt(history[key])+epsilon))*grads_wandb[key]

                grads_wandb = self.Initialize_gradients_to_zeros()

            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)


    def adam(self,train_X , train_Y):
        """
        Combination of momentum based gradient descent and RMS prop
            mt=β1⋅mt+(1-β1)⋅(δwtδL)

            vt=β2⋅vt+(1-β2)⋅(δLδwt)2vt=β2⋅vt+(1-β2)⋅(δwtδL)2

        We will also do a bias correction
            mt = mt / (1 - beta1^t)
            vt = vt / (1 - beta2^t)

        """

         
        beta1 = 0.9
        beta2 =0.999

        moment = self.Initialize_gradients_to_zeros()
        lookahead = self.Initialize_gradients_to_zeros()
        moment_hat = self.Initialize_gradients_to_zeros()
        lookahead_hat = self.Initialize_gradients_to_zeros()
        
        epsilon = 1e-8 # This is for mathematical stability 

        for epoch in range(self.epochs):
            grads_wandb = self.Initialize_gradients_to_zeros()
            
            for x,y in zip(train_X,train_Y):
                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

            for key in grads_wandb:
                grads_wandb[key]+= gradients[key]
        
            for key in self.params:
                moment[key] = beta1*moment[key] + (1-beta1)*grads_wandb[key]
                lookahead[key] = beta2*lookahead[key] + (1-beta2)*grads_wandb[key]

                moment_hat[key] = moment[key]/(1-np.power(beta1,epoch+1))
                lookahead_hat[key] = lookahead[key]/(1-np.power(beta2,epoch+1))

                self.params[key] = self.params[key] - (self.lr/(np.sqrt(lookahead_hat[key])+epsilon))*moment_hat[key]
            
            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)


    def nadam(self,train_X , train_Y):

        """
        Nesterov Adam
        """

        
        beta1 = 0.9
        beta2 =0.999

        moment = self.Initialize_gradients_to_zeros()
        lookahead = self.Initialize_gradients_to_zeros()
        moment_hat = self.Initialize_gradients_to_zeros()
        lookahead_hat = self.Initialize_gradients_to_zeros()
        
        epsilon = 1e-8 # This is for mathematical stability 

        lookahead_history = self.Initialize_gradients_to_zeros()

        num_points_seen = 0

        for epoch in range(self.epochs):
            
            grads_wandb = self.Initialize_gradients_to_zeros()
            

            for x,y in zip(train_X,train_Y):

                activations_A , activations_H = self.forward_pass(x)
                gradients = self.back_propagation(y , activations_A ,activations_H )

            num_points_seen += 1

            if (num_points_seen % self.mini_batch_size) == 0 :
            
                for key in grads_wandb:
                    grads_wandb[key]+= gradients[key]
                
                for key in moment:
                    moment[key] = beta1*moment[key] + ((1-beta1)*grads_wandb[key])
                for key in lookahead[key]:
                    lookahead[key] = beta2*lookahead[key] + ((1-beta2)*np.square(grads_wandb[key]))

                for key in moment_hat:
                    moment_hat[key] = moment[key]/(1 - np.power(beta1,epoch+1))
                for key in lookahead_hat[key]:
                    lookahead_hat[key] = lookahead[key]/(1 - np.power(beta2,epoch+1))
                
                for key in self.params:
                    self.params[key] = self.params[key] - (self.lr/(np.sqrt(lookahead_hat[key])+epsilon))*moment_hat[key]
            
                for key in lookahead_history:
                    lookahead_history[key] = lookahead[key]

                grads_wandb = self.Initialize_gradients_to_zeros()

            acc ,loss = self.evaluate_model_performance(train_X,train_Y)
            print('Accuracy = ', acc ,"Loss : ",loss)

                

    def evaluate_model_performance(self,train_X,train_Y):
        """
        This function is to evaluate accuracy after each timestep.
        """

        pred_labels , truth_labels ,cumullative_loss = [] , [] , []
        # predictions = []
        
        for x,y in tqdm(zip(train_X,train_Y)):
            # doing the forward pass for all test data
            _ , activations_H = self.forward_pass(x)
            
            # output for the last layer
            # pred = np.argmax(activations_H['h'+ str(self.n_hidden_layers -1 )])
            pred = (activations_H['h'+ str(self.n_hidden_layers -1 )])

            y_truth = np.argmax(y.reshape(len(y),1))

            # print('y_truth',y_truth)
            # print('pred',pred)
            
            pred_labels.append(np.argmax(pred))
            truth_labels.append(y_truth)
            # predictions.append(pred == y_truth)
            cumullative_loss.append(self.compute_loss(y,pred))


        
        # print((np.sum(predictions)*100)/len(predictions))
        accuracy = accuracy_score(pred_labels,truth_labels)
        loss = np.sum(cumullative_loss)/len(cumullative_loss)

        return accuracy,loss


    def fit(self,train_X ,train_Y):

        if self.optimiser == 'vanilla_GD':
            self.vanilla_GD(train_X,train_Y)
        elif self.optimiser == 'sgd':
            self.sgd(train_X,train_Y)
        elif self.optimiser == 'mgd':
            self.mgd(train_X,train_Y)
        elif self.optimiser == 'nag':
            self.nag(train_X,train_Y)
        elif self.optimiser == 'rms_prop':
            self.rms_prop(train_X,train_Y)
        elif self.optimiser == 'adam':
            self.adam(train_X,train_Y)
        elif self.optimiser == 'nadam':
            self.nadam(train_X,train_Y)
        else:
            return "Error in fit function. Optimiser Value must be specified"