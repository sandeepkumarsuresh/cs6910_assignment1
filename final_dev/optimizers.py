import wandb
import numpy as np
class Optimizers:
    """
    Contains all the Optimization Algorithms

    Methods:
    - vanilla_GD(NN,train_X,train_Y,val_X,val_Y): Vannilla Gradient Descent
    - sgd(NN,train_X,train_Y,val_X,val_Y): Stochastic Gradient Descent
    - mgd(NN,train_X,train_Y,val_X,val_Y): Momentum Gradient Descent
    - nag(NN,train_X,train_Y,val_X,val_Y): Nesterov Accelerated Gradient Descent
    - adam(NN,train_X,train_Y,val_X,val_Y): Perform backpropagation to compute gradients.
    - nadam(NN,train_X,train_Y,val_X,val_Y): Compute accuracy and loss on training data.
    - rms_prop(NN,train_X,train_Y,val_X,val_Y): Train the neural network using specified optimization algorithm.

    
    """


    def vanilla_GD(self,NN,train_X,train_Y,val_X,val_Y):

        """
            Perform training using Vanilla Gradient Descent optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """

        for i in range(NN.epochs):
            for x , y in zip(train_X,train_Y):

                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )
            
                NN.update_weights_and_bias(gradients)

            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})

    
    def sgd(self,NN,train_X ,train_Y,val_X,val_Y):

        """
            Perform training using SGD optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """
        for i in range(NN.epochs) :   
            for x ,y in zip(train_X,train_Y):
                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )

                # Updating weight and biases in the same loop

                NN.update_weights_and_bias(gradients)
            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})

    def mgd(self,NN, train_X , train_Y,val_X,val_Y):
        """
            Perform training using Momentum Gradient Descent optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """
        momentum = NN.momentum #0.9
        history = NN.Initialize_gradients_to_zeros()

        for i in range(NN.epochs):
            grads_wandb = NN.Initialize_gradients_to_zeros()
            lookahead_wandb = NN.Initialize_gradients_to_zeros()
            num_points_seen = 0
            for x ,y in zip(train_X,train_Y):
                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )

                for key in grads_wandb:
                    grads_wandb[key]+=gradients[key]
                
                num_points_seen +=1

                if(num_points_seen  % NN.mini_batch_size == 0):

                        # print(grads_wandb)
                    for key in lookahead_wandb:
                        lookahead_wandb[key] = momentum*history[key] + NN.lr*grads_wandb[key]
                    
                    


                    for key in NN.params:
                        NN.params[key] -=  lookahead_wandb[key] - (NN.lr*NN.alpha*grads_wandb[key])
                    
                    # NN.update_weights_and_bias(lookahead_wandb)


                    for key in history:
                        history[key] = lookahead_wandb[key]
            
                

            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})


    def nag(self,NN ,train_X , train_Y , val_X, val_Y):
        """
            Perform training using Nesterov Accelerated Gradient Descent optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """
        history = NN.Initialize_gradients_to_zeros()
        momentum = 0.9
        for i in range(NN.epochs):
            grads_wandb = NN.Initialize_gradients_to_zeros()
            lookaheads_wandb = NN.Initialize_gradients_to_zeros()
            num_points_seen = 0
            # Computing beta*u_{t-1}
            for key in lookaheads_wandb:
                lookaheads_wandb[key] = momentum*history[key]
            # Computing w_t - beta*u_{t-1}
            for key in NN.params:
                NN.params[key]-= lookaheads_wandb[key]
                        
            for x,y in zip(train_X,train_Y):
                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )

                for key in grads_wandb:
                    grads_wandb[key] += gradients[key]
                num_points_seen +=1

                if(num_points_seen  % NN.mini_batch_size == 0):

                    # Now moving further in the direction of that gradients
                
                    # Calaculating Lookaheads
                    for key in lookaheads_wandb:
                        lookaheads_wandb[key] = momentum*history[key] + NN.lr*grads_wandb[key]
                    
                    # Caluclating the update
                        
                    for key in NN.params:
                        NN.params[key] -= lookaheads_wandb[key]
                    
                    # updating the lookaheads to global history
                    
                    for key in history:
                        history[key] = lookaheads_wandb[key]
                    
                    grads_wandb = NN.Initialize_gradients_to_zeros()

            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})

    
    def rms_prop(self,NN,train_X , train_Y , val_X ,val_Y):
        """
            Perform training using RMS propagation Gradient Descent optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """
        epsilon = NN.epsilon #1e-8 # This is for mathematical stability 
        beta = NN.beta #0.9

        history = NN.Initialize_gradients_to_zeros()

        for epoch in range(NN.epochs):
            grads_wandb = NN.Initialize_gradients_to_zeros()

            num_points_seen = 0
            # Computing the gradients
            for x,y in zip(train_X,train_Y):
                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )

            
                for key in grads_wandb:
                    grads_wandb[key] += gradients[key]

                num_points_seen+=1

                if (num_points_seen % NN.mini_batch_size == 0):

                    # Computing the history of updates
                    for key in history:
                        history[key] = beta* history[key] + ((1- beta)*np.square(grads_wandb[key]))
                    
                    # Updating the parameters
                    for key in NN.params:
                        NN.params[key] = NN.params[key] - (NN.lr/(np.sqrt(history[key])+epsilon))*grads_wandb[key]

                    grads_wandb = NN.Initialize_gradients_to_zeros()

            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})


    def adam(self,NN,train_X , train_Y , val_X , val_Y):
        """
            Perform training using ADAM Gradient Descent optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """


         
        beta1 = NN.beta1 #0.9
        beta2 =NN.beta2 #0.999

        m = NN.Initialize_gradients_to_zeros()
        v = NN.Initialize_gradients_to_zeros()
        m_hat = NN.Initialize_gradients_to_zeros()
        v_hat = NN.Initialize_gradients_to_zeros()
        
        epsilon = NN.epsilon #1e-8 # This is for mathematical stability 


        for epoch in range(NN.epochs):
            grads_wandb = NN.Initialize_gradients_to_zeros()

            num_points_seen = 0

            
            for x,y in zip(train_X,train_Y):
                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )

                for key in grads_wandb:
                    grads_wandb[key]+= gradients[key]
                
                num_points_seen = num_points_seen + 1

                if(num_points_seen % NN.mini_batch_size == 0):
            
                    for key in NN.params:
                        m[key] = beta1*m[key] + (1.0-beta1)*grads_wandb[key]
                        v[key] = beta2*v[key] + (1.0-beta2)*np.square(grads_wandb[key])

                        m_hat[key] = m[key]/(1.0-np.power(beta1,epoch+1))
                        v_hat[key] = v[key]/(1.0-np.power(beta2,epoch+1))

                        NN.params[key] -= (NN.lr/(np.sqrt(v_hat[key])+epsilon))*m_hat[key]
                    
                    grads_wandb = NN.Initialize_gradients_to_zeros()

            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})


    def nadam(self,NN,train_X , train_Y , val_X ,val_Y):
        """
            Perform training using NADAM Gradient Descent optimization.

            Parameters:
            - NN (NN): Neural network object to train.
            - train_X (numpy.ndarray): Input features for the training data.
            - train_Y (numpy.ndarray): Target labels for the training data.
            - val_X (numpy.ndarray): Input features for the validation data.
            - val_Y (numpy.ndarray): Target labels for the validation data.

            Returns:
            - None

        """

        
        beta1 = 0.9
        beta2 =0.999

        m = NN.Initialize_gradients_to_zeros()
        v = NN.Initialize_gradients_to_zeros()
        m_hat = NN.Initialize_gradients_to_zeros()
        v_hat = NN.Initialize_gradients_to_zeros()
        
        epsilon = NN.epsilon # 1e-8 # This is for mathematical stability 

        gamma = 0.98

        lookahead_history = NN.Initialize_gradients_to_zeros()


        for epoch in range(NN.epochs):
            
            grads_wandb = NN.Initialize_gradients_to_zeros()
            lookahead_grads = NN.Initialize_gradients_to_zeros()
            num_points_seen = 0

            for key in lookahead_grads:
                lookahead_grads[key] = gamma*lookahead_history[key]
            
            for key in NN.params:
                NN.params[key] -= lookahead_grads[key]

            for x,y in zip(train_X,train_Y):

                activations_A , activations_H = NN.forward_pass(x)
                gradients = NN.back_propagation(y , activations_A ,activations_H )

 
            
                for key in grads_wandb:
                    grads_wandb[key]+= gradients[key]

                num_points_seen += 1

                if (num_points_seen % NN.mini_batch_size) == 0 :
                    
                    for key in m:
                        m[key] = beta1*m[key] + ((1.0-beta1)*grads_wandb[key])
                    for key in v:
                        v[key] = beta2*v[key] + (1.0-beta2)*np.square(grads_wandb[key])

                    for key in m_hat:
                        m_hat[key] = m[key]/(1.0 - np.power(beta1,epoch+1))
                    for key in v_hat:
                        v_hat[key] = v[key]/(1.0 - np.power(beta2,epoch+1))

                    for key in lookahead_grads:
                        lookahead_grads[key] = gamma * lookahead_history[key] +NN.lr*grads_wandb[key]

                    # # bias corrected nesterov momemtum
                    # for key in nesterov_lookahead_grads:
                    #     nesterov_lookahead_grads[key] = beta1*m_hat[key] + (1.0-beta1)*grads_wandb[key]

                    # bias corrected v_hat 
                    

                    for key in NN.params:
                        # NN.params[key] -= (NN.lr/(np.sqrt(v_hat[key])+epsilon))*(beta1*m_hat[key]+((1-beta1)/(1.0 - np.power(beta1,epoch+1))*grads_wandb[key])) - np.multiply(NN.lr*gamma,NN.params[key])
                        NN.params[key] -= (NN.lr/(np.sqrt(v_hat[key])+epsilon))*m_hat[key]

                

                    for key in lookahead_history:
                        lookahead_history[key] = lookahead_grads[key]

                    grads_wandb = NN.Initialize_gradients_to_zeros()

            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})
