import wandb
import numpy as np
class Optimizers:


    def vanilla_GD(self,NN,train_X,train_Y,val_X,val_Y):

        for i in range(NN.epochs):
            for x , y in zip(train_X,train_Y):
                
                # print('x shape' ,x.shape)
            
                activations_A , activations_H = NN.forward_pass(x)

                # print('actA shape',activations_A)
                # print('actH shape',activations_H)
                # print('_',_.shape)


                gradients = NN.back_propagation(y , activations_A ,activations_H )
            
                NN.update_weights_and_bias(gradients)
            # print("gradients",gradients)
                # For gradient Update
                # grads  = NN.update_weights_and_bias(gradients)
            # break
            # print('gradient_update_after_each_epoch',gradients)
            train_acc ,train_loss = NN.compute_acc_and_loss(train_X,train_Y)
            val_acc , val_loss = NN.compute_acc_and_loss(val_X,val_Y)
            print('train_Accuracy = ', train_acc ,"train_Loss : ",train_loss , "val_Accuracy:",val_acc, "val_loss:",val_loss)
            wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})

    
    def sgd(self,NN,train_X ,train_Y,val_X,val_Y):

        """
        Below is the implementation of the sthochastic gradient descent

            estimating the total gradient based on a single data point.
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
            # wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})

    def mgd(self,NN, train_X , train_Y,val_X,val_Y):
        """
        Below is the implementation of the momentum based gradient descent
        
        def do_mgd(NN.epochss):
            w,b,eta = -2,-2,1.0
            prev_uw,prev_ub,beta = 0,0,0.9
        
            for i in range(NN.epochss):
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
        This is how the gradient of all the previous updates is added to the current update.

        Update rule for NAG:
            wt+1 = wt - updatet
            While calculating the updatet, We will include the look ahead gradient (∇wlook_ahead).
            updatet = gamma * update_t-1 + η∇wlook_ahead

            ∇wlook_ahead is calculated by:
            wlook_ahead = wt -  gamma*update_t-1

            This look-ahead gradient will be used in our update and will prevent overshooting.
        
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
        Depends on the intial learning rate 
            parameters that worked are
                epsilon = 1e-8
                lr = 0.1
        """
        epsilon = 1e-8 # This is for mathematical stability 
        beta = 0.9

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
        Combination of momentum based gradient descent and RMS prop
            mt=β1⋅mt+(1-β1)⋅(δwtδL)

            vt=β2⋅vt+(1-β2)⋅(δLδwt)2vt=β2⋅vt+(1-β2)⋅(δwtδL)2

        We will also do a bias correction
            mt = mt / (1 - beta1^t)
            vt = vt / (1 - beta2^t)

        """

         
        beta1 = 0.9
        beta2 =0.999

        m = NN.Initialize_gradients_to_zeros()
        v = NN.Initialize_gradients_to_zeros()
        m_hat = NN.Initialize_gradients_to_zeros()
        v_hat = NN.Initialize_gradients_to_zeros()
        
        epsilon = 1e-8 # This is for mathematical stability 


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
        Nesterov Adam
        """

        
        beta1 = 0.9
        beta2 =0.999

        m = NN.Initialize_gradients_to_zeros()
        v = NN.Initialize_gradients_to_zeros()
        m_hat = NN.Initialize_gradients_to_zeros()
        v_hat = NN.Initialize_gradients_to_zeros()
        
        epsilon = 1e-8 # This is for mathematical stability 

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
            # wandb.log({'train_Accuracy': train_acc ,"train_Loss ":train_loss,"val_Accuracy":val_acc,"val_loss":val_loss})
