from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import FFNN
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import yaml

sweep_configuration = {
    'method': 'bayes', #grid, random
    'metric': {
    'name': 'val_Accuracy',
    'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [3,5]
        },
        'n_hidden_layers': {
            'values': [3,4,5]
        },
        's_hidden_layers': {
            'values': [32,64,128]
        },
        'weight_decay': {
            'values': [0,0.0005,0.5]
        },
        'lr': {
            'values': [1e-2,1e-3,]
        },
        'optimiser': {
            'values': ['sgd', 'mgd', 'nag', 'rms', 'adam','nadam']
        },
        'batch':{
            'values': [16,32,64]
        },
        'weight_para':{
            'values': ['random','Xavier']
        },
        'activation_para': {
            'values': ['tanh','sigmoid', 'relu']
        }
    }
}
sweep_id = wandb.sweep(sweep_configuration,project='test')



def normalise(data):
    """
    Input will be image matrix which should be flatten
    """
    return data/255



# def do_sweep():

#     wandb.init()
#     config = wandb.config
#     run_name = "hidden_layer:"+str(config.n_hidden_layers)+"_mini_batch_size:"+str(config.batch)+"_activations"+str(config.activation_para)
#     print(run_name)
#     wandb.run.name = run_name

#     s_of_hidden_layers = [784]+[config.s_hidden_layers]*config.n_hidden_layers + [10]
#     size_of_network = len(s_of_hidden_layers)
#     # s_of_hidden_layers_ = [config.s_hidden_layers[:-1] for _ in range(config.n_hidden_layers - 1)]
#     # s_of_hidden_layers_.append([10])
#     print('s_of_hidden_layers',s_of_hidden_layers)
#     model = FFNN.NN(n_hidden_layers=config.n_hidden_layers ,
#                     #  s_hidden_layer = [784 ,128, 32 , 10],
#                     size_of_network=size_of_network,
#                     s_hidden_layer = s_of_hidden_layers,
#                      epochs = config.epochs,
#                      optimiser=config.optimiser ,
#                      mini_batch_size=config.batch,
#                      lr = config.lr,
#                      weight_init_params = config.weight_para,
#                      activation=config.activation_para
#                      )
    
#     # Call model.fit here
#     model.fit(train_X_split,train_Y_split,val_X,val_Y)

    # model.vanilla_GD(train_X_split,train_Y_split)
    # model.nadam(train_X_split,train_Y_split)



if __name__ == '__main__':
    
    
    # wandb.init()
    # config = wandb.config

    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    """
    Question 1.
    
    For Ploting the Images , wandb accept in the form of image array and labels. This can be found in the documentation

    The code for plotting to visualize in matplotlib is given below

            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            plt.figure(figsize=(10,10))
            for i in range(25):
                plt.subplot(5,5,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(train_X[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[train_y[i]])
            plt.show()

    """
    # wandb.login()

    # run = wandb.init(
    #     project ="test"
    # )  

    # indices = [list(train_y).index(i) for i in range(10)]
    # images = []
    # labels = []
    # for i in indices:
    #     images.append(train_X[i])
    #     labels.append(class_names[train_y[i]])
    # plots = [wandb.Image(image, caption=caption) for image, caption in zip(images, labels)] # The plots needs to be an array for logging
    # wandb.log({'examples':plots})


    """
    Preprocessing the Data to feed into the network
    -----------------------------------------------------

    Here the values are from 0 to 255 --> brightness of the pixel values
    Therefore converting the pixel values to float values --> for easier calc of gradients
    """
    train_X = (train_X.astype(np.float32)).reshape(len(train_X),-1)
    test_X = test_X.astype(np.float32).reshape(len(test_X),-1)
    test_y = to_categorical(test_y.astype(np.float32))
    train_y = to_categorical(train_y.astype(np.float32))


    #----------------------------------------------------------------------------------
            # Normalising the data
    #----------------------------------------------------------------------------------

    train_X = normalise(train_X)
    test_X = normalise(test_X)


    #----------------------------------------------------------------------------------
            # Creating a Validation Data from the train Data
    #----------------------------------------------------------------------------------

    train_X_split ,val_X , train_Y_split , val_Y = train_test_split(train_X,train_y,test_size=0.10,random_state=42)


    
    # wandb.agent(sweep_id ,function=do_sweep,count=100)
    # wandb.finish()


    #----------------------------------------------------------------------------------
            # Creating a Confusion Matrix 
    
    """
    best hyperparameter

    hidden layer = 4
    batch size = 32
    activations = tanh
    epoch 10
    lr = 1e-3
    optimiser = nadam
    size of hidden layer = 64
    weight para = xavier
    val acc = 88.82%
    """
    #----------------------------------------------------------------------------------

    layers = [784,64,64,64,64,10]
    layer_size = len(layers)

    model_confusion = FFNN.NN(n_hidden_layers=4,
                    #  s_hidden_layer = [784 ,128, 32 , 10],
                    size_of_network=layer_size,
                    s_hidden_layer = layers,
                     epochs = 10,
                     optimiser='nadam' ,
                     mini_batch_size=32,
                     lr = 1e-3,
                     weight_init_params = 'Xavier',
                     activation='tanh'
                     )
    model_confusion.fit(train_X_split,train_Y_split,val_X,val_Y)
    

    test_accuracy , test_loss = model_confusion.evaluate_model_performance(test_X,test_y)

    print('test_accuracy:',test_accuracy)
    print('test loss:',test_loss)