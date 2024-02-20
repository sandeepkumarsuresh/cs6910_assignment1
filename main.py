from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import FFNN
from tqdm import tqdm
import wandb

if __name__ == '__main__':

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

    """
    Passing the Data into the Feed Forward Network
    """
    # initializing the model
    model = FFNN.NN(n_hidden_layers=4 , s_hidden_layer = [784 ,128, 32 , 10] )

    # Forward Pass

    model.train(train_X,train_y)











    #Ploting the data
    # for i in range(9):
    #     plt.subplot(330  + 1 + i )
    #     plt.imshow(train_X[i] , cmap = plt.get_cmap('gray'))
    # plt.show()
