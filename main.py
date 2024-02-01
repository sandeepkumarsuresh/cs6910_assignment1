from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import FFNN
from tqdm import tqdm
if __name__ == '__main__':

    # Loading the Fashion-MNIST Dataset
    
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

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
    model = FFNN.NN(n_hidden_layers=5 , s_hidden_layer = [784 ,128, 64, 32, 10] )

    # Forward Pass

    model.train(train_X,train_y)









    #Ploting the data
    # for i in range(9):
    #     plt.subplot(330  + 1 + i )
    #     plt.imshow(train_X[i] , cmap = plt.get_cmap('gray'))
    # plt.show()
