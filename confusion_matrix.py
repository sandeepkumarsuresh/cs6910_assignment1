from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import FFNN
import os
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

def normalise(data):
    """
    Input will be image matrix which should be flatten
    """
    return data/255

if __name__ == '__main__':

    wandb.init()

    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

    train_X = (train_X.astype(np.float32)).reshape(len(train_X),-1)
    test_X = test_X.astype(np.float32).reshape(len(test_X),-1)
    test_y = to_categorical(test_y.astype(np.float32))
    train_y = to_categorical(train_y.astype(np.float32))


    train_X = normalise(train_X)
    test_X = normalise(test_X)

    train_X_split ,val_X , train_Y_split , val_Y = train_test_split(train_X,train_y,test_size=0.10,random_state=42)


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

    pred_labels , truth_labels = [] , []

    for x_test,y_test in zip(test_X,test_y):
        _ , activations_H = model_confusion.forward_pass(x_test)
        pred = (activations_H['h'+ str(model_confusion.size_of_network -1 )])
        pred_labels.append(np.argmax(pred))
        y_truth = np.argmax(y_test.reshape(len(y_test),1))
        truth_labels.append(y_truth)

    save_dir = "Plots"
    os.makedirs(save_dir,exist_ok="True")


    plt.figure(figsize=(15,5))

    # plt.imshow(confusion_matrix(truth_labels,pred_labels,normalize='True'),cmap='viridis')
    # plt.colorbar()
    # plt.title("Confusion Matrix")
    ax = sns.heatmap(confusion_matrix(truth_labels, pred_labels, normalize='true'), cmap='viridis', annot=True)
    ax.set_title("Confusion Matrix", size=16)
    ax.set_ylabel("True", size=14)
    ax.set_xlabel("Predictions", size=14)
    plt.savefig(os.path.join(save_dir, "Confusion_Matrix.png"))
    plt.close()




    test_accuracy , test_loss = model_confusion.evaluate_model_performance(test_X,test_y)

    print('test_accuracy:',test_accuracy)
    print('test loss:',test_loss)