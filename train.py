import argparse
import FFNN
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser()

parser.add_argument("-wp", "--wandb_project",
                    type=str, 
                    default="myprojectname",
                    help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("-we", "--wandb_entity",
                    type=str,
                    default="myname",
                    help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-d", "--dataset",
                    type=str, 
                    default="fashion_mnist",
                    choices=["mnist", "fashion_mnist"],
                    required=True,
                    help="Dataset to use for training. Choices: ['mnist', 'fashion_mnist']")
parser.add_argument("-e", "--epochs",
                    type=int,
                    default=1,
                    help="Number of epochs to train neural network.")
parser.add_argument("-b", "--batch_size", 
                    type=int, 
                    default=4,
                    help="Batch size used to train neural network.")
parser.add_argument("-l", "--loss", 
                    type=str, 
                    default="cross_entropy",
                    choices=["mean_squared_error", "cross_entropy"],
                    help="Loss function to use. Choices: ['mean_squared_error', 'cross_entropy']")
parser.add_argument("-o", "--optimizer", 
                    type=str, 
                    default="sgd",
                    choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                    required=True,
                    help="Optimizer to use. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
parser.add_argument("-lr", "--learning_rate", 
                    type=float, 
                    default=0.1,
                    help="Learning rate used to optimize model parameters.")
parser.add_argument("-m", "--momentum", 
                    type=float, 
                    default=0.5,
                    help="Momentum used by momentum and nag optimizers.")
parser.add_argument("-beta", "--beta", 
                    type=float, 
                    default=0.5,
                    help="Beta used by rmsprop optimizer.")
parser.add_argument("-beta1", "--beta1", 
                    type=float, 
                    default=0.5,
                    help="Beta1 used by adam and nadam optimizers.")
parser.add_argument("-beta2", "--beta2", 
                    type=float, 
                    default=0.5,
                    help="Beta2 used by adam and nadam optimizers.")
parser.add_argument("-eps", "--epsilon", 
                    type=float, 
                    default=0.000001,
                    help="Epsilon used by optimizers.")
parser.add_argument("-w_d", "--weight_decay", 
                    type=float, 
                    default=0.0,
                    help="Weight decay used by optimizers.")
parser.add_argument("-w_i", "--weight_init", 
                    type=str, 
                    default="random",
                    choices=["random", "Xavier"],
                    help="Weight initialization method. Choices: ['random', 'Xavier']")
parser.add_argument("-nhl", "--num_layers", 
                    type=int, 
                    default=1,
                    help="Number of hidden layers used in feedforward neural network.")
parser.add_argument("-sz", "--hidden_size", 
                    type=int, 
                    default=4,
                    help="Number of hidden neurons in a feedforward layer.")
parser.add_argument("-a", "--activation", 
                    type=str, 
                    default="sigmoid",
                    choices=["identity", "sigmoid", "tanh", "ReLU"],
                    help="Activation function to use. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU']")

args = parser.parse_args()

def normalise(data):
    """
    Input will be image matrix which should be flatten
    """
    return data/255

def main(args):

    print('function call inside main')

    if args.dataset == 'fashion_mnist':
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    elif args.dataset == 'mnist':
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = (train_X.astype(np.float32)).reshape(len(train_X),-1)
    test_X = test_X.astype(np.float32).reshape(len(test_X),-1)
    test_y = to_categorical(test_y.astype(np.float32))
    train_y = to_categorical(train_y.astype(np.float32))

    train_X = normalise(train_X)
    test_X = normalise(test_X)

    train_X_split ,val_X , train_Y_split , val_Y = train_test_split(train_X,train_y,test_size=0.10,random_state=42)




    # Creating a class object 
    
    s_of_hidden_layers = [784]+[args.hidden_size]*args.num_layers + [10]
    total_layer_size = len(s_of_hidden_layers)
    print(total_layer_size)
    model = FFNN.NN(n_hidden_layers=args.num_layers,
                    #  s_hidden_layer = [784 ,128, 32 , 10],
                    size_of_network=total_layer_size,
                    s_hidden_layer = s_of_hidden_layers,
                     epochs = args.epochs,
                     optimiser= args.optimizer ,
                     mini_batch_size=args.batch_size,
                     lr = args.learning_rate,
                     weight_init_params = args.weight_init,
                     activation=args.activation
                     )


if __name__ == '__main__':

    main(args)

