import argparse
import FFNN
import optimizers
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


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
                    default=2,
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




    train_X_split ,val_X , train_Y_split , val_Y = train_test_split(train_X,train_y,test_size=0.10,random_state=42)

    datagen = ImageDataGenerator(
                            # featurewise_center=True,
                            # featurewise_std_normalization=True,
        
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2
                            # rotation_range=20,
                            # width_shift_range=0.2,
                            # height_shift_range=0.2
                            # # horizontal_flip=True,
                            )
    
    # datagen.fit(train_X_split)
    train_X_split = np.expand_dims(train_X_split, axis=-1)  
    augmented_data_generator = datagen.flow(train_X_split, train_Y_split, batch_size=len(train_X_split), shuffle=False)
    augmented_data = augmented_data_generator.next()

    train_X_augmented = (augmented_data[0].astype(np.float32)).reshape(len(augmented_data[0]), -1)
    train_Y_augmented = to_categorical(augmented_data[1].astype(np.float32))

    # train_X_split = (train_X_split.astype(np.float32)).reshape(len(train_X_split),-1)
    test_X = test_X.astype(np.float32).reshape(len(test_X),-1)
    val_X = val_X.astype(np.float32).reshape(len(val_X),-1)

    # train_Y_split = to_categorical(train_Y_split.astype(np.float32))
    test_y = to_categorical(test_y.astype(np.float32))
    val_Y = to_categorical(val_Y.astype(np.float32))


    train_X_augmented = normalise(train_X_augmented)
    test_X = normalise(test_X)
    val_X = normalise(val_X)






    # Creating a class object for the neural network
    
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


    # Creating a class object optimizers

    opt = optimizers.Optimizers()

    if args.optimizer == 'vanilla_GD':
        opt.vanilla_GD(model, train_X_augmented, train_Y_augmented, val_X, val_Y)
    elif args.optimizer == 'sgd':
        opt.sgd(model, train_X_augmented, train_Y_augmented, val_X, val_Y)
    elif args.optimizer == 'mgd':
        opt.mgd(model, train_X_augmented, train_Y_augmented, val_X, val_Y)
    elif args.optimizer == 'nag':
        opt.nag(train_X_augmented, train_Y_augmented, val_X, val_Y)
    elif args.optimizer == 'rms_prop':
        opt.rms_prop(model, train_X_augmented, train_Y_augmented, val_X, val_Y)
    elif args.optimizer == 'adam':
        opt.adam(model, train_X_augmented, train_Y_augmented, val_X, val_Y)
    elif args.optimizer == 'nadam':
        opt.nadam(model, train_X_augmented, train_Y_augmented, val_X, val_Y)
    else:
        return "Error in fit function. Optimiser Value must be specified"


if __name__ == '__main__':

    main(args)

