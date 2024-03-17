# CS6910_Assignment1 Feed Forward Neural Network 

This repository contains all the files which implements FeedForward Neural Network from Scratch.



## Getting Started

### For Evaluation use final_dev

This assignment has two folders :

local_dev contains all the relating to the local implementation of the project. This contains the wandb sweep codes and other experiments which were done
as the part of this assignment. 

final_dev contains all files to run the code in the your local systems . 

Kindly use final dev to run the experiments in your system .

Plots folder contains the plots generated as part of the experiment.

## Files in final_dev

* activations_and_lossfunctions.py -- Contains the Activations and the loss Functions
* FFNN.py -- Contains class Definition for Forward and Backward Propagation Model
* optimizers.py -- Contains all the Optimization Code
* train.py -- Main Function which loads the dataset and run trains the network

## Prerequisites

Before running the code you need to install all the requirements

```
pip3 install -r requirements.txt
```


## Command-line Arguments For Running the Script
Weights & Biases Configuration

    -wp, --wandb_project: Project name used to track experiments in Weights & Biases dashboard. (default: "dl_ass1)

### Training Configuration

    -d, --dataset: Dataset to use for training. Choices: ['mnist', 'fashion_mnist']. (required)
    -e, --epochs: Number of epochs to train neural network. (default: 2)
    -b, --batch_size: Batch size used to train neural network. (default: 4)
    -l, --loss: Loss function to use. Choices: ['mean_squared_error', 'cross_entropy']. (default: "cross_entropy")
    -o, --optimizer: Optimizer to use. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']. (required)
    -lr, --learning_rate: Learning rate used to optimize model parameters. (default: 0.1)
    -m, --momentum: Momentum used by momentum and nag optimizers. (default: 0.5)
    -beta, --beta: Beta used by rmsprop optimizer. (default: 0.5)
    -beta1, --beta1: Beta1 used by adam and nadam optimizers. (default: 0.5)
    -beta2, --beta2: Beta2 used by adam and nadam optimizers. (default: 0.5)
    -eps, --epsilon: Epsilon used by optimizers. (default: 0.000001)
    -w_d, --weight_decay: Weight decay used by optimizers. (default: 0.0)
    -w_i, --weight_init: Weight initialization method. Choices: ['random', 'Xavier']. (default: "random")
    -nhl, --num_layers: Number of hidden layers used in feedforward neural network. (default: 1)
    -sz, --hidden_size: Number of hidden neurons in a feedforward layer. (default: 4)
    -a, --activation: Activation function to use. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU']. (default: "sigmoid")


### Default Hyperparameter Based On Best Sweep Result

Hyperparameter | Values/Usage
-------------------- | --------------------
n_classes | 10
n_hlayers | 4
epochs | 10
activation |  'tanh'
loss | 'cross_entropy', 
output_activation | 'softmax'
batch_size |  32
initializer | 'xavier'
hlayer_size |  64


### Example

```
usage: python3 train.py [-h] 
	[-d {mnist,fashion_mnist}] 
	[-e EPOCHS] 
	[-b BATCH_SIZE] 
	[-l {mean_squared_error,cross_entropy}] 
	[-o {sgd,momentum,nag,rmsprop,adam,nadam}] 
	[-lr LEARNING_RATE] 
	[-m MOMENTUM] 
	[-beta BETA] 
	[-beta1 BETA1] 
	[-beta2 BETA2] 
	[-eps EPSILON] 
	[-w_d WEIGHT_DECAY] 
	[-w_i {random,Xavier}] 
	[-nhl NUM_LAYERS] 
	[-sz HIDDEN_SIZE] 
	[-a {sigmoid,tanh,ReLU}]
```
The below code in the terminal runs with the default hyperparameter which gave the best result
```
python3 train.py 
```
## References

* https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
* https://www.kaggle.com/code/venkatkrishnan/data-augmentation-deep-learning


## Author

* **Sandeep Kumar Suresh** - *Github Link* - [Github](https://github.com/sandeepkumarsuresh/cs6910_assignment1)
                            *Wandb Link* - [Wandb](https://wandb.ai/ee23s059/dl_ass1/reports/CS6910-Assignment-1--Vmlldzo3MTc2NjEz)
