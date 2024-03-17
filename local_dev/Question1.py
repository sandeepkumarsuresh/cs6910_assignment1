from keras.datasets import fashion_mnist , mnist
import wandb
if __name__ == '__main__':

    wandb.login()

    run = wandb.init(
        project ="dl_ass1"
    )  


    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    # (train_X, train_y), (test_X, test_y) = mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
    indices = [list(train_y).index(i) for i in range(10)]
    images = []
    labels = []
    for i in indices:
        images.append(train_X[i])
        labels.append(class_names[train_y[i]])
    plots = [wandb.Image(image, caption=caption) for image, caption in zip(images, labels)] # The plots needs to be an array for logging
    wandb.log({'Fashion MNIST Plots':plots})
    # wandb.log({'MNIST Plots':plots})
