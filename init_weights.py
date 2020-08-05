import tensorflow as tf
import numpy as np
import os
from run_models import *

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Number of runs (number of different set of weights for a specific network)
iterations = 20

mnist_path = "./starting_weights499/mnist/"  
cifar10_path = "./starting_weights499/cifar10/"
cifar100_path = "./starting_weights499/cifar100/"

make_folder(mnist_path)
make_folder(cifar10_path)
make_folder(cifar100_path)

for it in range(iterations):

    # MNIST    
    model = mnist_model()
    model.save_weights(mnist_path + "weights" + str(it) + ".h5")

    # CIFAR10
    model = cifar10_model()
    model.save_weights(cifar10_path + "weights" + str(it) + ".h5")

    # CIFAR100
    model = cifar100_model()
    model.save_weights(cifar100_path + "weights" + str(it) + ".h5")
