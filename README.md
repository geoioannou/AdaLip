# AdaLip:  An Adaptive Learning Rate Method per Layer for Stochastic Optimization



## Abstract

Various works have been published around the optimization of Neural Networks thatemphasize the significance of the learning rate. In this study we analyze the need fora different treatment for each layer and how this affects training. We propose a noveloptimization technique, called AdaLip, that utilizes the Lipschitz constant of the gradientsin order to construct an adaptive learning rate per layer that can work on top of alreadyexisting optimizers, like SGD or Adam. A detailed experimental framework was used toprove the usefulness of the optimizer on three benchmark datasets. It showed that AdaLipimproves the training performance and the convergence speed, but also made the wholetraining process more robust to the selection of the initial global learning rate.

Paper under review, submitted to JMLR.


-------------------------------------------------

To run the code:


First initiaze the weights with `python init_weights.py` changing the variable `iterations` to how many different sets of weights you want to save.

Then, use `python script.py` to train a network in MNIST, CIFAR10 or CIFAR100 using AdaLip or any of the other optimizers said in `script.py`.

You can change the dataset, the optimizer, the learning rates, the number of different runs in the `script.py`.

-----------------------------------
Dependencies:

Python 3.6.5

numpy (1.18.1)

tensorflow (2.1.0)
