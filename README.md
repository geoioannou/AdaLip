# AdaLip:  An Adaptive Learning Rate Method per Layer for Stochastic Optimization



## Abstract

Various works have been published around the optimization of Neural Networks thatemphasize the significance of the learning rate. In this study we analyze the need fora different treatment for each layer and how this affects training. We propose a noveloptimization technique, called AdaLip, that utilizes the Lipschitz constant of the gradientsin order to construct an adaptive learning rate per layer that can work on top of alreadyexisting optimizers, like SGD or Adam. A detailed experimental framework was used toprove the usefulness of the optimizer on three benchmark datasets. It showed that AdaLipimproves the training performance and the convergence speed, but also made the wholetraining process more robust to the selection of the initial global learning rate.

Paper under review, submitted to JMLR.
