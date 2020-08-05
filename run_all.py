import numpy as np
import tensorflow as tf

from run_models import *

from adalip_opts import AdamLip, AdaLip, RMSLip
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from adalip_u_opts import AdamLip_U, AdaLip_U, RMSLip_U

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

args = sys.argv

dataset = args[1]
opt = args[2]
lr = args[3]
it = args[4]



if dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = create_mnist()
    model = mnist_model()
    ep = 15
elif dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = create_cifar10()
    model = cifar10_model()
    ep = 30
elif dataset == "cifar100":
    (x_train, y_train), (x_test, y_test) = create_cifar100()
    model = cifar100_model()
    ep = 70


tb = tf.keras.callbacks.TensorBoard(log_dir="logs\\" + dataset + "\\" + opt + "_" + lr + "\\" + str(it))

if opt == "adam":
    optim = Adam(learning_rate=float(lr))
elif opt == "sgd":
    optim = SGD(learning_rate=float(lr))
elif opt == "rmsprop":
    optim = RMSprop(learning_rate=float(lr))

elif opt == "AdamLip":
    optim = AdamLip(learning_rate=float(lr))
elif opt == "AdaLip":
    optim = AdaLip(learning_rate=float(lr))
elif opt == "RMSLip":
    optim = RMSLip(learning_rate=float(lr))


elif opt == "AdamLip_U":
    optim = AdamLip_U(learning_rate=float(lr))
elif opt == "AdaLip_U":
    optim = AdaLip_U(learning_rate=float(lr))
elif opt == "RMSLip_U":
    optim = RMSLip_U(learning_rate=float(lr))


model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])

if (dataset == "mnist") or (dataset == "cifar10"):
    model.load_weights('starting_weights/' + dataset + '/weights' + str(it) + '.h5')
    model.fit(x_train, y_train, batch_size=128, epochs=ep, validation_data=(x_test, y_test), callbacks=[tb])
else:
    model.load_weights('starting_weights/' + dataset + '/weights' + str(it) + '.h5')
    imgen = ImageDataGenerator(rotation_range=15,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
    imgen.fit(x_train)

    model.fit_generator(imgen.flow(x_train, y_train, batch_size=128), epochs=ep, steps_per_epoch=x_train.shape[0]/128 ,validation_data=(x_test, y_test), callbacks=[tb])

