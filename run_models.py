import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.datasets import mnist, cifar10, cifar100


def mnist_model():
    inp = Input(shape=(28, 28, 1,))
    c1 = Conv2D(32, kernel_size=3, activation='relu', use_bias=False)(inp)
    c2 = Conv2D(64, kernel_size=3, activation='relu', use_bias=False)(c1)

    mp = MaxPooling2D(pool_size=2)(c2)
    fl = Flatten()(mp)
    d1 = Dense(128, activation='relu', use_bias=False)(fl)
    out = Dense(10, activation='softmax')(d1)

    model = Model(inp, out)
    return model


def cifar10_model():
    inp = Input(shape=(32, 32, 3,))
    c1 = Conv2D(32, kernel_size=3, activation='relu', use_bias=False)(inp)
    c2 = Conv2D(32, kernel_size=3, activation='relu', use_bias=False)(c1)
    mp1 = MaxPooling2D(pool_size=2)(c2)

    c3 = Conv2D(64, kernel_size=3, activation='relu', use_bias=False)(mp1)
    c4 = Conv2D(64, kernel_size=3, activation='relu', use_bias=False)(c3)
    mp2 = MaxPooling2D(pool_size=2)(c4)

    c5 = Conv2D(128, kernel_size=3, activation='relu', use_bias=False, padding='same')(mp2)
    c6 = Conv2D(128, kernel_size=3, activation='relu', use_bias=False, padding='same')(c5)
    mp3 = MaxPooling2D(pool_size=2)(c6)

    fl = Flatten()(mp3)
    d1 = Dense(200, activation='relu', use_bias=False)(fl)
    out = Dense(10, activation='softmax')(d1)

    model = Model(inp, out)
    return model



def cifar100_model():
    inp = Input(shape=(32, 32, 3,))

    b = Conv2D(64, (3, 3), padding='same', activation='relu')(inp)
    b = BatchNormalization()(b)
    b = MaxPooling2D(pool_size=(2, 2))(b)

    b = Conv2D(128, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    b = MaxPooling2D(pool_size=(2, 2))(b)


    b = Conv2D(256, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    # b = Dropout(0.4)(b)

    b = Conv2D(256, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    # b = Dropout(0.4)(b)
    b = MaxPooling2D(pool_size=(2, 2))(b)


    b = Conv2D(512, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    # b = Dropout(0.4)(b)

    b = Conv2D(512, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    # b = Dropout(0.4)(b)
    b = MaxPooling2D(pool_size=(2, 2))(b)
    
    b = Conv2D(512, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    
    b = Conv2D(512, (3, 3), padding='same', activation='relu')(b)
    b = BatchNormalization()(b)
    # b = Dropout(0.4)(b)
    b = MaxPooling2D(pool_size=(2, 2))(b)

    fl = Flatten()(b)

    d1 = Dense(512, activation='relu')(fl)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.5)(d1)

    out = Dense(100, activation='softmax')(d1)
    model = Model(inp, out)

    return model


def create_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return ((x_train, y_train), (x_test, y_test))

def create_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return ((x_train, y_train), (x_test, y_test))

def create_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))


    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

    return ((x_train, y_train), (x_test, y_test))
