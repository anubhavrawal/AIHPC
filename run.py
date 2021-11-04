import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import utils

import numpy as np

img_rows, img_cols = 28, 28

# Specify training parameters
batch_size = 128
epochs = 5
num_classes = 10
verbose = 1
'''
def get_dataset(num_classes, rank=0, size=1):
    (x_train, y_train), (x_test, y_test) = mnist.load_data('MNIST-data-%d' % rank)
    x_train = x_train[rank::size]
    y_train = y_train[rank::size]
    x_test = x_test[rank::size]
    y_test = y_test[rank::size]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)
'''
def get_dataset(num_classes, rank=0, size=1):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def get_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


 
def train(learning_rate=1.0):
    (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
    model = get_model(num_classes)

    # Specify the optimizer (Adadelta in this example), using the learning rate input parameter of the function so that Horovod can adjust the learning rate during training
    optimizer = Adadelta(lr=learning_rate)

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_test, y_test))
    model.save("test.h5")
    return model

model = train(learning_rate=0.1)

_, (x_test, y_test) = get_dataset(num_classes)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss:", loss)
print("accuracy:", accuracy)