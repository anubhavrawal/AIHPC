import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

import numpy as np

# -----------------------------------------------------------
# Hyperparameters
batch_size = 128
num_classes = 10
epochs = 12

activation = 'relu'
verbose = 1


# -----------------------------------------------------------
# Image Datasets

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
('x_train shape:', (60000, 28, 28, 1))


model = Sequential()
  
# Convolution Layer
model.add(Conv2D(32, kernel_size=(3, 3),
				activation=activation,
				input_shape=input_shape)) 

# Convolution layer
model.add(Conv2D(64, (3, 3), activation=activation))

# Pooling with stride (2, 2)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Delete neuron randomly while training (remain 75%)
#   Regularization technique to avoid overfitting
model.add(Dropout(0.25))

# Flatten layer 
model.add(Flatten())

# Fully connected Layer
model.add(Dense(128, activation=activation))

# Delete neuron randomly while training (remain 50%) 
#   Regularization technique to avoid overfitting
model.add(Dropout(0.5))

# Apply Softmax
model.add(Dense(num_classes, activation='softmax'))

# Loss function (crossentropy) and Optimizer (Adadelta)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
			optimizer= tf.keras.optimizers.Adadelta(),
			metrics=['accuracy'])

# Fit our model
model.fit(x_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		verbose=verbose,
		validation_data=(x_test, y_test))


model.save("my_model.h5")

# Calculate Test loss and Test Accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)

