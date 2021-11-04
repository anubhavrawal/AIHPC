import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np

model = load_model("test.h5")

#Resizze to 28px28p
#Use painteditor with 3-5pixel tickness
img = Image.open('/mnt/c/Users/Anubhav_R/Desktop/tmp/AIHPC/3.png').convert("L")

plt.subplot(1, 2, 1)
img_tmp = np.resize(img, (28,28))
plt.imshow(img_tmp)
plt.savefig('temp')


img = np.resize(img, (28,28))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,28,28,1)

#y_pred = model.predict(im2arr)
#print(decode_predictions(y_pred, top=3)[0])

im2arr = im2arr / 255 

yh = np.argmax(model.predict(im2arr.reshape(1,28,28,1)), axis=-1)
print(yh)



