import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load dataset (hand-writing integer)
#28x28 numbers 0-9
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#normalized the data from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#load model build in hw_int_reader_build.py
hw_int_reader = tf.keras.models.load_model('hand_writing_int_reader.model')

predictions = hw_int_reader.predict(x_test)
predicted_int = np.argmax(predictions[0])
print("predicted integer: ", predicted_int)

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()

