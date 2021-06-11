import tensorflow as tf
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


import pickle
import numpy as np
import time

#get the training data X, y built in dog_cat_data_build.py
pickle_in = open("dog_cat_dataset/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("dog_cat_dataset/y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0
y=np.array(y)

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name_of_cnn = "{}-conv-{}-node-{}-dense{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            model = Sequential()

            #layer 1
            model.add(Conv2D(layer_size, (3,3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            #layer 2
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            #layer 3
            model.add(Flatten()) # converts 3D feature maps to 1D feature vectors
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            #output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(name_of_cnn))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

#to monitor the training
#paste this to the terminal with the working directory
#tensorboard --logdir=logs/