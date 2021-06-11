import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

#load dataset (hand-writing integer)
#28x28 numbers 0-9
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#normalized the data from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#show the image
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()

#build the model
model = tf.keras.models.Sequential()

#input layer (flattened image)
model.add(tf.keras.layers.Flatten())

#fully-connected layer 1
#128 neurons
#relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#fully-connected layer 2
#128 neurons
#relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#output layer
#10 neurons, each for a possible number prediction
#softmax activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit
model.fit(x_train, y_train, epochs=3)

#evaluate with test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print("val loss", val_loss)
print("val acc", val_acc)

#save model
model.save('hand_writing_int_reader.model')