#!/usr/local/bin/python3

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):     #  logs.get('acc')
            print("\n Reached 60% accuracy so early stopping")
            self.model.stop_training = True

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

callbacks = myCallback()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(tf.argmax(classifications[0],0))
print(test_labels[0])
