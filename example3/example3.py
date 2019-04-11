#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings



mnist = tf.keras.datasets.mnist


def plot_sample(M):
   fig, ax = plt.subplots()
   ax.imshow(M)
   plt.show()


(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


# Build the model
model = tf.keras.models.Sequential([
                 tf.keras.layers.Flatten(input_shape=(28, 28)),
                 tf.keras.layers.Dense(512, activation=tf.nn.relu),
                 tf.keras.layers.Dense(100, activation=tf.nn.relu),
                 tf.keras.layers.Dense(10, activation=tf.nn.softmax) ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Training
print('Training')
history = model.fit(x_train, y_train, epochs=10,
                    validation_split=0.0,
                    verbose=0)
                    #validation_data = (x_test, y_test),


# plot learning curve
err = history.history['loss']
acc = history.history['accuracy']
fig, ax = plt.subplots()
ax.plot(err,label='loss')
ax.plot(acc,label='accuracy')
ax.set_title('Learning curve')

# Show efficiency
print('Evaluating')
loss,acc = model.evaluate(x_test, y_test,verbose=0)
print('accuracy:',acc)


# Show Results
predictions = model.predict(x_test)

print('let us check 4 random examples')
samples = [randint(0,x_test.shape[0]) for _ in range(4)]

fig, ax = plt.subplots()
gs = gridspec.GridSpec(2, 2)
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])

axs = [ax0,ax1,ax2,ax3]
for i in range(len(samples)):
   ax = axs[i]
   ind = samples [i]
   img = x_test[ind,:,:]
   label = y_test[ind]
   predicted = np.argmax(predictions[ind])
   ax.imshow(img)
   ax.set_xticks([])
   ax.set_yticks([])
   ax.set_title('The NN says: %s'%(predicted))
plt.show()
