#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from numpy.random import uniform as rand
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings


def lineal_norm(x):
   """ maps an input vector x to the interval [0,1] """
   return (x-np.min(x))/(np.max(x)-np.min(x))


def gen_data(N,noise=0.1, norm=True):
   """
   Returns N samples of the sin(x) function.
   noise to add noise: y = sin(x) + noise*random(-1,1)
   """
   x = np.random.uniform(0,2*np.pi,300)
   y = np.sin(x) + noise*rand(-1,1,len(x))
   if norm:
      x = lineal_norm(x)
      y = lineal_norm(y)
   return x, y

# Generate the training and testing dataset
IN_train, OUT_train = gen_data(300)
IN_test, OUT_test = gen_data(100)


# Let's design the model
#               .-O-.
#              /  O  \
# input --> O-;   O   ;-O  --> output
#        lineal\  O  /lineal
#               '-O-'
#              sigmoid
model = keras.Sequential([
      keras.layers.Dense(1, activation=None, input_shape=(1,)),
      keras.layers.Dense(15, activation='tanh'),
      keras.layers.Dense(5, activation='tanh'),
      keras.layers.Dense(1, activation=None) ])


model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['accuracy'])


# Train the model
history = model.fit(IN_train, OUT_train, epochs=1000,
                    validation_data = (IN_test,OUT_test),
                    verbose=0)

# plot learning curve
err = history.history['loss']
val_err = history.history['val_loss']
acc = history.history['accuracy']
fig, ax = plt.subplots()
ax.plot(err,label='loss')
ax.plot(acc,label='accuracy')
ax.plot(val_err,label='val_loss')
ax.set_title('Learning curve')
ax.legend()


# Prediction over the whole domain
x_predict = np.linspace(0,2*np.pi,500)
x_predict = lineal_norm(x_predict)

y_predict = model.predict(x_predict)
y_predict = lineal_norm(y_predict)

fig, ax = plt.subplots()
ax.scatter(IN_train, OUT_train, label='train')
ax.scatter(IN_test,  OUT_test,label='test')
ax.plot(x_predict, y_predict,'k',lw=2, label='prediction')
ax.legend()
ax.set_title('Results')
plt.show()


# Save result
model.save('my_model.h5')

