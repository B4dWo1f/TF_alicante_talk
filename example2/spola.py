#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings



# Load training and testing data
n_out = 1   # number of outputs
M = np.loadtxt('Spola.train')
IN = M[:,:-n_out]
OUT = M[:,-n_out:]

M = np.loadtxt('Spola.test')
IN_test = M[:,:-n_out]
OUT_test = M[:,-n_out:]
inp_shape = IN_test.shape[1:]

print('Read data samples:')
print('%s for training  //  %s for testing'%(IN.shape[0],IN_test.shape[0]))
print('input dimension: %s'%(IN.shape[1]))
print('output dimension: %s'%(OUT.shape[1]))

# Build the model
print('Build the model')
model = keras.Sequential([
      keras.layers.Dense(2, activation=None, input_shape=inp_shape),
      keras.layers.Dense(5, activation=tf.nn.softmax),
      keras.layers.Dense(1, activation=tf.nn.relu) ])

model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['accuracy'])

print('start training')
history = model.fit(IN, OUT, epochs = 5000,
                    validation_data = (IN_test,OUT_test),
                    verbose=0)

print('Saving after training')
model.save('spola.h5')

# plot learning curve
err = history.history['loss']
acc = history.history['accuracy']
fig, ax = plt.subplots()
ax.plot(err,label='loss')
ax.plot(acc,label='accuracy')
ax.set_title('Learning curve')


# Plot result
print('Calculate & plot result')
OUT_predict = model.predict(IN_test)

fig = plt.figure()
ax0 = plt.subplot(121, projection='polar')   # expected
ax1 = plt.subplot(122, projection='polar')   # predicted

fig.suptitle("$Neural$ $Network$", fontsize=30)

# Plot Expected outputs
# this is just a fancy polar plot, the relevant lines are just the scatter
X = IN_test[:,0]
Y = IN_test[:,1]
Z = OUT_test.flatten()
ax0.scatter(Y,X,c=Z)  # plot the expected result
ax0.set_rmax(36)
ax0.set_theta_zero_location("N")
ax0.set_title('Expected output',y=1.1)
ax0.set_xticklabels(['N', '', 'W', '', 'S', '', 'E', ''])
ax0.set_rlabel_position(45)

# Plot Predicted outputs
X = IN_test[:,0]
Y = IN_test[:,1]
Z = OUT_predict.flatten()
ax1.scatter(Y,X,c=Z)  # plot the predicted result
ax1.set_rmax(36)
ax1.set_theta_zero_location("N")
ax1.set_title('Predicted output',y=1.1)
ax1.set_xticklabels(['N', '', 'W', '', 'S', '', 'E', ''])
ax1.set_rlabel_position(45)

fig.tight_layout()
plt.show()
