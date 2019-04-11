#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#
# to get rid of the TF compilation warnings
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Get the data
# In this case we create it manually
IN_train = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

OUT_train = np.array([[0],
                      [1],
                      [1],
                      [0]])

# There is no test data, since the inuputs are constrained to $\{0,1\}$.
# We use the training set as testing
IN_test = IN_train
OUT_test = OUT_train

# Try to load previously trained model
try: model = keras.models.load_model('my_model.h5')
except OSError:
   model = keras.Sequential([
         keras.layers.Dense(2, activation=tf.nn.sigmoid, input_shape=(2,)),
         keras.layers.Dense(1, activation=None) ])


model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['accuracy'])

print('Before training:')
print('Input   xpct Out   Output')
predicted = model.predict(IN_train)
for i in range(IN_train.shape[0]):
   print(IN_train[i],'   ',OUT_train[i],'      %.2f'%(predicted[i]))

# Train the model
history = model.fit(IN_train, OUT_train, epochs=5000,
                    validation_data = (IN_test,OUT_test),
                    verbose=0)

# plot learning curve
err = history.history['loss']
acc = history.history['accuracy']
fig, ax = plt.subplots()
ax.plot(err,label='loss')
ax.plot(acc,label='accuracy')
ax.set_title('Learning curve')


print('')
print('After training:')
print('Input    xpct Out      Output')
predicted = model.predict(IN_train)
for i in range(IN_train.shape[0]):
   print(IN_train[i],'   ',OUT_train[i],'      %.2f'%(predicted[i]))
plt.show()

# Save result
model.save('my_model.h5')

