#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
# TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
Dataset = tf.data.Dataset

fname = 'word2vec_0.input'
N0 = 10000 #inp_shape[0]
ndim = 500

df = pd.read_csv(fname, names=['input','output'])
df = df.sample(frac=1)
print(len(df.index))

inputs  = df['input'].values.astype(int)
outputs = np.expand_dims(df['output'].values, axis=1)
print(np.min(inputs), np.max(inputs))
print(np.min(outputs), np.max(outputs))
exit()
inputs = inputs[:100]
outputs = outputs[:100]
# inputs  = df['input'].values
# outputs = df['output'].values
N0 = np.max(outputs)
inputs = np.expand_dims(tf.one_hot(inputs,N0).numpy(),axis=2)
inp_shape = inputs.shape[1:]

print(inputs.shape)
print(outputs.shape)
print(N0)


# x = Dataset.from_tensor_slices(inputs.transpose()).map(lambda z: tf.one_hot(z, N0))
# y = Dataset.from_tensor_slices(outputs.transpose())  #.map(lambda z: tf.one_hot(z, N0))
# train = Dataset.zip((x, y)).shuffle(500).repeat().batch(32)


# train = tf.data.experimental.make_csv_dataset(fname, batch_size=32)



model = models.Sequential()
model.add(Dense(2000, activation='tanh', input_shape=inp_shape))
model.add(Dense(ndim, activation='tanh'))
model.add(Dense(2000, activation='tanh'))
model.add(Dense(N0,   activation='tanh'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

model.fit(inputs,outputs, epochs=10, verbose=1)
# model.fit(train, epochs=10, steps_per_epoch=10, verbose=1)


model2 = models.Sequential()
model2.add(Dense(2000, activation='tanh',
                       input_shape=inp_shape,
                       weights=model.layers[0].get_weights()))
model2.add(Dense(ndim, activation='tanh',
                       weights=model.layers[1].get_weights()))

activations = model2.predict(inputs)
print(activations)
