#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# TensorFlow
import os
HOME = os.getenv('HOME')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname=f'{HOME}/tensorflow_datasets/climate/jena_climate_2009_2016.csv.zip',
    extract=True)

csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
   start_index = start_index + history_size
   if end_index is None: end_index = len(dataset) - target_size

   data, labels = [],[]
   for i in range(start_index, end_index):
      indices = range(i-history_size, i, step)
      data.append(dataset[indices])

   if single_step: labels.append(target[i+target_size])
   else:
      print('====')
      labels.append(target.iloc[i:i+target_size])
   return np.array(data), np.array(labels)


tf.random.set_seed(13)
TRAIN_SPLIT = 300000


features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']

features = df[features_considered]
features.index = df['Date Time']
print(features.head())
features.plot(subplots=True)


# Normalize
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std




past_history = 720
future_target = 72
STEP = 6

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)



BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()



exit()
################################################################################
uni_data = df['T (degC)']
uni_data.index = df['Date Time']


# Normalize data
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data-uni_train_mean)/uni_train_std


# Prepare data
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])


def create_time_steps(length):
   return list(range(-length, 0))


def show_plot(plot_data, delta, title):
   labels = ['History', 'True Future', 'Model Prediction']
   marker = ['.-', 'rx', 'go']
   time_steps = create_time_steps(plot_data[0].shape[0])
   if delta: future = delta
   else: future = 0

   plt.title(title)
   for i, x in enumerate(plot_data):
      if i:
         plt.plot(future, plot_data[i], marker[i], markersize=10,
                                                   label=labels[i])
      else:
         plt.plot(time_steps, plot_data[i].flatten(), marker[i],
                                                      label=labels[i])
   plt.legend()
   plt.xlim([time_steps[0], (future+5)*2])
   plt.xlabel('Time-Step')
   return plt


def baseline(history):
   return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')

plt.show()




train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


# Model
simple_lstm_model = models.Sequential()
simple_lstm_model.add( LSTM(8, input_shape=x_train_uni.shape[-2:]) )
simple_lstm_model.add( Dense(1) )

simple_lstm_model.compile(optimizer='adam', loss='mae')


for x, y in val_univariate.take(1):
   print(simple_lstm_model.predict(x).shape)



EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)


for x, y in val_univariate.take(3):
   plot = show_plot([x[0].numpy(), y[0].numpy(),
                     simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
   plot.show()
