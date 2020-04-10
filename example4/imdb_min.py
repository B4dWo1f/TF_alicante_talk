#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
   fig, ax = plt.subplots()
   ax.plot(history.history[metric])
   ax.plot(history.history['val_'+metric], '')
   ax.set_xlabel("Epochs")
   ax.set_ylabel(metric)
   ax.legend([metric, 'val_'+metric])
   plt.show()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print('Vocabulary size:',encoder.vocab_size)

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is:',encoded_string)

original_string = encoder.decode(encoded_string)
print('The original string:',original_string)

for index in encoded_string:
  print(f'{index} ----> {encoder.decode([index])}')


BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (train_examples.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None],[])))

test_dataset = (test_examples.padded_batch(BATCH_SIZE,padded_shapes=([None],[])))

train_dataset = (train_examples.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=([None],[])))

test_dataset = (test_examples.padded_batch(BATCH_SIZE,padded_shapes=([None],[])))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(encoder.vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

plot_graphs(history, 'accuracy')

test_loss, test_acc = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')


