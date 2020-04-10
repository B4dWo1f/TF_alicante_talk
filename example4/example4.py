#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
# NLP
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train, valid, test = tfds.load(name="imdb_reviews",
                               split=('train[:60%]', 'train[60%:]', 'test'),
                               as_supervised=True)


Nsamples = int(1e3)
Nwords = 5  # correlation words distance
Nraw = int(1e3)
# Ndim = int(1e4)


train_iter = train.__iter__()

sentences = []
for i in range(Nsamples):
   x,y = train_iter.get_next()
   sentences.append(x.numpy().decode('utf-8'))

tokenizer = Tokenizer(num_words=Nraw, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

exit()
sequences = tokenizer.texts_to_sequences(sentences)
sequences = pad_sequences(sequences,padding='post',maxlen=300)

for seq in sequences:
   print(len(seq))

