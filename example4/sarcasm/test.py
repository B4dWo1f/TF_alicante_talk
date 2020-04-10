#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from urllib.request import urlretrieve
import json


vocab_size = 10000
embedding_dim = 100
max_length = 200
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
fname = '/tmp/sarcasm.json'

if not os.path.isfile(fname): A = urlretrieve(url,fname)
with open(fname, 'r') as f:
    datastore = json.load(f)

sentences,labels = [],[]
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# from random import randint
# out = False
# i = 0
# while not out:
#    print(sentences[i])
#    print(labels[i])
#    print('')
#    if labels[i] == 1:
#       out = True
#    i = randint(0,len(labels))
# exit()

training_sentences = sentences[0:training_size]
testing_sentences  = sentences[training_size:]
training_labels    = labels[0:training_size]
testing_labels     = labels[training_size:]


# Curate the data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                                    padding=padding_type,
                                                    truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                                  padding=padding_type,
                                                  truncating=trunc_type)


import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


model = tf.keras.Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
# model.add(LSTM(5))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('\nEmbedding')
e = model.layers[0]
print('************')
print(e)
print(type(e))
print('************')
weights = e.get_weights()[0]
print(weights.shape)
exit()

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# plot learning curve
err = history.history['loss']
val_err = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
fig.subplots_adjust(wspace=0.,hspace=0.15)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(err,label='loss')
ax1.plot(val_err,label='val_loss')
ax2.plot(acc,label='accuracy')
ax2.plot(val_acc,label='val_accuracy')
ax1.set_title('Learning curve')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax1.legend()
ax2.legend()
plt.show()



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# def decode_sentence(text):
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_sentence(training_padded[0]))
# print(training_sentences[2])
# print(labels[2])


import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
for x,y in zip(sentence,pred):
   print(x)
   print(y)
   print('')
