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
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


# Get data
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']


encoder = info.features['text'].encoder

# print('Vocabulary size:',encoder.vocab_size)
# sample_string = 'Hello TensorFlow.'
# print(sample_string)
# encoded_string = encoder.encode(sample_string)
# print('Encoded string is:',encoded_string)
# original_string = encoder.decode(encoded_string)
# print('The original string:',original_string)

Nwords = int(1e4)
tokenizer = Tokenizer(num_words=Nwords, oov_token='<OOV>')
XXX = 5000
reviews,lengths = [],[]
i = 0
for dataset in [train_examples]:  #, test_examples]:
   for x,y in tqdm(dataset):
      text = encoder.decode(x)
      reviews.append( text.replace('<br />','') )
      lengths.append( len(text) )
      if i > XXX: break
      i += 1
tokenizer.fit_on_texts(reviews)
word_index = tokenizer.word_index
print(f'Tokenizer found {len(word_index)} different words')
vocabulary = json.loads(tokenizer.get_config()['word_counts'])
x,y = [],[]
for k,v in vocabulary.items():
   x.append(k)
   y.append(v)
inds = np.argsort(y)
inds = inds[-Nwords:][::-1]

maxlen=1100
sequences = tokenizer.texts_to_sequences(reviews)
sequences = pad_sequences(sequences,padding='post',maxlen=maxlen)

# with open('word_index.dict','w') as f:
#    for word,index in word_index.items():
#       f.write(f'{word},{index}\n')

# import pandas as pd
# df = pd.read_csv('word_index.dict',names=['words','index'])
# word_index = dict(zip(df['words'],df['index']))


f_dataset = open('word2vec.input','w')
Nforward = 3
samples = []
for _ in range(1):
   for word in vocabulary:
   # for word,ind in word_index.items():
      ind = word_index[word]
      if word == '<OOV>': continue
      rows,cols = np.where(sequences==ind)
      for r,c in zip(rows,cols):
         seq = sequences[r]
         n = np.random.randint(1,Nforward)
         try:
            samples.append( (ind, seq[c+n]) )
            f_dataset.write(f'{ind},{seq[c+n]}\n')
         except IndexError:
            pass
      f_dataset.flush()
f_dataset.close()


# fig, ax = plt.subplots()
# ax.hist(lengths,bins=100)
# ax.set_xlabel('length of reviews (words)')
# ax.set_ylabel('# of reviews')


plt.show()
