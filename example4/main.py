#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.dates as mdates
from tqdm import tqdm
# TensorFlow
import os
HOME = os.getenv('HOME')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to get rid of the TF warnings
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Reshape, Flatten
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping



# Get the data
# T0 = dt.datetime(2015,1,1)
url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
csv_path = f'{HOME}/tensorflow_datasets/climate/jena_climate_2009_2016.csv.zip'

zip_path = get_file(origin = url, fname = csv_path,
                    archive_format='zip',extract=True)

df = pd.read_csv(csv_path)
# Convert dates & sort
df['Date Time'] = pd.to_datetime(df['Date Time'],format='%d.%m.%Y %H:%M:%S')
# df = df[df['Date Time'] > T0]
df0 = df.sort_values(by='Date Time')
Nsamples = len(df0.index.values)
print(Nsamples)


df0 = df0.loc[df['Date Time'].apply(lambda x: x.minute) == 0]
Nsamples = len(df0.index.values)
print(Nsamples)


## Explore data
print('\nProperties:')
for prop in df.columns:
   print('  -',prop)
print('')

fig, ax = plt.subplots()
ax.plot(df['Date Time'], df['T (degC)'])
ax.set_xlabel('DateTime')
ax.set_ylabel('T (°C)')
fig.tight_layout()

fig, ax = plt.subplots()
X = [x.replace(year=2020) for x in df['Date Time']]
Y = df['T (degC)']
C = [x.year for x in df['Date Time']]
img = ax.scatter(X, Y, c=C, cmap='cool')
cbar = fig.colorbar(img) #, orientation='horizontal', shrink=0.8)
cbar.ax.set_ylabel('Year')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
ax.set_title('By Year')
ax.set_ylabel('T (°C)')

fig, ax = plt.subplots()
X = [x.time() for x in df['Date Time']]
C = [x.month for x in df['Date Time']]
Y = df['T (degC)']
img = ax.scatter(X, Y, c=C, cmap='cool')
cbar = fig.colorbar(img) #, orientation='horizontal', shrink=0.8)
cbar.ax.set_ylabel('Month')
ax.set_title('By Month')
ax.set_ylabel('T (°C)')


plt.show()

exit()



# Normalize!!
cols = df0.columns.values[1:]
df_num = df0[cols]
mean = df_num.mean()
std = df_num.std()
df = df0.copy()
df[cols] = (df[cols]-mean)/std

exit()

## Prepare Input/Output data
# We will use the weather of the past 5 days to predict the weather of the next
# 12 hours.
# The data is collected every 10 min (with some exceptions that we are going to
# ignore), that means that we need:
# Input: 5days * 24hours * 60hours / 10 min between samples = 720 samples
# Output: 0.5days * 24hours * 60hours / 10 min between samples = 72 samples
#                                            12h
#                      <------5 days------> <--->                    time
#     ----------------'--------------------|----------------------------->
#                     ~~~~~~~~~~~~~~~~~~~~ ^ ~~~~
#                     <--  input length -->| output length
#                                Forecast for this point

Nhistory = int(3*24*60/10)
Nfuture  = int(0.5*24*60/10)
columns_in = ['T (degC)','p (mbar)', 'wv (m/s)', 'wd (deg)']
columns_out = ['T (degC)']

## Train/Test Split
# Random split of the original data
print(f'\n\nTotal number of samples: {Nsamples}')
inds = np.array(range(len(df.index)))
np.random.shuffle(inds)
Ntrain = int(len(inds)*0.8)
Nvalid = int(len(inds)*0.199)
Ntest  = int(len(inds)*0.001)
train_ind = inds[:Ntrain]
valid_ind = inds[Ntrain:Ntrain+Nvalid]
test_ind  = inds[Ntrain+Nvalid:Ntrain+Nvalid+Ntest]
# train = df.iloc[inds[:Ntrain]]
# valid = df.iloc[inds[Ntrain:Ntrain+Nvalid]]
# test = df.iloc[inds[Ntrain+Nvalid:]]


def get_sample(df,ind,Nhistory,Nfuture,columns_in,columns_out):
   Nsamples = len(df.index)
   if ind-Nhistory > 0 and ind+Nfuture < Nsamples:
      inp = df.iloc[ind-Nhistory:ind][columns_in]
      out = df.iloc[ind:ind+Nfuture][columns_out]
      return inp.values, out.values
   else: return None,None


def prepare_dataset(df,inds,Nhistory,Nfuture,columns_in,columns_out):
   inps, outs = [],[]
   for ind in tqdm(inds):
      inp,out = get_sample(df,ind,Nhistory,Nfuture,columns_in,columns_out)
      if inp is None or out is None: continue
      inps.append( inp )
      outs.append( out )
   inps = np.array(inps)
   outs = np.array(outs)
   if len(inps.shape) == 1: inps = np.expand_dims(inps, axis=0)
   if len(outs.shape) == 1: outs = np.expand_dims(outs, axis=0)
   return inps,outs



train_inp,train_out = prepare_dataset(df, train_ind,
                                      Nhistory, Nfuture,
                                      columns_in, columns_out)
inp_shape = train_inp.shape[1:]
out_shape = train_out.shape[1:]
print('Train dataset:')
print(train_inp.shape)
print(train_out.shape)

valid_inp,valid_out = prepare_dataset(df, valid_ind,
                                      Nhistory, Nfuture,
                                      columns_in, columns_out)
print('Validation dataset:')
print(valid_inp.shape)
print(valid_out.shape)

test_inp,test_out = prepare_dataset(df, test_ind,
                                    Nhistory, Nfuture,
                                    columns_in, columns_out)
print('Test dataset:')
print(test_inp.shape)
print(test_out.shape)


rnn = True
model = models.Sequential()
if rnn: 
   fmodel = 'rnn.h5'
   model.add( LSTM(8, input_shape=inp_shape) )
else:
   fmodel = 'mlp.h5'
   model.add( Flatten(input_shape=inp_shape) )
   # model.add( Dense(500) )
model.add( Dense(100) )
model.add( Dense(len(columns_out)*Nfuture) )
model.add( Reshape((Nfuture,len(columns_out))) )

model.compile(optimizer='adam', loss='mae')
model.summary()

stopper = EarlyStopping(monitor='val_loss', patience=10,
                        min_delta=1e-9, baseline=0.198)
history = model.fit(train_inp, train_out, epochs=100,
                                          validation_data=(valid_inp,valid_out),
                                          verbose=1)
                                          # callbacks=[stopper])

model.save(fmodel)
err = history.history['loss']
val_err = history.history['val_loss']
fig, ax = plt.subplots()
ax.plot(err,label='loss')
ax.plot(val_err,label='val_los')
ax.legend()




print('\nTTTTEEEEEESSSSSSTTTTTTT!!!!!!')
outs = model.predict(test_inp)
print(outs.shape)
print(model.evaluate(test_inp,test_out))

fig = plt.figure()
n = 3
gs = gridspec.GridSpec(n, n)
axs = []
for i in range(n):
   for j in range(n):
      axs.append( fig.add_subplot(gs[i, j]) )


rand_inds = np.random.choice(range(len(outs)), int(n*n))
for i,ind in enumerate(rand_inds):
   Xinp = range(len(test_inp[ind]))
   Xout = [len(test_inp[ind]) + x for x in range(len(test_out[ind]))]
   Xout = [x for x in range(len(test_out[ind]))]
   # axs[i].plot(Xinp,test_inp[ind],label='In')
   axs[i].plot(Xout,test_out[ind],label='Out')
   axs[i].plot(Xout,outs[ind],label='NN')
axs[0].legend()
fig.savefig(fmodel.replace('.h5','.png'))

plt.show()
