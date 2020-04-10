#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import shuffle
import data
import pandas as pd

class NotEnoughData(Exception):
   """
   Custom exception to report missing data from the df
   """
   pass


Nhistory = 30
Nnext = 2
columns = ['eur2usd','usd2eur']
fmodel = 'eur2usd.h5'
fdata = 'data/ecb.dat'

## Read data
df = data.read_data(fdata)
dates = df.index
# Fix missing days
idx = pd.date_range(dates[0], dates[-1], freq = "D")
df = df.reindex(idx,method='nearest')   #XXX this should be interpolation
df.index = pd.DatetimeIndex(df.index)
df = df[columns]
print(df.describe())

# Normalize data
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
df_scaled = deepcopy(df)
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df) 
df_scaled.loc[:,:] = scaled_values


def get_inp_out(df,date,Ninp,Nout,columns):
   """
   df: dataframe containing all the information
   date: date to build the input-output pair
   Ninp: Number of points before date for the input
   Nout: Number of points to use as output
   """
   day = dt.timedelta(days=1)
   inp = df.loc[date-(Ninp-1)*day:date][columns]
   out = df.loc[date+day:date+Nout*day][columns]
   if len(inp) != Ninp or len(out) != Nout:
      raise NotEnoughData
   else: return inp.values, out.values


data_dates,X,Y = [],[],[]
for i,data in df_scaled.iterrows():
   try: x,y = get_inp_out(df_scaled,i,Nhistory,Nnext,columns)
   except: continue
   X.append(x)
   Y.append(y)
   data_dates.append(i)


## Save last element for testing
final_date = data_dates[-1]
final_X = X[-1]
final_Y = Y[-1]

data_dates = data_dates[:-1]
X = X[:-1]
Y = Y[:-1]

print(data_dates[0],'<-->',data_dates[-1])

x_data = np.array(X)
y_data = np.array(Y)

inds = list(range(len(x_data)))
shuffle(inds)
validation_split = 0.1
test  = np.array(inds[:int(validation_split*len(inds))])
train = np.array(inds[int(validation_split*len(inds)):])

# x_train = np.expand_dims(x_data[train], axis=2)
x_train = x_data[train]
y_train = y_data[train]  #np.expand_dims(y_data[train], axis=1)
# x_test  = np.expand_dims(x_data[test], axis=2)
x_test  = x_data[test]
y_test  = y_data[test]  #np.expand_dims(y_data[test], axis=1)


print('*************')
print('All data:',x_data.shape, y_data.shape)
print('Training:',x_train.shape, y_train.shape)
print('Testing :',x_test.shape, y_test.shape)
print('*************')

#####
# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping

class ExternalStop(Callback):
   def on_epoch_end(self, epoch, logs={}):
      if os.path.isfile('STOP'):
         print('\n\nExternal stop')
         print(epoch)
         self.model.stop_training = True



try:
   model = load_model(fmodel)
   print(f'Loaded model: {fmodel}')
   model.summary()
except OSError:
   print('New model')
   model = Sequential()
   model.add( LSTM(Nhistory, input_shape=x_train.shape[1:],
                             activation='tanh',
                             recurrent_activation='tanh') )
   model.add( Dense(40, activation='tanh') )
   model.add( Dense(np.prod(y_train.shape[1:]), activation='tanh') )
   model.add( Reshape(y_train.shape[1:]) )
   model.compile(optimizer='Adam', loss='mae', metrics=['mse'])

   model.summary()

   # Callbacks
   Stopper = ExternalStop()
   Early = EarlyStopping(min_delta=1e-4, patience=90,verbose=2,
                         restore_best_weights=True)
   hist = model.fit(x_train,y_train, epochs=900,
                                     # steps_per_epoch=797,
                                     validation_data=(x_test,y_test),
                                     verbose=1,
                                     callbacks=[Stopper,Early] )

   if True:
      metrics,values = [],[]
      for k,v in hist.history.items():
         metrics.append(k)
         values.append(v)

      fig = plt.figure()  #figsize=(20,10))
      gs = gridspec.GridSpec(len(metrics), 1)
      fig.subplots_adjust(wspace=0.,hspace=0.15)
      axs = []
      for i in range(len(metrics)):
         if i == 0: axs.append(plt.subplot(gs[i]))  # Original plot
         else: axs.append( plt.subplot(gs[i], sharex=axs[0]) )  # dists

      for i in range(len(metrics)):
         ax = axs[i]
         label = metrics[i]
         val = values[i]
         ax.plot(val, label=label)
         ax.legend()
         ax.set_ylim(ymin=0)
   print('\n\nTraining done.')

print('Training data range:',data_dates[0],'<-->',data_dates[-1])
Y_pred = model.predict( np.expand_dims(final_X,axis=0) )[0]
Y_pred = scaler.inverse_transform(Y_pred)
final_Y = scaler.inverse_transform(final_Y)

print(final_date.date(),'','       '.join([*columns]))
print('            pred (real)   pred (real)')
Xx,Yy1,Yy2 = [],[],[]
for i in range(final_Y.shape[0]):
   txt = str((final_date+(i+1)*dt.timedelta(days=1)).date())
   for y1,y2 in zip(Y_pred[i],final_Y[i]):
      txt += f'  {y1:.3f}({y2:.3f})'
   Xx.append((final_date+(i+1)*dt.timedelta(days=1)).date())
   Yy1.append(y1)   # real
   Yy2.append(y2)   # prediction
   # txt += '\n'
   print(txt)

fig, ax = plt.subplots()
# Column 1
ax.plot(df.index,df['eur2usd'])
ax.scatter(Xx,final_Y[:,0])
ax.scatter(Xx,Y_pred[:,0])
# Column 2
ax.plot(df.index,df['usd2eur'])
ax.scatter(Xx,final_Y[:,1])
ax.scatter(Xx,Y_pred[:,1])
ax.axvline(final_date,color='k',ls='--')
plt.show()
print('---------')
model.save(fmodel)
