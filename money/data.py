#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Pandas
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
from matplotlib import gridspec


date_fmt = '%Y/%m/%d-%H:00'

def fix_df(df):
   """Remove dupliacte entries. precision is set by delta (in minutes)"""
   clean_date = lambda t: t.replace(minute=0, second=0, microsecond=0)
   df.index = df.index.map(clean_date)
   df = df.loc[~df.index.duplicated(keep='first')]
   # df.to_csv(f"data/{fname.split('/')[-1]}", date_format=date_fmt)
   return df.sort_index()


def read_data(fname,old=False,remove_duplicates=True):
   if old: delimiter = '   '
   else: delimiter = None
   df = pd.read_csv(fname, delimiter=delimiter, header=0, engine='python',
                           names=['Dates','eur2usd'], parse_dates=True,
                           index_col=0)
   df['usd2eur'] = 1/df['eur2usd']
   if remove_duplicates: fix_df(df)
   return df.sort_index()


def write_data(df,fname):
   df = df.sort_index()
   df.to_csv(fname, date_format=date_fmt, index=True, columns=['eur2usd'])


def col2pd(X,Y):
   df = pd.DataFrame({'Dates':X, 'eur2usd':Y})
   df['usd2eur'] = 1/df['eur2usd']
   df = df.set_index('Dates')
   return df.sort_index()
   

def add_row(df,date,rate):
   df = df.append(pd.DataFrame({'eur2usd':rate, 'usd2eur':1/rate},index=[date]))
   df = fix_df(df)
   df.index.name = 'Dates'
   return df.sort_index()
