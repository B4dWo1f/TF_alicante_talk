#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from numpy.random import uniform as rand

def lineal_norm(x):
   """ maps an input vector x to the interval [0,1] """
   return (x-np.min(x))/(np.max(x)-np.min(x))


def gen_data(N,noise=0.1, norm=True):
   """
   Returns N samples of the sin(x) function.
   noise to add noise: y = sin(x) + noise*random(-1,1)
   """
   x = rand(0,2*np.pi, N)
   y = np.sin(x) + noise*rand(-1,1,len(x))
   if norm:
      x = lineal_norm(x)
      y = lineal_norm(y)
   return x, y


def gen_data_cool(N,noise=0.1, norm=True):
   """
   Returns N samples of the sin(x) function.
   noise to add noise: y = sin(x) + noise*random(-1,1)
   """
   x0 = rand(-8,0,N)
   x1 = rand(0,3*np.pi,N)

   y0 = x0*x0/10 + noise*rand(-1,1,len(x0))
   y1 = np.cos(x1/2-np.pi)+1 + noise*rand(-1,1,len(x1))


   x = np.concatenate([x0, x1])
   y = np.concatenate([y0, y1])
   if norm:
      x = lineal_norm(x)
      y = lineal_norm(y)
   return x, y