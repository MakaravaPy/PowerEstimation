# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 14:39:01 2015

@author: makara01
"""
import numpy
from scipy import signal
from numpy.random import normal
from pylab import *


def ARMAgeneration(ar_coef,ma_coef,sigma,n,outcast=0, s=10):  #generate an random sample of an ARMA process
# ar_coef - array of length p with the AR(p) coefficients
# ma_coef - array of length q with the MA(q) coefficients 
# n  - length of simulated (returned) time series
# sigma - standard deviation of noise
# outcast - number of outcast datapoints (default = 100)
   
    seed(s)
    l=max(len(ar_coef),len(ma_coef))
    if(outcast==0):
      outcast=100
    noise=normal(0,sigma,n+outcast)
    ARMA=array([])
    signal=0.0
    l=max(len(ar_coef),len(ma_coef))
    for i in range(n+outcast):
        if(i<l):
          ARMA=append(ARMA,noise[i])
        else:
          signal=0.0
          for j in range(len(ar_coef)):
              signal=signal+ar_coef[j]*ARMA[i-j-1]
          for j in range(len(ma_coef)):
              signal=signal+ma_coef[j]*noise[i-j-1]
          ARMA=append(ARMA,signal+noise[i])
    return ARMA[outcast:]

