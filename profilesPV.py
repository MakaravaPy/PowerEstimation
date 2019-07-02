# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:40:00 2014

@author: makara01
"""

import numpy as np
from scipy import asarray as ar,exp
import random

def ProfilePower(location, Season):
    
###
#   Function to set up the power values (PV) depended on:
#       - location (high, normal, low)
#       - season (winter, spring, summer, autumn)
###    
  #  originalMVA = 2.2
  #  Amplitude values are in kVA!!!!!
    if location == 'High':
       if Season == 'Winter':
                Ampl = 160
                mu = 51
                std = 6
       if Season == 'Spring':
                Ampl = 553
                mu = 53
                std = 9
       if Season == 'Summer':
                Ampl = 600
                mu = 55
                std = 10
       if Season == 'Autumn':
                Ampl = 292
                mu = 52
                std = 7

    if location == 'Normal':
       if Season == 'Winter':
                Ampl = 85
                mu = 50
                std = 5
       if Season == 'Spring':
                Ampl = 309
                mu = 52
                std = 8
       if Season == 'Summer':
                Ampl = 344
                mu = 53
                std = 9
       if Season == 'Autumn':
                Ampl = 162
                mu = 51
                std = 6
                
    if location == 'Low':
       if Season == 'Winter':
                Ampl = 3
                mu = 49
                std = 4
       if Season == 'Spring':
                Ampl = 24
                mu = 53
                std = 7
       if Season == 'Summer':
                Ampl = 26
                mu = 54
                std = 7
       if Season == 'Autumn':
                Ampl = 11
                mu = 50
                std = 5
    return Ampl, mu, std

random.seed()    
def gaussian(x, Ampl, baseMVA, mu, sig, location):
###
#   Function for creating PV pseudo-data based on: 
#      - gaussian function woth noise
#   depended on: 
#      - location
#       - season
###    
    n = len(x)
    np.random.seed(99)
    Win = Ampl * np.exp(-(x-mu)**2/(2*sig**2))
    if location == 'High':
        noise = np.random.normal(0,35, n)
        for i in range(n):
                if Win[i] < 0.1:
                    noise[i] = 0.0
    if location == 'Normal':
        noise = np.random.normal(0,20, n)
        for i in range(n):
                if Win[i] < 0.1:
                    noise[i] = 0.0    
    if location == 'Low':
        noise = np.random.normal(0, 2, n)
        for i in range(n):
                if Win[i] < 0.1:
                    noise[i] = 0.0             
    fittedPV = Win+abs(noise)  
    return fittedPV/1000