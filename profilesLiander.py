# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:13:40 2016

@author: makara01
"""

import numpy as np

def ProfilePowerLiander(Type, Season):
    
###
#   Function to set up the power values (PV) depended on:
#       - type (type I, type II, type III)
#       - season (winter, spring, summer, autumn)
###    

    if Type == 'Type I':
       if Season == 'Winter':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/WinterTypeI.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Spring':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/SpringTypeI.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Summer':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/SummerTypeI.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Autumn':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/AutumnTypeI.dat'
          profile = np.loadtxt(filename1)       

    if Type == 'Type II':
       if Season == 'Winter':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/WinterTypeII.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Spring':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/SpringTypeII.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Summer':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/SummerTypeII.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Autumn':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/AutumnTypeII.dat'
          profile = np.loadtxt(filename1)    
                                
    if Type == 'Type III':
       if Season == 'Winter':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/WinterTypeIII.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Spring':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/SpringTypeIII.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Summer':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/SummerTypeIII.dat'
          profile = np.loadtxt(filename1)      
       if Season == 'Autumn':
          filename1 = '/Users/makara01/Documents/oioi/DataLiander/AutumnTypeIII.dat'
          profile = np.loadtxt(filename1)    
               
    return profile

