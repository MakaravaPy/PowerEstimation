# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:16:41 2016

@author: makara01
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
import pylab
import re
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

class LogFormatterTeXExponent(pylab.LogFormatter, object):
	    def __init__(self, *args, **kwargs):
	        super(LogFormatterTeXExponent, self).__init__(*args, **kwargs)
    
	    def __call__(self, *args, **kwargs):
	        label = super(LogFormatterTeXExponent, self).__call__(*args, **kwargs)
	        label = re.sub(r'e(\S)0?(\d+)', r'\\times 10^{\1\2}', str(label))
	        label = "$" + label + "$"
	        return label



#fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(5, 5))

b11=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.9]gamma0.1.txt') 
b12=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.8]gamma0.1.txt')
b13=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.7]gamma0.1.txt')
b14=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.6]gamma0.1.txt')
b15=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.5]gamma0.1.txt')
b16=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.4]gamma0.1.txt')
b17=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.3]gamma0.1.txt')
b18=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.2]gamma0.1.txt')
b19=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_S/phi[change,-0.1]gamma0.1.txt')


b21=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.9]gamma0.1.txt') 
b22=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.8]gamma0.1.txt')
b23=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.7]gamma0.1.txt')
b24=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.6]gamma0.1.txt')
b25=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.5]gamma0.1.txt')
b26=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.4]gamma0.1.txt')
b27=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.3]gamma0.1.txt')
b28=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.2]gamma0.1.txt')
b29=np.loadtxt('Case2fixgamma_ARMAOnline/RMSE_No_Online/phi[change,-0.1]gamma0.1.txt')

n_K=11
data_s = np.zeros((9,10,n_K))
data_s_no = np.zeros((9,10,n_K))
#
for i in range(9):              
        data_s[i,:,:] = locals()["b1"+str(i+1)]

for i in range(9):              
        data_s_no[i,:,:] = locals()["b2"+str(i+1)]
        



fig = plt.figure()
im = plt.matshow(np.log(data_s[:,:,5]), norm=LogNorm(vmin=data_s[:,:,5].min(), vmax=data_s[:,:,5].max()))
plt.matshow(np.log(data_s[:,:,5]))
plt.colorbar(im)
plt.ylabel(r'$\varphi_2$', fontsize=25,fontweight='bold') 
plt.xlabel(r'$\varphi_1$',fontweight='bold', fontsize=25)
ax=plt.gca()
ax.xaxis.set_ticks(np.arange(0,10))
ax.yaxis.set_ticks(np.arange(0,9))
ax.yaxis.set_ticklabels(['-0.9','-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1'])
ax.xaxis.set_ticklabels(['1.0','1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9'])





fig = plt.figure()
im = plt.matshow(np.log(data_s_no[:,:,5]),  norm=LogNorm(vmin=data_s_no[:,:,5].min(), vmax=data_s_no[:,:,5].max()))
plt.matshow(np.log(data_s_no[:,:,5]))
plt.colorbar(im)
plt.ylabel(r'$\varphi_2$', fontsize=25,fontweight='bold') 
plt.xlabel(r'$\varphi_1$',fontweight='bold', fontsize=25)
ax=plt.gca()
ax.xaxis.set_ticks(np.arange(0,10))
ax.yaxis.set_ticks(np.arange(0,9))
ax.yaxis.set_ticklabels(['-0.9','-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1'])
ax.xaxis.set_ticklabels(['1.0','1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9'])

