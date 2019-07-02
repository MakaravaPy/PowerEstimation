# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:16:41 2016

@author: makara01
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

b11=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma0.1.txt') # 1oe phi-eto phi0 -guess, vtoroe phi- eto true phi
b12=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma0.5.txt')
b13=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma0.75.txt')
b14=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma1.0.txt')
b15=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma1.5.txt')
b16=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma1.25.txt')
b17=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma1.75.txt')
b18=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma2.0.txt')
b19=np.loadtxt('phi[1.1,-0.4]/Phi0-phi[1.2,-0.3]gamma10.0.txt')

b21=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma0.1.txt')
b22=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma0.5.txt')
b23=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma0.75.txt')
b24=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma1.0.txt')
b25=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma1.5.txt')
b26=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma1.25.txt')
b27=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma1.75.txt')
b28=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma2.0.txt')
b29=np.loadtxt('phi[1.1,-0.4]/Phi1-phi[1.2,-0.3]gamma10.0.txt')

data_phi1 = []
data_phi2 = []

data_phi1.append(b11)
data_phi1.append(b12)
data_phi1.append(b13)
data_phi1.append(b14)
data_phi1.append(b15)
data_phi1.append(b16)
data_phi1.append(b17)
data_phi1.append(b18)
data_phi1.append(b19)

data_phi2.append(b21)
data_phi2.append(b22)
data_phi2.append(b23)
data_phi2.append(b24)
data_phi2.append(b25)
data_phi2.append(b26)
data_phi2.append(b27)
data_phi2.append(b28)
data_phi2.append(b29)

bp1=axes[0].boxplot(data_phi1)
for box in bp1['boxes']:
    box.set( color='c', linewidth=1.5)

for whisker in bp1['whiskers']:
    whisker.set(color='c', linewidth=2)

for cap in bp1['caps']:
    cap.set(color='#7570b3', linewidth=2)

for median in bp1['medians']:
    median.set(color='m', linewidth=2)

for flier in bp1['fliers']:
    flier.set(marker='.', color='k', alpha=0.05)
    
axes[0].set_title('Phi 1',fontweight='bold', fontsize=20)
axes[0].axhline(y=1.2, color='y', linewidth=2)

bp2=axes[1].boxplot(data_phi2)
for box in bp2['boxes']:
    box.set( color='c', linewidth=1.5)
   
for whisker in bp2['whiskers']:
    whisker.set(color='c', linewidth=2)

for cap in bp2['caps']:
    cap.set(color='#7570b3', linewidth=2)

for median in bp2['medians']:
    median.set(color='m', linewidth=2)

for flier in bp2['fliers']:
    flier.set(marker='.', color='k', alpha=0.05)


axes[1].set_title('Phi 2',fontweight='bold', fontsize=20)
axes[1].axhline(y=-0.3, color='y', linewidth=2)



for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_phi1))])
    ax.set_xlabel('Gamma',fontweight='bold', fontsize=20)
    ax.set_ylabel('Data',fontweight='bold', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=13)



plt.setp(axes, xticks=[y+1 for y in range(len(data_phi1))],
         xticklabels=['0.1','0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0', '10.0'])

plt.figtext(0.91, 0.23, 'True phi',
            backgroundcolor='y', color='black', weight='roman', 
            size='x-large')         
  

       
plt.show()