# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:16:41 2016

@author: makara01
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(5, 5))

b11=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.0,change]gamma0.1.txt') 
b12=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.1,change]gamma0.1.txt')
b13=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.2,change]gamma0.1.txt')
b14=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.3,change]gamma0.1.txt')
b15=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.4,change]gamma0.1.txt')
b16=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.5,change]gamma0.1.txt')
b17=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.6,change]gamma0.1.txt')
b18=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.7,change]gamma0.1.txt')
b19=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.8,change]gamma0.1.txt')
b110=np.loadtxt('Case2fixgamma_ARMAOnline/Phi1/Phi1-phi[1.9,change]gamma0.1.txt')


data_phi1 = []
data_phi2 = []
data_phi3 = []
data_phi4 = []
data_phi5 = []
data_phi6 = []
data_phi7 = []
data_phi8 = []
data_phi9 = []
data_phi10 = []

data_f1 = []
data_f2 = []
data_f3 = []
data_f4 = []
data_f5 = []
data_f6 = []
data_f7 = []
data_f8 = []
data_f9 = []
data_f10 = []


nk=10

for i in range(nk):
        data_phi1.append(b11[0+i:80+i:10]) # every 10 elements over nodes, nodes = 8
        data_phi2.append(b12[0+i:80+i:10])
        data_phi3.append(b13[0+i:80+i:10])
        data_phi4.append(b14[0+i:80+i:10])
        data_phi5.append(b15[0+i:80+i:10])
        data_phi6.append(b16[0+i:80+i:10])
        data_phi7.append(b17[0+i:80+i:10])
        data_phi8.append(b18[0+i:80+i:10])
        data_phi9.append(b19[0+i:80+i:10])
        
        
for i in range(1,nk):        
      
        data_f1.append(locals()["data_phi"+str(i)][0]) # teper razbivaju otnositelno phi pri fixirovanom phi2
        data_f2.append(locals()["data_phi"+str(i)][1])
        data_f3.append(locals()["data_phi"+str(i)][2])
        data_f4.append(locals()["data_phi"+str(i)][3])
        data_f5.append(locals()["data_phi"+str(i)][4])
        data_f6.append(locals()["data_phi"+str(i)][5])
        data_f7.append(locals()["data_phi"+str(i)][6])
        data_f8.append(locals()["data_phi"+str(i)][7])
        data_f9.append(locals()["data_phi"+str(i)][8])
        data_f10.append(locals()["data_phi"+str(i)][9])


for i in range(nk):
        
        axes[0].axhline(y=1.0, color='c', linewidth=2)
        axes[1].axhline(y=1.1, color='c', linewidth=2)
        axes[2].axhline(y=1.2, color='c', linewidth=2)
        axes[3].axhline(y=1.3, color='c', linewidth=2)
        axes[4].axhline(y=1.4, color='c', linewidth=2)
        axes[5].axhline(y=1.5, color='c', linewidth=2)
        axes[6].axhline(y=1.6, color='c', linewidth=2)    
        axes[7].axhline(y=1.7, color='c', linewidth=2)
        axes[8].axhline(y=1.8, color='c', linewidth=2)
        axes[9].axhline(y=1.9, color='c', linewidth=2)        
        
        bp1=axes[0].boxplot(data_f1)
        bp2=axes[1].boxplot(data_f2)
        bp3=axes[2].boxplot(data_f3)
        bp4=axes[3].boxplot(data_f4)
        bp5=axes[4].boxplot(data_f5)
        bp6=axes[5].boxplot(data_f6)
        bp7=axes[6].boxplot(data_f7)
        bp8=axes[7].boxplot(data_f8)
        bp9=axes[8].boxplot(data_f9)
        bp10=axes[9].boxplot(data_f10)
        
#     
        axes[0].set_title('Estimation of Beta 1',fontweight='bold', fontsize=20)
        
        for ax in axes:
            ax.set_xticks([y+1 for y in range(len(data_f1))])
            ax.set_yticks([])            
            #ax.set_xticklabels([])
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=13)
        
        
        
        
        axes[0].set_ylabel('1.0',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[1].set_ylabel('1.1',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[2].set_ylabel('1.2',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[3].set_ylabel('1.3',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[4].set_ylabel('1.4',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[5].set_ylabel('1.5',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[6].set_ylabel('1.6',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[7].set_ylabel('1.7',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[8].set_ylabel('1.8',fontweight='bold', fontsize=15, rotation='horizontal')
        axes[9].set_ylabel('1.9',fontweight='bold', fontsize=15, rotation='horizontal')
        
        plt.rc('font', weight='bold')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.xlabel('Beta 2',fontweight='bold', fontsize=20)  
        plt.setp(axes, xticks=[y+1 for y in range(len(data_f1))],
                 xticklabels=['-0.9','-0.8', '-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1'])
        plt.figtext(0.0, 0.53, 'Beta 1',
                    #backgroundcolor='y', color='black', weight='roman', 
                    size='x-large')    
        plt.grid(True)
            
        plt.show()