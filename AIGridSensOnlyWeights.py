# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:19:02 2019

@author: makara01
"""

from pypower.api import ppoption, runpf, makeYbus
from pypower.printpf import printpf
from pypower.ppoption import ppoption 
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I
from pypower.t.t_is import t_is 
from cmath import cos, sin 
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from scipy.io import loadmat     
from numpy import matlib                                                           
import numpy as np
from tools.ReadData import convert_mcase, convert_to_python_indices
from tools.Kalman import get_system_matrices,jacobian
from tools.DataPF import separate_Yslack
import profilesPV as ProfilePV
from matplotlib.pyplot import * 

fname = "Net1_UKGDS_60_subgrid.m"
fnameWhole = "Net1_UKGDS_60.m"
convert_mcase(fname)
convert_mcase(fnameWhole)
from Net1_UKGDS_60 import Net1_UKGDS_60_ as netWhole
from Net1_UKGDS_60_subgrid import Net1_UKGDS_60_ as net

mat = loadmat('MatpowerPQ.mat')
loadsP15to25 = mat['loadsP15to25']  # Active power (11,96)
loadsQ15to25 = mat['loadsQ15to25']  # Reactive power (11,96)
genP19 = mat['genP19']
genQ19 = mat['genQ19']



t_f = 96
t_ges = 1440
delta_t = 15
time= np.arange(delta_t, t_ges+delta_t,delta_t)
slack_idx=0
p0=1e-3
iterstop=50000000
accuracy=1e-9
roundY=None
baseMVA = 1000




################# Changing Loads ####################
filename1 = 'dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1) #(96,30)
location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, t_f-1, t_f), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = -PVdata[:,19] # in kVA 
loadsP15to25[6,:] = -PVdata[:,14] /1000 +loadsP15to25[6,:]# in kVA ***************
casedata = net()
bus_var = np.arange(2,13,1)  # buses that are varied 

casedataWhole = netWhole()
convert_to_python_indices(casedataWhole)
ppc = casedataWhole
ppopt = ppoption(PF_ALG=2)
ppc = ext2int(ppc)
baseMVA, bus, gen, branch = ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]#original grid which has 77 buses
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch) #build bus admittance matrix 
condition = np.hstack((1,np.arange(16,27,1))) 
Yws = Ybus[np.ix_(condition, condition)]

v_mag = np.zeros((13,t_f)) 
v_ang = np.zeros((13,t_f))
loadP_all = np.zeros((13,t_f))
loadQ_all = np.zeros((13,t_f))
genP_all = np.zeros((2,t_f))
genQ_all = np.zeros((2,t_f))
P_into_00 = np.zeros((1,t_f))
Q_into_00 = np.zeros((1,t_f))



for n in range(len(time)):
    casedata['bus'][bus_var,2] = loadsP15to25[:,n]  #Changing the values for the active power
    casedata['bus'][bus_var,3] = loadsQ15to25[:,n]  #Changing the values for the reactive power
    casedata['gen'][1,1] = LoadP[n]/(baseMVA*1000) #genP19[:,n]              #Changing the values for the gen-active power
    casedata['gen'][1,2] = 0#genQ19[:,n]              #Changing the values for the gen-reactive power
    ppopt = ppoption(PF_ALG=2)
    resultPF, success = runpf(casedata, ppopt)
    print 
    
    if success == 0:
        print ('ERROR in step %d', n) 
            
    
    slack_ang = resultPF['bus'][1,8]
    v_mag[:,n] = resultPF['bus'][:,7]               
    v_ang[:,n] = resultPF['bus'][:,8] - slack_ang   
    loadP_all[:,n] = resultPF['bus'][:,2]
    loadQ_all[:,n] = resultPF['bus'][:,3]
    genP_all[:,n] = resultPF['gen'][:,1]
    genQ_all[:,n] = resultPF['gen'][:,2]
    P_into_00[:,n]=-resultPF['branch'][0,15]
    Q_into_00[:,n]=-resultPF['branch'][0,16]
    
Pn = -loadP_all[2:,:]
Qn = -loadQ_all[2:,:]

# there is a generator at bus 5
Pn[4,:] = Pn[4,:]+genP_all[1,:]
Qn[4,:] = Qn[4,:]+genQ_all[1,:]
S= np.vstack((Pn,Qn))                       # power
S0 = np.vstack((P_into_00, Q_into_00))

n_K=S.shape[0]/2    #11
t_f = S.shape[1]    #96

ReV = (11/(np.sqrt(3)))*v_mag*np.cos(np.radians(v_ang))
ImV = (11/(np.sqrt(3)))*v_mag*np.sin(np.radians(v_ang))
V = np.vstack((ReV[2:,:],ImV[2:,:])) # voltage without slack bus
ReVs = 11/(np.sqrt(3))*v_mag[1,:]*np.cos(np.radians(v_ang[1,:]))
ImVs = 11/(np.sqrt(3))*v_mag[1,:]*np.sin(np.radians(v_ang[1,:]))
Vs = np.vstack((ReVs, ImVs)) #voltage of slack bus
V_slack=Vs  

Sfctemp = -S0/11
Pfc = Sfctemp[0,:]
Qfc = Sfctemp[1,:]
Sfc = np.vstack((matlib.repmat(Pfc,len(Pn[:,0]),1), matlib.repmat(Qfc,len(Qn[:,0]),1)))
    
volt_meas_idx  = [3,5,8,9,10]    # nodes at which voltage is measured (ignoring slack node)
actpow_meas_idx   = [6,8]    # nodes at which active power is measured 
reactpow_meas_idx = [6,8]     # nodes at which reactive power is measured 
Sfc[6,:] = FittedPV[:]

m = [mm-1 for mm in volt_meas_idx] 
vmeas = np.zeros(n_K,dtype=bool); vmeas[m] = True
pmeas = np.zeros(n_K,dtype=bool)
pmeas[[p-1 for p in actpow_meas_idx]] = True
qmeas = np.zeros(n_K,dtype=bool)
qmeas[[q-1 for q in reactpow_meas_idx]] = True 

Pres = S[:n_K,:][~pmeas,:] - Sfc[:n_K,:][~pmeas,:]
Qres = S[n_K:,:][~qmeas,:] - Sfc[n_K:,:][~qmeas,:]
data = np.vstack((Pres,Qres)) 
qvals = np.zeros(data.shape[0])



################################ simulated data

#nm=22 is the number of power equation(active power and reactive power)
nm = S.shape[0]


#### EKF for weights ###

p0=0.1   
beta = 0.9
Q = 1e-4*np.eye(nm)#shape (22,22); covariance matrix of weight equation
R = 1e-4*np.eye(nm)#shape (22,22); covariance matrix of power


def Sigmoid(x, beta):
    return 1/(1+np.exp(-beta * x))


def IteratedKalmanFilter_for_weights(beta,iterstop=5, accuracy=1e-9):
    
    w_est=0.05*np.ones((nm,t_f+1))
    x_est =np.zeros((nm, t_f+1))
       
    P = p0*np.eye(nm)#shape (22,22); conditional mean covariance matrix
    for k in range(1,t_f):             
        wf = w_est[:,k-1]      
        xf = x_est[:,k-1]
        
        eta = wf
        est_error1 = 1
        counter1 = 1
        temp1 = np.zeros((nm))                           
        while ((est_error1 > accuracy) and (counter1 < iterstop)):
            Pf = P + Q
            H = np.diagflat(Sigmoid(xf,beta))#shape(22,22)
            K=np.dot(np.dot(Pf,H.T),np.linalg.pinv(np.dot(H,np.dot(Pf,H.T))+R))#shape(22,22)
            temp1 = eta                      
            eta =  eta+ np.dot(K,(S[:,k-1] -np.dot(np.diagflat(wf),Sigmoid(xf,beta))))
            est_error1 = np.linalg.norm(temp1-eta)
            counter1 += 1
                             
            w_est[:,k]= eta
            xf = np.dot(np.diagflat(eta),Sigmoid(xf,beta)) 
            x_est[:,k] = xf                                    
            P = np.dot(np.eye(nm)-np.dot(K,H),Pf)

    return w_est, x_est 
    
w_est, x_est = IteratedKalmanFilter_for_weights(beta,iterstop=5,accuracy=1e-10)    


figure(1); clf()  
for k in range(n_K):
    subplot(3,4,k+1)  
    ylabel('Power', fontsize=5, fontweight='bold')
    plot(time,S[k,:],'b-o',linewidth=3.5,label="true", ms=2)
    plot(time,x_est[k,1:],'r-o',linewidth=3.5,label="estimated", ms=0.1)
    legend(loc="upper left", fontsize=5)
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45, fontsize=5)
    yticks(fontweight='bold',  fontsize=5)
    xlabel('Time [h]', fontsize=5, fontweight='bold')
    title("node %d" %(k+1), fontweight='bold')
    grid()
figure(2); clf()  
for k in range(n_K):
    subplot(3,4,k+1)  
    ylabel('Weight', fontsize=5, fontweight='bold')
    plot(time,w_est[k,1:],'r-o',linewidth=3.5,label="estimated", ms=2)
    legend(loc="upper left", fontsize=0.1)
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45, fontsize=5)
    yticks(fontweight='bold',  fontsize=5)
    xlabel('Time [h]', fontsize=5, fontweight='bold')
    title("node %d" %(k+1), fontweight='bold')
    grid()
    
figure(3); clf()  
for k in range(n_K):
    subplot(3,4,k+1)  
    ylabel('Error', fontsize=5, fontweight='bold')
    plot(time,x_est[k,1:]-S[k,:],'r-o',linewidth=3.5,label="estimated", ms=1)
    legend(loc="upper left", fontsize=0.5)
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45, fontsize=5)
    yticks(fontweight='bold',  fontsize=5)
    xlabel('Time [h]', fontsize=5, fontweight='bold')
    title("node %d" %(k+1), fontweight='bold')
    grid() 
