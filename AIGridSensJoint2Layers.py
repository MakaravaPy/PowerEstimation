# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:47:26 2019

@author: makara01
"""


from pypower.api import ppoption, runpf, makeYbus
from pypower.ext2int import ext2int
import time as t
from scipy.io import loadmat
import numpy as np
from numpy import matlib
from matplotlib.pyplot import figure, plot, clf, subplot, xticks, yticks, legend,\
    xlabel, ylabel, grid, title, subplots_adjust

from GridSens.NLO.Kalman import get_system_matrices, jacobian
from GridSens.tools.data_tools import separate_Yslack
from GridSens.NLO import misc
from GridSens.tools.load import convert_mcase, convert_to_python_indices
import GridSens.tools.profilesPV as ProfilePV

################# Load grid ########################
mat = loadmat('MatpowerPQ.mat')
loadsP15to25 = mat['loadsP15to25']  # Active power  
loadsQ15to25 = mat['loadsQ15to25']  # Reactive power
genP19 = mat['genP19']
genQ19 = mat['genQ19']

mat1 = loadmat('NLO_data.mat')
Y=mat1['Y']

fname = "Net1_UKGDS_60_subgrid.m"
fnameWhole = "Net1_UKGDS_60.m"
convert_mcase(fname)
convert_mcase(fnameWhole)
from Net1_UKGDS_60_subgrid import Net1_UKGDS_60_ as net
from Net1_UKGDS_60 import Net1_UKGDS_60_ as netWhole

t_f = 96 # time variation
t_ges = 1440  # all time in min
delta_t = 15  # time intervals in min
time= np.arange(delta_t, t_ges+delta_t,delta_t)

casedataWhole = netWhole()
convert_to_python_indices(casedataWhole)
ppc = casedataWhole
ppopt = ppoption(PF_ALG=2)
ppc = ext2int(ppc)
baseMVA, bus, gen, branch = ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
baseMVA = 1000
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch) # nodal admittance matrix Ybus
condition = np.hstack((1,np.arange(16,27,1))) 
Yws = Ybus[np.ix_(condition, condition)]

filename1 = 'dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1)

location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, t_f-1, t_f), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = -PVdata[:,19] # in kVA 

################# Change loads ####################
loadsP15to25[6,:] = -PVdata[:,14] /1000 +loadsP15to25[6,:]# in kVA
casedata = net()
bus_var = np.arange(2,13,1)  # buses that are varied 


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
    #casedata['bus'][3,2] = LoadP[n]          #Changing the values for PV values (active power), reactive==0 ???
    casedata['gen'][1,1] = LoadP[n]/(baseMVA*1000) #genP19[:,n]              #Changing the values for the gen-active power
    casedata['gen'][1,2] = genQ19[:,n]              #Changing the values for the gen-reactive power
    ppopt = ppoption(PF_ALG=2)
    resultPF, success = runpf(casedata, ppopt)
    print 
    
    if success == 0:
        print ('ERROR in step %d', n) 
            
    
    slack_ang = resultPF['bus'][1,8]
    v_mag[:,n] = resultPF['bus'][:,7]               # Voltage, magnitude
    v_ang[:,n] = resultPF['bus'][:,8] - slack_ang   # Voltage, angle
    loadP_all[:,n] = resultPF['bus'][:,2]
    loadQ_all[:,n] = resultPF['bus'][:,3]
    genP_all[:,n] = resultPF['gen'][:,1]
    genQ_all[:,n] = resultPF['gen'][:,2]
    P_into_00[:,n]=-resultPF['branch'][0,15]
    Q_into_00[:,n]=-resultPF['branch'][0,16]

  
    
Pn = -loadP_all[2:,:]
Qn = -loadQ_all[2:,:]
Pn[4,:] = Pn[4,:]+genP_all[1,:]
Qn[4,:] = Qn[4,:]+genQ_all[1,:]
S= np.vstack((Pn,Qn))                       # power
S0 = np.vstack((P_into_00, Q_into_00))

#V: voltage without slack bus; Vs: voltage of the slack bus
ReV = (11/(np.sqrt(3)))*v_mag*np.cos(np.radians(v_ang))
ImV = (11/(np.sqrt(3)))*v_mag*np.sin(np.radians(v_ang))
V = np.vstack((ReV[2:,:],ImV[2:,:]))
ReVs = 11/(np.sqrt(3))*v_mag[1,:]*np.cos(np.radians(v_ang[1,:]))
ImVs = 11/(np.sqrt(3))*v_mag[1,:]*np.sin(np.radians(v_ang[1,:]))
Vs = np.vstack((ReVs, ImVs))

y = np.r_[ReV[2:,:],ImV[2:,:]]

Sfctemp = -S0/11
Pfc = Sfctemp[0,:]
Qfc = Sfctemp[1,:]
Sfc = np.vstack((matlib.repmat(Pfc,len(Pn[:,0]),1), matlib.repmat(Qfc,len(Qn[:,0]),1)))

################# Configure the estimation task ####################
nm = S.shape[0]
n_K = S.shape[0]/2
t_f = S.shape[1]
slack_idx=0
p0=1e-3
iterstop=50
accuracy=1e-9
roundY=None
volt_meas_idx  = [2,3,9,11]      # nodes at which voltage is measured (ignoring slack node)
actpow_meas_idx   = [1,5,8]    # nodes at which active power is measured 
reactpow_meas_idx = [1,5,8]    # nodes at which reactive power is measured

Sfc[6,:] = FittedPV[:] # with MVA 
#n_up is the the number of nodes with unmeasured power
n_up=nm-len(actpow_meas_idx)-len(reactpow_meas_idx)

S_vorst = Sfc

m = [mm-1 for mm in volt_meas_idx] 
vmeas = np.zeros(n_K,dtype=bool); vmeas[m] = True
pmeas = np.zeros(n_K,dtype=bool)
pmeas[[p-1 for p in actpow_meas_idx]] = True
qmeas = np.zeros(n_K,dtype=bool)
qmeas[[q-1 for q in reactpow_meas_idx]] = True 

Pres = S[:n_K,:][~pmeas,:] - S_vorst[:n_K,:][~pmeas,:]
Qres = S[n_K:,:][~qmeas,:] - S_vorst[n_K:,:][~qmeas,:]
data = np.vstack((Pres,Qres))

################### Kalman Filter ##############################
DeltaS_est = np.zeros((n_up,t_f))    
UncDeltaS = np.zeros_like(DeltaS_est)    
S_est = np.zeros((nm,t_f))

#Cm: shape(2*len(volt_meas_idx), 2*n_k) and composed of 0/1; the matrix mapping all voltage to measured voltage
#Dnm: shape(2*n_k, n_up); the matrix mapping unmeasured power to all power
#Dm: shape(2*n_k, len(actpow_meas_idx+reactpow_meas_idx)); the matrix mapping measured power to all power
Cm, Dnm, Dm = get_system_matrices(pmeas,qmeas,vmeas)
#Sg (len(actpow_meas_idx+reactpow_meas_idx),96) is measured power
#Sf (n_up, 96) is forcasted (unmeasured) power
#Sm is the combination of Sg and Sf
Sm, Sg, Sf = misc.PowerInputSim(Dm, Dnm, S, S_vorst,pmeas,qmeas)

Ywo_slack,Y_slack = separate_Yslack(Yws,slack_idx=0) 

y0 = np.zeros((len(volt_meas_idx),t_f))
y1 = np.zeros((len(volt_meas_idx),t_f))
for k,ind in enumerate(vmeas.nonzero()[0]):
    y0[k,:] = V[ind,:]
    y1[k,:] = V[ind+n_K,:]
y = np.vstack((y0,y1)) 

def Sigmoid(x, beta=0.9):
    return 1/(1+np.exp(-beta * x))

#mu(22,1),MU(22,22)
def calcM(mu):
    divisor = 3*(mu[:n_K]**2 + mu[n_K:]**2)
    return np.r_[np.c_[np.diag(mu[:n_K]/divisor), np.diag(mu[n_K:]/divisor)],
				     np.c_[np.diag(mu[n_K:]/divisor), -np.diag(mu[:n_K]/divisor)]]

def transition(xf,wf):
    xfsig = Sigmoid(xf)
    F = np.diagflat(wf*xfsig*(1-xfsig))
    return F

def transition2(xf,wf):
    F1 = transition(xf, wf[:n_up])
    mid = wf[:n_up]*Sigmoid(xf)
    F2 = transition(mid, wf[n_up:])
    F = np.dot(F2, F1)
    return F

def weight_jacobian(Dnm,xf):
    xfsig = Sigmoid(xf)
    xfsig_diag = np.diagflat(xfsig)
    H = np.dot(Dnm,xfsig_diag)
    return H

def weight_jacobian2(Dnm,xf,eta_w):
    mid = eta_w[:n_up]*Sigmoid(xf)
    H2 = weight_jacobian(Dnm,mid)
    F2 = transition(mid, eta_w[n_up:])
    H1 = np.dot(weight_jacobian(Dnm, xf),F2)
    H = np.concatenate((H1, H2), axis=1)
    return H
 
def JointIteratedKalmanFilter(y,Sm,Yws,Vs,iterstop=50,accuracy=1e-9):
    
    p0=1.0
    P = p0*np.eye(n_up)
    Q = 1e-2*np.eye(n_up)
    R = 1e-2*np.eye(2*len(vmeas.nonzero()[0]))
    
    Y00,Ys = separate_Yslack(Yws,slack_idx,dcomplex=False,prec=roundY)
    Slack = np.linalg.solve(Y00,np.dot(Ys,Vs)).__array__()

    muiter=np.zeros((nm))
    temp1 = np.zeros((n_up))
    temp2 = np.zeros((nm))
    
    V_est = np.zeros((2*n_K,t_f+1)) 
    v0 = 11/(np.sqrt(3))*np.hstack((np.ones(n_K), np.zeros(n_K)))
    V_est[:,0] = v0

    x_est= 0*np.ones((n_up,t_f+1))
    w_est= -0.004*np.ones((2*n_up,t_f+1))        
        
    mu = np.hstack((v0[:n_K]*np.cos(v0[n_K:]),v0[:n_K]*np.sin(v0[n_K:])))
    
    #kalman filter paras for weight estimation
    Pw = p0*np.eye(2*n_up)#shape (n_up,n_up); conditional mean covariance matrix
    Qw = 1e-4*np.eye(2*n_up)#covariance matrix of weight equation; n_up<=22
    
    t0=t.time()
    for k in range(1,t_f+1):
        
        wf = w_est[:,k-1]
        ##################################
        ## Weight Estimation
        ##################################
        eta_w = wf
        est_error3 = 1
        temp_w = np.zeros((nm))
        counter3 = 1
        Pwf = Pw + Qw
        
        while(est_error3 > accuracy) and (counter3 < 500):
            
            eta = eta_w[n_up:]*Sigmoid(eta_w[:n_up]*Sigmoid(x_est[:,k-1]))
            MU= calcM(mu)
            muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1]+np.dot(Dnm,eta)) - Slack[:,k-1]
            est_error2 = 1
            counter2 = 1
            while  (est_error2 > accuracy) and (counter2 < iterstop):
                    MU= calcM(muiter)
                    temp2 = muiter
                    muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1] + np.dot(Dnm,eta)) - Slack[:,k-1]
                    est_error2 = np.linalg.norm(muiter - temp2)
                    counter2 += 1
                   
            mu = muiter
            
            Dh = jacobian(Y00,Ys,mu,Vs[:,k-1])
            H = np.dot(Cm,np.dot(np.linalg.inv(Dh),Dnm))
            Hw = weight_jacobian2(H, x_est[:,k-1], eta_w)#shape(len(vol), n_up)
            Kw = np.dot(np.dot(Pwf,Hw.T),np.linalg.pinv(np.dot(Hw,np.dot(Pwf,Hw.T))+R))#shape(n_up,22)

            temp_w = eta_w                          
            eta_w =  eta_w + np.dot(Kw,y[:,k-1] - np.dot(Cm,mu) - np.dot(Hw,wf))
            est_error3 = np.linalg.norm(temp_w - eta_w)

            counter3 += 1              
        w_est[:,k]= eta_w
        Pw = np.dot(np.eye(2*n_up)-np.dot(Kw,Hw),Pwf)
#            DeltaW_est[:,k-1] = w_est[:,k]
#            UncDeltaW[:,k-1] = np.sqrt(np.diag(P))
        
        ##################################
        ## State Estimation
        ##################################
        wf = eta_w
        xf = wf[n_up:]*Sigmoid(wf[:n_up]*Sigmoid(x_est[:,k-1]))
        
        F = transition2(x_est[:,k-1], wf)
        Pf = np.dot(F,np.dot(P,F.T)) + Q
        
        mu = V_est[:,k-1]
        eta = xf
        est_error1 = 1
        counter1 = 1
         
        while(est_error1 > accuracy) and (counter1 < iterstop):
            
            MU= calcM(mu)
            muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1]+np.dot(Dnm,eta)) - Slack[:,k-1]
            est_error2 = 1
            counter2 = 1
            while  (est_error2 > accuracy) and (counter2 < iterstop):
                    MU= calcM(muiter)
                    temp2 = muiter
                    muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1] + np.dot(Dnm,eta)) - Slack[:,k-1]
                    est_error2 = np.linalg.norm(muiter - temp2)
                    counter2 += 1
                   
            mu = muiter
            #Dh(22,22), H(8,16), K(16,8)
            Dh = jacobian(Y00,Ys,mu,Vs[:,k-1])
            H = np.dot(Cm,np.dot(np.linalg.inv(Dh),Dnm))
            K=np.dot(np.dot(Pf,H.T),np.linalg.pinv(np.dot(H,np.dot(Pf,H.T)) + R))
            temp1 = eta
            eta = xf + np.dot(K,y[:,k-1] - np.dot(Cm,mu) - np.dot(H,xf))
            est_error1 = np.linalg.norm(temp1-eta)
            counter1 += 1
        
        x_est[:,k]= eta
        P = np.dot(np.eye(n_up)-np.dot(K,H),Pf)
        V_est[:,k] = mu
        DeltaS_est[:,k-1] = x_est[:,k]
        UncDeltaS[:,k-1] = np.sqrt(np.diag(P))
        S_est[:,k-1] = Sm[:,k-1] + np.dot(Dnm,DeltaS_est[:,k-1]) # S_est = S + D_ng * DeltaS_est
    t1=t.time()-t0
    print t1
    return S_est, x_est, w_est          

S_est, x_est, w_est = JointIteratedKalmanFilter(y,Sm,Yws,Vs,iterstop=50,accuracy=1e-9)

mse = np.mean(np.square(S_est-S),axis=1)

print np.sum(mse)/(n_up)

figure(44); clf()    # active power

for k in range(n_K):
    subplot(3,4,k+1)
    plot(time,-S[k+n_K,:],'b-D',linewidth=2.5,label="True Value", ms=5)
    plot(time,-S_est[k+n_K,:],'r-o',linewidth=3.5,label="RNN-2", ms=5)
    plot(time,-Sfc[k+n_K,:],'c-',linewidth=2.5,label="Pseudo") 
    #plot(time,xx_est[k,:],'m-*',linewidth=3.5,label="estimated fixed",ms=5)
    legend(loc="upper right")
    title("node %d" %(k+1), fontweight='bold')
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45)
    yticks(fontweight='bold')
    grid()
subplots_adjust(left=0.05,right=0.98)

figure(10); clf()    # weight
color = ['b', 'g', 'r', 'c', 'm', 'y','b', 'g']+['b', 'g', 'r', 'c', 'm', 'y','b', 'g']
for i,k in enumerate([2,3,4,6,7,9,10,11]+[2,3,4,6,7,9,10,11]):
    plot(time,w_est[i,1:],color[i],linewidth=2.5,label="Node {} Active".format(k), ms=1)
    plot(time,w_est[i+8,1:],color[i],linewidth=2.5,label="Node {} Reactive".format(k), ms=1)
    legend(loc="upper right")
    title("Weight Updating process", fontweight='bold')
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45)
    yticks(fontweight='bold')
    grid()
subplots_adjust(left=0.05,right=0.98)


figure(49); clf()
plot(time,-S[1+n_K,:],'b-D',linewidth=2.5,label="true value", ms=5)
plot(time,-S_est[1+n_K,:],'r-o',linewidth=3.5,label="estimated online", ms=5)
plot(time,-Sfc[1+n_K,:],'c-',linewidth=2.5,label="pseudo-measurements") 
#plot(time,xx_est[k,:],'m-*',linewidth=3.5,label="estimated fixed",ms=5)
legend(loc="upper right")
title("node 2", fontweight='bold')
xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
xticks(rotation=45)
yticks(fontweight='bold')
xlabel('Time [h]', fontsize=30, fontweight='bold')
grid()

figure(450); clf()
plot(time,-S[8+n_K,:],'b-D',linewidth=2.5,label="true value", ms=5)
plot(time,-S_est[8+n_K,:],'r-o',linewidth=3.5,label="estimated online", ms=5)
plot(time,-Sfc[8+n_K,:],'c-',linewidth=2.5,label="pseudo-measurements") 
#plot(time,xx_est[k,:],'m-*',linewidth=3.5,label="estimated fixed",ms=5)
legend(loc="upper right")
title("node 9", fontweight='bold')
xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
xticks(rotation=45)
yticks(fontweight='bold')
xlabel('Time [h]', fontsize=30, fontweight='bold')
grid()

np.save('./dynamic_iter/AR_true.npy',S)
np.save('./dynamic_iter/dynamic.npy',S_est)
np.save('./dynamic_iter/Pseudo.npy',Sfc)
