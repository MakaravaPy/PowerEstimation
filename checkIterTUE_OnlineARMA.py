# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:41:39 2018

@author: makara01
"""

from pypower.api import ppoption, runpf, makeYbus, case4gs, case6ww, case9, case9Q, case14, case24_ieee_rts, case30, \
    case30Q, case30pwl, case39, case57, case118, case300, t_case30_userfcns
from pypower.printpf import printpf
from pypower.ppoption import ppoption 
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I
from pypower.t.t_is import t_is 
from cmath import cos, sin 
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from scipy.io import loadmat                                                                
import numpy as np
from GridSens.NLO.Kalman import get_system_matrices, jacobian
from GridSens.tools.data_tools import process_admittance, separate_Yslack
from GridSens.NLO import misc
import GridSens.tools.data_tools as dt
from GridSens.tools.load import convert_mcase, convert_to_python_indices
from matplotlib.pyplot import * 
from GridSens.NLO.Kalman import get_system_matrices, LinearKalmanFilter, LinearKalmanFilter_AR2, IteratedExtendedKalman
import GridSens.tools.profilesPV as ProfilePV
from numpy import matlib
import statsmodels.api as sm
import GridSens.tools.ARMAmodel as ARMA

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
from Net1_UKGDS_60_subgrid import Net1_UKGDS_60_ as net
from Net1_UKGDS_60 import Net1_UKGDS_60_ as netWhole

t_f = 96
# time variation
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

#Yws = Y
t_f = 96
slack_idx=0
p0=1e-3
iterstop=50
accuracy=1e-9
roundY=None

filename1 = '/Users/makara01/Documents/oioi/dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1)

location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
    

Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, t_f-1, t_f), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = -PVdata[:,19] # in kVA 

################# Changing Loads ####################
loadsP15to25[6,:] = -PVdata[:,14] /1000 +loadsP15to25[6,:]# in kVA ***************
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
    casedata['gen'][1,2] = 0#genQ19[:,n]              #Changing the values for the gen-reactive power
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

n_K=S.shape[0]/2
t_f = S.shape[1]

ReV = (11/(np.sqrt(3)))*v_mag*np.cos(np.radians(v_ang))
ImV = (11/(np.sqrt(3)))*v_mag*np.sin(np.radians(v_ang))
V = np.vstack((ReV[2:,:],ImV[2:,:])) # voltage without slack bus
ReVs = 11/(np.sqrt(3))*v_mag[1,:]*np.cos(np.radians(v_ang[1,:]))
ImVs = 11/(np.sqrt(3))*v_mag[1,:]*np.sin(np.radians(v_ang[1,:]))
Vs = np.vstack((ReVs, ImVs))
V_slack=Vs  
y = np.r_[ReV[2:,:],ImV[2:,:]]
Sfctemp = -S0/11
Pfc = Sfctemp[0,:]
Qfc = Sfctemp[1,:]
Sfc = np.vstack((matlib.repmat(Pfc,len(Pn[:,0]),1), matlib.repmat(Qfc,len(Qn[:,0]),1)))


n_K=S.shape[0]/2
t_f = S.shape[1]
t = np.linspace(15,1440,n_K)
slack_idx=0
p0=1e-3
iterstop=50
accuracy=1e-9
roundY=None
volt_meas_idx  = [2,3,9,11]      # nodes at which voltage is measured (ignoring slack node)
actpow_meas_idx   = [1,5,8]    # nodes at which active power is measured 
reactpow_meas_idx = [1,5,8]    # nodes at which reactive power is measured 
Sfc[6,:] = FittedPV[:] # with MVA **********************************************************************************************************************************************

S_vorst = Sfc


m = [mm-1 for mm in volt_meas_idx] 
vmeas = np.zeros(n_K,dtype=bool); vmeas[m] = True
pmeas = np.zeros(n_K,dtype=bool)
pmeas[[p-1 for p in actpow_meas_idx]] = True
qmeas = np.zeros(n_K,dtype=bool)
qmeas[[q-1 for q in reactpow_meas_idx]] = True 

Pres = S[~pmeas,:] - S_vorst[~pmeas,:]
Qres = S[n_K:,:][-qmeas,:] - S_vorst[n_K:,:][-qmeas,:]
data = np.vstack((Pres,Qres))


qvals = np.zeros(data.shape[0])
R = 1e-2*np.eye(2*len(vmeas.nonzero()[0])) 
nm = S.shape[0]
nx = 2*(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))




##########################################################################
phi0=np.tile(np.array([1.45,-0.5]), (nx/2,1))
Beta=np.zeros((nx/2,2,t_f+1))
Beta[:,:,0]=np.hstack((phi0[:,0].reshape(nx/2,1),phi0[:,1].reshape(nx/2,1)))

sigma = np.zeros((nx/2,t_f+1))
sigma_0 = 10* np.ones((nx/2))
sigma[:,0]=sigma_0[:] # already squared

Psi = np.zeros((nx/2,t_f,2))
e=np.zeros((nx/2,t_f))
gamma =  np.zeros((nx/2,t_f+1))
la=np.zeros((t_f+1))
la[0] = 1.0
F=np.eye(2) 
gamma[:,0] = np.ones((nx/2))

n_K= len(pmeas)
DeltaS_est = np.zeros((nx,t_f))    
UncDeltaS = np.zeros_like(DeltaS_est)    
S_est=np.zeros((nm,t_f))
S_est[-2]=0
S_est[-1]=0
Cm, Dnm, Dm = get_system_matrices(pmeas,qmeas,vmeas)  
Sm, Sg, Sf = misc.PowerInputSim(Dm, Dnm, S, S_vorst,pmeas,qmeas)
Ywo_slack,Y_slack = dt.separate_Yslack(Yws,slack_idx=0) 
#y  = misc.MeasVoltageVector(t_f,Vs[0,:],Cm,V,Ywo_slack,Y_slack) 
y0 = np.zeros((len(volt_meas_idx),t_f))
y1 = np.zeros((len(volt_meas_idx),t_f))
for k,ind in enumerate(vmeas.nonzero()[0]):
    y0[k,:] = V[ind,:]
    y1[k,:] = V[ind+n_K,:]
y = np.vstack((y0,y1)) 
Dtilde = np.zeros((Dnm.shape[1],nx))
for i in range(Dnm.shape[1]):
        Dtilde[i,2*i] = 1.0
        
        
Dnm = np.dot(Dnm,Dtilde)

x0 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
def calcM(mu):
		divisor = 3*(mu[:n_K]**2 + mu[n_K:]**2)
		return np.r_[np.c_[np.diag(mu[:n_K]/divisor), np.diag(mu[n_K:]/divisor)],
				     np.c_[np.diag(mu[n_K:]/divisor), -np.diag(mu[:n_K]/divisor)]]
         
 
def IteratedExtendedKalman_moi(y,Sm,R,Yws,Vs,iterstop=50,accuracy=1e-9):
        p0=1.0
        P = p0*np.eye(nx) 
        Y00,Ys = separate_Yslack(Yws,slack_idx,dcomplex=False,prec=roundY)
        Slack = np.linalg.solve(Y00,np.dot(Ys,Vs)).__array__()

    
        muiter=np.zeros((nm))
        temp1 = np.zeros((nx))
        temp2 = np.zeros((nm))
        V_est = np.zeros((2*n_K,t_f+1)) 
        v0 = 11/(np.sqrt(3))*np.hstack((np.ones(n_K), np.zeros(n_K)))
        
        if isinstance(v0,np.ndarray):
            V_est[:,0] = v0
        else:
            V_est[:n_K,0] = v0
        x_est=np.zeros((nx,t_f+1))         
            
        mu = np.hstack((v0[:n_K]*np.cos(v0[n_K:]),v0[:n_K]*np.sin(v0[n_K:])))
        Pf = P
        
        A11 = Beta[:nx/2,0,0]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
        A12 = Beta[:nx/2,1,0]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
        A21 = np.ones((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
        A22 = np.zeros((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
        A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) # (2*(n_k-1),2*(n_k-1))
                
        Q11 = sigma[:nx/2,0]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
        Q12 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
        Q21 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
        Q22 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
        Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22)))) # (2*(n_k-1),2*(n_k-1))
        for k in range(1,t_f+1):             
                    xf = np.dot(A,x_est[:,k-1])
                    Pf = np.dot(A,np.dot(P,A.T)) + Q
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
                                    muiter =np.dot(np.dot(np.linalg.inv(Y00),MU),Sm[:,k-1]+np.dot(Dnm,eta)) - Slack[:,k-1]
                                    est_error2 = np.linalg.norm(muiter-temp2)
                                    counter2 += 1
                                   
                            mu = muiter     
                            Dh = jacobian(Y00,Ys,mu,Vs[:,k-1])
                            H = np.dot(Cm,np.dot(np.linalg.inv(Dh),Dnm))
                            K=np.dot(np.dot(Pf,H.T),np.linalg.pinv(np.dot(H,np.dot(Pf,H.T))+R))
                            temp1 = eta
                            eta = xf + np.dot(K,y[:,k-1] - np.dot(Cm,mu) - np.dot(H,xf-eta))
                            est_error1 = np.linalg.norm(temp1-eta)
                            print est_error1
                            counter1 += 1
        
                    x_est[:,k]= eta
                    P = np.dot(np.eye(nx)-np.dot(K,H),Pf)
                    V_est[:,k] = mu
                    DeltaS_est[:,k-1] = x_est[:,k]
                    UncDeltaS[:,k-1] = np.sqrt(np.diag(P))
                    S_est[:,k-1] = Sm[:,k-1] + np.dot(Dnm,DeltaS_est[:,k-1]) # S_est = S + D_ng * DeltaS_est
                    
                    la[k]=0.99*la[k-1]+1-0.99
                    gamma[:,k] = gamma[:,k-1]/(la[k]+gamma[:,k-1])
                    #gamma[:,k] = 1.0/(k+0.1)   
                    Psi[:,k-1,:] = np.dstack((x_est[::2,k-2],x_est[::2,k-3]))     
                    psi = Psi[:,k-1,:].squeeze()
                    beta = Beta[:,:,k-1].squeeze()      
                    e = x_est[::2,k-1]-np.diag(np.dot(beta, psi.T))
                    beta += np.dot(np.diag(gamma[:,k]/sigma[:,k-1] *e), (np.dot(F, psi.T)).T ) 
                    sigma[:,k]=sigma[:,k-1]+gamma[:,k]*(e**2-sigma[:,k-1])  
                    Beta[:,:,k] = beta                         
                    
                    A11 = Beta[:nx/2,0,k]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
                    A12 = Beta[:nx/2,1,k]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
                    A21 = np.ones((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
                    A22 = np.zeros((nx/2))*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
                    A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) 

                    Q11 = sigma[:nx/2,k]*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
                    Q12 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
                    Q21 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
                    Q22 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
                    Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22))))
        return S_est, x_est, Beta, sigma          

S_est, x_est, beta, sigma = IteratedExtendedKalman_moi(y,Sm,R,Yws,Vs,iterstop=50,accuracy=1e-9)


figure(41); clf()    # active power
for k in range(n_K):
    subplot(3,4,k+1)
    ylabel('Active power [MW]', fontsize=20, fontweight='bold')
    plot(time,-S[k,:],'c-o',linewidth=2.5,label="true value")
    plot(time,-S_est[k,:],'m-x',linewidth=2.5,label="estimated")
   
    plot(time,-S_vorst[k,:],'y-',linewidth=4.5,label="pseudo-measurements",alpha=0.5)   
    legend(loc="upper right")
    title("node %d" %(k+1), fontweight='bold')
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45)
    yticks(fontweight='bold')
    grid()
subplots_adjust(left=0.05,right=0.98)

