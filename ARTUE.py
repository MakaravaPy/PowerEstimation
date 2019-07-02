

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:08:37 2015

@author: makara01
"""


from pypower.api import ppoption, runpf, makeYbus, case4gs, case6ww, case9, case9Q, case14, case24_ieee_rts, case30, \
    case30Q, case30pwl, case39, case57, case118, case300, t_case30_userfcns
from pypower.printpf import printpf
from pypower.ppoption import ppoption 
from pypower.loadcase import loadcase 
import numpy as np                       # for mathematics
import networkx as nx                    # only required to plot network graph
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_I
from pypower.t.t_is import t_is 
from cmath import cos, sin 
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from scipy import delete,stats
from matplotlib.pyplot import *
from numpy import matlib
from scipy.io import loadmat                                                                
from GridSens.NLO.Kalman import get_system_matrices, LinearKalmanFilter, LinearKalmanFilter_AR2, IteratedExtendedKalman, IteratedExtendedKalman_AR2
    # NLO solver
import GridSens.tools.data_tools as dt
from GridSens.NLO import misc                              # miscellaneous functions
from matplotlib.pyplot import * 
from GridSens.tools.load import convert_mcase, convert_to_python_indices
import pandas as pd
import GridSens.tools.profilesPV as ProfilePV
import GridSens.tools.ARMAmodel as ARMA
import statsmodels.api as sm


mat = loadmat('MatpowerPQ.mat')
loadsP15to25 = mat['loadsP15to25']  # Active power  
loadsQ15to25 = mat['loadsQ15to25']  # Reactive power
genP19 = mat['genP19']
genQ19 = mat['genQ19']




fname = "Net1_UKGDS_60_subgrid.m"
fnameWhole = "Net1_UKGDS_60.m"
convert_mcase(fname)
from Net1_UKGDS_60_subgrid import Net1_UKGDS_60_ as net
from Net1_UKGDS_60 import Net1_UKGDS_60_ as netWhole

t_f = 96

##################################################################################
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

filename1 = '/_gesichert/makara01/Documents/DataNguyen/Klant55/Monthly/June/dataMonthKlant55.dat'
PVdata = np.loadtxt(filename1)

location = 'Normal' # ['High', 'Normal', 'Low']
Season = 'Summer' # ['Winter', 'Spring', 'Summer', 'Autumn']
    

Ampl, mu, sig = ProfilePV.ProfilePower(location, Season)  # paramater identification according your choise of location and season
FittedPV = ProfilePV.gaussian(np.linspace(0, t_f-1, t_f), Ampl, baseMVA, mu, sig, location) # creating PV pseudo-data according your choise of location and season
LoadP = -PVdata[:,19] # in kVA 

################# Changing Loads ####################
filename2 = '/_gesichert/makara01/Documents/DataNguyen/Klant55/Monthly/June/dataMonthKlant55.dat'
PVdata2 = np.loadtxt(filename2)
loadsP15to25[6,:] = -PVdata[:,14] /1000 +loadsP15to25[6,:]# in kVA **********************************************************************************************************************************************

######################################################

casedata = net()
convert_to_python_indices(casedata)
ppc = casedata
ppopt = ppoption(PF_ALG=2)
ppc = ext2int(ppc)
figure(754)
g = nx.Graph()
i = ppc['bus'][:, BUS_I].astype(int)
g.add_nodes_from(i, bgcolor='green')
#nx.draw_networkx_nodes(g,pos=nx.spring_layout(g))
fr = ppc['branch'][:, F_BUS].astype(int)
to = ppc['branch'][:, T_BUS].astype(int)
g.add_edges_from(zip(fr, to), color='magenta')
nx.draw(g, with_labels=True, node_size=1000,node_color='skyblue',width=0.5)
show()

# time variation
t_ges = 1440  # all time in min
delta_t = 15  # time intervals in min
time= np.arange(delta_t, t_ges+delta_t,delta_t)

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




ReV = v_mag*np.cos(np.radians(v_ang))
ImV = v_mag*np.sin(np.radians(v_ang))
V = (11/(np.sqrt(3)))*np.vstack((ReV[2:,:],ImV[2:,:])) # voltage without slack bus

ReVs = v_mag[1,:]*np.cos(np.radians(v_ang[1,:]))
ImVs = v_mag[1,:]*np.sin(np.radians(v_ang[1,:]))
Vs = (11/(np.sqrt(3)))*np.vstack((ReVs, ImVs))
V_slack=Vs                                          # voltage at slack bus







# Create forecasts for NLO



Sfctemp = -S0/11
Pfc = Sfctemp[0,:]
Qfc = Sfctemp[1,:]
Sfc = np.vstack((matlib.repmat(Pfc,len(Pn[:,0]),1), matlib.repmat(Qfc,len(Qn[:,0]),1)))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
n_K = S.shape[0]/2
###################################################################################
volt_meas_idx  = [1,2,3,4,5,6,7,8,9,10,11]      # nodes at which voltage is measured (ignoring slack node)
actpow_meas_idx   = []    # nodes at which active power is measured 
reactpow_meas_idx = []    # nodes at which reactive power is measured 

# which type of pseudo-measurement to use
#S_vorst = np.zeros((22,t_f))
Sfc[6,:] = FittedPV[:] # with MVA ***********************************************************************************************************************************************************************************

S_forecast = Sfc

#ar_c = ['-0.9','-0.8','-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1','0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8', '0.9']
Qar = ['0.0001']#, '0.0001', '0.001', '0.01', '0.1', '0.5', '1']
ar_c1 =  ['-1.9','-1.8','-1.7','-1.6','-1.5','-1.4','-1.3','-1.2','-1.1','1.0','1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8', '1.9']

#a = np.zeros((len(Qar)*len(ar_c1)**2,4))    
#count=0
#for ar1 in ar_c1:
#    for ar2 in ar_c1:
#        for q in Qar:
#            
####################### ARMA Generation ####################################
ar_coef=[1.7, -0.7] # ar_coef - array of length p with the AR(p) coefficients
#ar_coef=[float(ar1), float(ar2)]
p_ar = np.size(ar_coef)
ma_coef=[0.0]       # ma_coef - array of length q with the MA(q) coefficients 
q_ma = np.size(ma_coef)

sigma0=10#q                # sigma - standard deviation of noise
arma=ARMA.ARMAgeneration(ar_coef,ma_coef,sigma0,t_f, s=50) #generate an random sample of an ARMA process, fix seed or change for randomness 
S_vorst=np.zeros((2*n_K,t_f))
S_vorstarma=np.zeros((2*n_K,t_f))

for i in range(2*n_K):
    S_vorstarma[i,:] = arma[:]+S_forecast[i,:] #

S_vorst = S_forecast
#%% construct matrices to incorporate voltage measurement locations
m = [mm-1 for mm in volt_meas_idx] # translate to Python array indexing
vmeas = np.zeros(n_K,dtype=bool); vmeas[m] = True

#%% construct matrices to incorporate power measurement locations 
pmeas = np.zeros(n_K,dtype=bool)
pmeas[[p-1 for p in actpow_meas_idx]] = True
qmeas = np.zeros(n_K,dtype=bool)
qmeas[[q-1 for q in reactpow_meas_idx]] = True                       

Cm,Dnm,Dm = get_system_matrices(pmeas,qmeas,vmeas)

#%% voltage and power measurement values, pseudo-measurement values
Ywo_slack,Y_slack = dt.separate_Yslack(Yws,slack_idx=0)
y  = misc.MeasVoltageVector(t_f,V_slack[0,:],Cm,V,Ywo_slack,Y_slack)  # measured voltages
Sm = misc.PowerInputSim(Dm, Dnm, S, S_vorst,pmeas,qmeas)[0]   # total power input
Pres = S[-pmeas,:] - S_vorst[-pmeas,:]

Qres = S[n_K:,:][-qmeas,:] - S_vorst[n_K:,:][-qmeas,:]
data = np.vstack((Pres,Qres))
qvals = np.zeros(data.shape[0])

arma_mod20 = sm.tsa.ARMA(data[6,:], (2,0)).fit() #**********************************************************************************************************************************************
Const = arma_mod20.params[0]
PHI1 = arma_mod20.params[1]
PHI2 = arma_mod20.params[2]


A11 = PHI1*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
A12 = PHI2*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
A21 = 1*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
A22 = 0*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
A=np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22)))) # (2*(n_k-1),2*(n_k-1))


sigma =sigma0#float(q)
Q11 = sigma*np.eye(2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx))
Q12 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
Q21 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
Q22 = np.zeros((2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx),2*n_K-np.size(actpow_meas_idx)-np.size(reactpow_meas_idx)))
Q=np.vstack((np.hstack((Q11,Q12)),np.hstack((Q21,Q22)))) # (2*(n_k-1),2*(n_k-1))
            

            
R = 1e-3*np.eye(y.shape[0])     # Variance of the output noise

S1 = np.vstack((np.vstack((np.zeros(np.shape(S[0,:])), S[1:,:])),S[:,:]))
S_est, V_est, UncS,DeltaS,UncDeltaS = LinearKalmanFilter_AR2(-S1,y,Sm,A,Q,R,pmeas,qmeas,vmeas,Yws,V_slack,slack_idx=0,roundY=4, v0 = V[:,0],x0=np.hstack((np.zeros(np.shape(data[:,0])),data[:,0])))

###
################## Estimated active power ##########################
###
figure(4); clf()    # active power
#suptitle("Active power \n Voltage is measured at all nodes, active power is measured at nodes {2,5,8,11}", fontsize=20, fontweight='bold')
for k in range(n_K):
    subplot(3,4,k+1)
    ylabel('Active power [MW]', fontsize=20, fontweight='bold')
    plot(time,-S[k,:],'b-o',linewidth=2.5,label="true value")
    plot(time,-S_est[k,:],'r-x',linewidth=2.5,label="estimated")
    #fill_between(time, -S_est[k,:]-UncS[k,:], -S_est[k,:]+UncS[k,:], facecolor='yellow', alpha=0.5,label="S +/- UncS")    
    plot(time,-S_vorst[k,:],'g-o',linewidth=2.5,label="pseudo-measurements",alpha=0.5)   
   # plot(time,-S_vorstarma[k,:],'c-D',linewidth=2.5,label="pseudo-measurements arma",alpha=0.5) 
    legend(loc="upper right")
    title("node %d" %(k+1), fontweight='bold')
    xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 12, fontweight='bold')
    xticks(rotation=45)
    xlabel('Time [h]', fontsize=20, fontweight='bold')
    yticks(fontweight='bold')
    grid()
subplots_adjust(left=0.05,right=0.98)
#
xx= []
xx = np.loadtxt("/Users/makara01/Documents/oioi/SS2.out")
figure(45); clf()    # active power
#suptitle("Active power \n Voltage is measured at all nodes, active power is measured at nodes {2,5,8,11}", fontsize=20, fontweight='bold')
ylabel('Active power [MW]', fontsize=35, fontweight='bold')
xlabel('Time [h]', fontsize=35, fontweight='bold')
plot(time,-S[1,:],'b-D',linewidth=2.5,label="measurements",ms=15)
plot(time,-Sfc[1,:]-Const,'m-',linewidth=4.5,label="pseudo-measurements",alpha=0.5)    
plot(time,xx[:],'k-v',linewidth=2.5,label="simple model",ms=15)
plot(time,-S_est[1,:],'c-o',linewidth=2.5,label="ARMA(2,0)",ms=15)
legend(loc="upper right", fontsize=40)
xticks(np.linspace(0,t_f*15,13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], fontsize = 25, fontweight='bold')
xticks(rotation=45, fontsize=25)
yticks(fontweight='bold',fontsize=25)
grid()


