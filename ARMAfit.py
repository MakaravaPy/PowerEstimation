
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 14:21:54 2023

@author: makara01
"""

import pandas as pd
from matplotlib.pyplot import * 
import numpy as np
import statsmodels.api as sm
import GridSens.tools.ARMAmodel as ARMAchka
from statsmodels.graphics.api import qqplot
from GridSens.tools.prepare_data import process_csv,make_equidist, reduce_Ts
import GridSens.tools.profilesLiander as ProfilePL
from scipy.signal import decimate	


xl_file = pd.ExcelFile('/Users/makara01/Documents/oioi/2014-01-20singlephase.xlsx')
df = xl_file.parse("active power")
dat = df['1_main'] 
dat1 = df['4_main']
dat2 = df['5_main']
dat3 = df['5_1']
dat4 = df['5_2']
dat5 = df['5_3']
dat6 = df['4_1']


Type = 'Type I' # ['Type I', 'Type II', 'Type III']
Season = 'Winter' # ['Winter', 'Spring', 'Summer', 'Autumn']
pseudo = ProfilePL.ProfilePowerLiander(Type, Season)
pseudo1 = ProfilePL.ProfilePowerLiander('Type II', Season)
pseudo2 = ProfilePL.ProfilePowerLiander('Type III', Season)
pseudo_val = decimate(pseudo,15)
pseudo_val1 = decimate(pseudo1,15)
pseudo_val2 = decimate(pseudo2,15)

data = dat2-pseudo_val
data1 = dat2-pseudo_val1
data2 = dat2-pseudo_val2

AR_lag = 2
MA_lag = 0
sigma0 = 0.1
model = sm.tsa.AR(data).fit(2)
model1 = sm.tsa.AR(data1).fit(2)
model2 = sm.tsa.AR(data2).fit(2)

results = sm.tsa.AR(data).fit(maxlags=15, ic='aic')
lag_order = results.k_ar
print lag_order


pred = model.predict(2, len(data)) 
pred1 = model1.predict(2, len(data)) 
pred2 = model2.predict(2, len(data)) 

print model.params
predd = model.predict(lag_order, len(data))
predd = pd.concat([data[:lag_order],predd], axis=0)

figure(2); clf()

plot(predd/10e-6,'c-o',linewidth=2.5,label=r"$S^{true}-S^{pseudo}$",ms=15, mec = "None")
plot(data/10e-6, 'k-v', linewidth=2.0,label="ARMA fit",ms=5)
legend(loc="upper left", fontsize=50)
ylabel('Active power [MW]', fontsize=35, fontweight='bold')
xlabel('Time [h]', fontsize=35, fontweight='bold')
xticks(np.linspace(0,len(data),13), ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '23:00', '18:00', '20:00', '22:00', '24:00'], rotation=45, fontsize = 25, fontweight='bold')
yticks(fontweight='bold',fontsize=25)
grid()


