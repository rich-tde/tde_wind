import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'
import csv
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
step = ''

dataL = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/LowRes_red.csv', delimiter=',', dtype=float)
tfbL = dataL[:, 1]   
Lum_L = dataL[:, 2]   
data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/_red.csv', delimiter=',', dtype=float)
tfb = data[:, 1]   
Lum = data[:, 2] 
dataH = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/HiRes_red.csv', delimiter=',', dtype=float)
tfbH = dataH[:, 1]
Lum_H = dataH[:, 2]
dataDoub = np.loadtxt(f'{abspath}/data/DoubleRad_red.csv', delimiter=',', dtype=float)
tfbDou = dataDoub[:, 1]
Lum_Dou = dataDoub[:, 2]

diffL = []
diffH = []
diffDou = []
for iL, time in enumerate(tfbL):
    i = np.argmin(np.abs(tfb-time))
    diffL.append(1-Lum_L[iL]/Lum[i])

for iH, time in enumerate(tfbH):
    i = np.argmin(np.abs(tfb-time))
    diffH.append(1-Lum_H[iH]/Lum[i])

for iDou, time in enumerate(tfbDou):
    i = np.argmin(np.abs(tfb-time))
    diffDou.append(1-Lum_Dou[iDou]/Lum[i])

diffL = np.array(diffL)
diffH = np.array(diffH)
diffDou = np.array(diffDou)

""""
other way to do it using csv:
tfb = []
L = []
with open(f'{path}/{check}_red.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        tfb.append(np.round(float(row[1]),2))  # Adjust index if needed
        L.append(np.round(float(row[2]),2)) # Adjust index if needed
"""
# Plot the data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax1.scatter(tfbDou, Lum_Dou, s = 4, label = 'DoubleRad', c ='navy')
ax1.scatter(tfbL, Lum_L, s = 4, label= 'Low', c= 'b')
ax1.scatter(tfb, Lum, s = 4, label = 'Fid', c ='darkorange')
ax1.scatter(tfbH, Lum_H, s = 4, label= 'High', c = 'dodgerblue')
ax1.set_yscale('log')
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)

ax2.scatter(tfbDou, np.abs(diffDou), color = 'navy', s = 4, label = 'DoubleRad')
ax2.scatter(tfbL, np.abs(diffL), color = 'b', s = 4, label = 'Low')
ax2.scatter(tfbH, np.abs(diffH), color = 'dodgerblue', s = 4, label = 'High')
ax2.set_xlabel(r'$t_{\rm fb}$', fontsize = 20)
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
ax2.axhline(y=0.1, c = 'k', linestyle = '--')
ax1.grid()
ax2.set_yscale('log')
ax1.legend(fontsize = 18)   
ax2.legend(fontsize = 14)                                                                                                                        

plt.show()
