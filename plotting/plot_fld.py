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
tfbL = np.round(dataL[:, 1], 2)     
Lum_L = np.round(dataL[:, 2], 2)     
data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/_red.csv', delimiter=',', dtype=float)
tfb = np.round(data[:, 1], 2)     
Lum = np.round(data[:, 2], 2)   
# dataH = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/HiRes_red.csv', delimiter=',', dtype=float)
# tfbH = np.round(dataH[:, 1], 2)
# Lum_H = np.round(dataH[:, 2], 2)
# dataDoub = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}DoubleRad/DoubleRad_red.csv', delimiter=',', dtype=float)
# tfbDou = np.round(dataDoub[:, 1], 2)
# Lum_Dou = np.round(dataDoub[:, 2], 2)

diffL = []
diffH = []
diffDou = []
for i, time in enumerate(tfb):
    iL = np.argmin(np.abs(tfbL-time))
    diffL.append(1-Lum_L[iL]/Lum[i])
    # iH = np.argmin(np.abs(tfbH-time))
    # diffH.append(1-Lum_H[iH]/Lum[i])
    # iDou = np.argmin(np.abs(tfbDou-time))
    # diffDou.append(1-Lum_Dou[iL]/Lum[i])
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
ax1.scatter(tfbL, Lum_L, label= 'Low', c= 'b')
ax1.scatter(tfb, Lum, label = 'Fid', c ='dodgerblue')
# ax1.scatter(tfbH, Lum_H, label= 'High', c = 'dodgerblue')
# ax1.scatter(tfbDou, Lum_Dou, lable = 'DoubleRad', c ='navy')
ax1.set_yscale('log')
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)

ax2.scatter(tfb[:len(diffL)], diffL, color = 'b', label = 'Low')
# ax2.scatter(tfb[:len(diffH)], diffL, color = 'dodgerblue', label = 'High')
# ax2.scatter(tfb[:len(diffDou)], diffL, color = 'navy', label = 'DoubleRad')
ax2.set_xlabel(r'$t_{\rm fb}$', fontsize = 20)
ax2.set_ylabel(r'$\Delta_{\rm rel}$ from Fid', fontsize = 16)
ax1.grid()
ax1.legend(fontsize = 18)   
ax2.legend(fontsize = 14)                                                                                                                        

plt.show()

