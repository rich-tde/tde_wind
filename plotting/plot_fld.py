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
from Utilities.time_extractor import days_since_distruption

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
tfallback = 2.5777261297507925 * 24 * 3600 #2.5 days
Ledd = 1.26e38 * Mbh # erg/s

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
ax1.axhline(y=Ledd, c = 'k', linestyle = '--')
ax1.text(0.1, 1.2*Ledd, r'$L_{\rm Edd}$', fontsize = 18)
ax1.set_yscale('log')
ax1.set_ylim(5e36, 4e42)
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)

ax2.scatter(tfbDou, np.abs(diffDou), color = 'navy', s = 4, label = 'DoubleRad')
ax2.scatter(tfbL, np.abs(diffL), color = 'b', s = 4, label = 'Low')
ax2.scatter(tfbH, np.abs(diffH), color = 'dodgerblue', s = 4, label = 'High')
ax2.set_xlabel(r'$t/t_{\rm fb}$', fontsize = 20)
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
ax1.grid()
ax2.grid()
ax2.set_yscale('log')
ax1.legend(fontsize = 18)   

# Get the existing ticks on the x-axis
for ax in [ax1, ax2]:
    original_ticks = ax.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    ax.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', width=0.7, length=7)
    ax.tick_params(axis='x', which='minor', width=0.5, length=5)
    ax.set_xlim(np.min(tfb), np.max(tfb))
plt.savefig(f'/Users/paolamartire/shocks/Figs/multiple/fld.pdf')
plt.show()


#%% Compute efficiency and Rsh
eta_sh = np.zeros(len(tfb))
R_sh = np.zeros(len(tfb))
for i,time in enumerate(tfb):
    Mdot_th = mstar*prel.Msol_cgs/(3*tfallback) * (time/tfallback)**(-5/3) # CGS
    eta_sh[i] = Lum[i]/(Mdot_th*prel.c_cgs**2) # CGS
    R_sh[i] = prel.G_cgs * Mbh *prel.Msol_cgs / (prel.c_cgs**2 * eta_sh[i]) # CGS

eta_shL = np.zeros(len(tfbL))
R_shL = np.zeros(len(tfbL))
for i,time in enumerate(tfbL):
    Mdot_th = mstar*prel.Msol_cgs/(3*tfallback) * (time/tfallback)**(-5/3) # CGS
    eta_shL[i] = Lum_L[i]/(Mdot_th*prel.c_cgs**2) # CGS
    R_shL[i] = prel.G_cgs * Mbh *prel.Msol_cgs / (prel.c_cgs**2 * eta_shL[i]) # CGS

eta_shH = np.zeros(len(tfbH))
R_shH = np.zeros(len(tfbH))
for i,time in enumerate(tfbH):
    Mdot_th = mstar*prel.Msol_cgs/(3*tfallback) * (time/tfallback)**(-5/3) # CGS
    eta_shH[i] = Lum_H[i]/(Mdot_th*prel.c_cgs**2) # CGS
    R_shH[i] = prel.G_cgs * Mbh *prel.Msol_cgs / (prel.c_cgs**2 * eta_shH[i]) # CGS

eta_shDou = np.zeros(len(tfbDou))
R_shDou = np.zeros(len(tfbDou))
for i,time in enumerate(tfbDou):
    Mdot_th = mstar*prel.Msol_cgs/(3*tfallback) * (time/tfallback)**(-5/3) # CGS
    eta_shDou[i] = Lum_Dou[i]/(Mdot_th*prel.c_cgs**2) # CGS
    R_shDou[i] = prel.G_cgs * Mbh *prel.Msol_cgs / (prel.c_cgs**2 * eta_shDou[i]) # CGS
         
# %%
plt.figure(figsize=(8, 6))
plt.scatter(tfb, R_sh, label = 'Fid', s = 4, color = 'darkorange')
plt.scatter(tfbL, R_shL, label = 'Low', s = 4, color = 'b')
plt.scatter(tfbH, R_shH, label = 'High', s = 4, color = 'dodgerblue')
plt.scatter(tfbDou, R_shDou, label = 'DoubleRad', s = 4, color = 'navy')
plt.yscale('log')
plt.xlabel(r'$t/t_{\rm fb}$', fontsize = 20)
plt.ylabel(r'$R_{\rm sh}$ [cm]', fontsize = 20)
plt.legend(fontsize = 18)

# Get the existing ticks on the x-axis
original_ticks = plt.xticks()[0]
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
plt.xticks(new_ticks, labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks])       
plt.tick_params(axis='x', which='major', width=0.7, length=7)
plt.tick_params(axis='x', which='minor', width=0.5, length=5)
plt.xlim(np.min(tfb), np.max(tfb))
plt.grid()