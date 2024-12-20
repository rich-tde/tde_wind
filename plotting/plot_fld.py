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
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses

dataL = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/LowResrich_red.csv', delimiter=',', dtype=float)
tfbL = dataL[:, 1]   
Lum_L = dataL[:, 2]   
data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/rich_red.csv', delimiter=',', dtype=float)
tfb = data[:, 1]   
Lum = data[:, 2] 
data8 = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/8_red.csv', delimiter=',', dtype=float)
tfb8 = data8[:, 1]   
Lum8 = data8[:, 2] 
dataH = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/HiResrich_red.csv', delimiter=',', dtype=float)
tfbH = dataH[:, 1]
Lum_H = dataH[:, 2]
dataDoub = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}DoubleRad/DoubleRadrich_red.csv', delimiter=',', dtype=float)
tfbDou = dataDoub[:, 1]
Lum_Dou = dataDoub[:, 2]

diff8 = []
diffL = []
diffH = []
diffDou = []

for i8, time in enumerate(tfb8):
    i = np.argmin(np.abs(tfb-time))
    diff8.append(1-Lum8[i8]/Lum[i])

for iL, time in enumerate(tfbL):
    i = np.argmin(np.abs(tfb-time))
    diffL.append(1-Lum_L[iL]/Lum[i])

for iH, time in enumerate(tfbH):
    i = np.argmin(np.abs(tfb-time))
    diffH.append(1-Lum_H[iH]/Lum[i])

for iDou, time in enumerate(tfbDou):
    i = np.argmin(np.abs(tfb-time))
    diffDou.append(1-Lum_Dou[iDou]/Lum[i])

diff8 = np.array(diff8)
diffL = np.array(diffL)
diffH = np.array(diffH)
diffDou = np.array(diffDou)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.scatter(tfb, Lum, s = 20, label= 'NSIDE = 4',  c= 'darkorange')
ax1.scatter(tfb8, Lum8,  s = 4, label = 'NSIDE = 8', c ='darkgreen')
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
ax1.legend(fontsize = 16)
ax2.scatter(tfb8, np.abs(diff8), color = 'darkgreen', s = 4)
ax2.set_xlabel(r'$t/t_{\rm fb}$', fontsize = 20)
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.set_xlim(0,0.9)
    ax.grid()
plt.savefig(f'/Users/paolamartire/shocks/Figs/multiple/fld_NSIDE.pdf')


# Plot the data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.scatter(tfbDou, Lum_Dou, s = 4, label = 'DoubleRad', c ='navy')
ax1.scatter(tfbL, Lum_L, s = 4, label= 'Low', c= 'b')
ax1.scatter(tfb, Lum, s = 4, label = 'Fid', c ='darkorange')
ax1.scatter(tfbH, Lum_H, s = 4, label= 'High', c = 'dodgerblue')
ax1.axhline(y=Ledd, c = 'k', linestyle = '--')
ax1.text(0.1, 1.3*Ledd, r'$L_{\rm Edd}$', fontsize = 18)
ax1.set_yscale('log')
ax1.set_ylim(1e37, 5e42)
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
ax1.grid()

ax2.scatter(tfbDou, np.abs(diffDou), color = 'navy', s = 4, label = 'DoubleRad')
ax2.scatter(tfbL, np.abs(diffL), color = 'b', s = 4, label = 'Low')
ax2.scatter(tfbH, np.abs(diffH), color = 'dodgerblue', s = 4, label = 'High')
ax2.set_yscale('log')
ax2.set_ylim(1e-2, 1e2)
ax2.set_xlabel(r'$t [t_{\rm fb}]$', fontsize = 20)
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
ax2.grid()
ax1.legend(fontsize = 18)   
ax2.tick_params(axis='y', which='minor', length = 3)
ax2.tick_params(axis='y', which='major', length = 5)

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


