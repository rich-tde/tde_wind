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

data_oldP = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/_red.csv', delimiter=',', dtype=float)
tfb_oldP = data_oldP[:, 1]   
Lum_oldP = data_oldP[:, 2]

data_oldK = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/_redKONST.csv', delimiter=',', dtype=float)
tfb_oldK = data_oldK[:, 1]   
Lum_oldK = data_oldK[:, 2]
print(Lum_oldP[50],Lum_oldK[50], tfb_oldP[50])
dataL = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/LowRes_redNEW.csv', delimiter=',', dtype=float)
tfbL = dataL[:, 1]   
Lum_L = dataL[:, 2]  
dataP = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/_redNEW.csv', delimiter=',', dtype=float)
tfbP = dataP[:, 1]   
LumP = (dataP[:, 2])
data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/_redKONSTNEW.csv', delimiter=',', dtype=float)
tfb = data[:, 1]   
Lum = (data[:, 2])
dataH = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/HiRes_redNEW.csv', delimiter=',', dtype=float)
tfbH = dataH[:, 1]
Lum_H = dataH[:, 2]

diffL = []
diffH = []

for iL, time in enumerate(tfbL):
    i = np.argmin(np.abs(tfb-time))
    diffL.append(1-Lum_L[iL]/Lum[i])

for iH, time in enumerate(tfbH):
    i = np.argmin(np.abs(tfb-time))
    diffH.append(1-Lum_H[iH]/Lum[i])

diffL = np.array(diffL)
diffH = np.array(diffH)

plt.figure()
plt.scatter(tfb_oldP, Lum_oldP, s = 10, label = 'P', c = 'r')
plt.scatter(tfb_oldK, Lum_oldK/(4*np.pi), s = 2, label = r'K/4$\pi$', c ='darkorange')
plt.scatter(tfb_oldK, Lum_oldK, s = 8, label = 'K', c ='dodgerblue')
plt.yscale('log')
plt.ylim(1e37, 5e42)
plt.xlabel(r'$t/t_{\rm fb}$', fontsize = 25)
plt.grid()
plt.legend(fontsize = 18)
plt.title('Fiducial run with old extrapolation.')

# Plot the data (divide by 4pi before)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.scatter(tfbP, LumP/(4*np.pi), s = 10, label = r'new Fid P/4$\pi$', c = 'r')
ax1.scatter(tfb, Lum/(4*np.pi), s = 4, label = r'new Fid K', c ='darkorange')
ax1.scatter(tfbH, Lum_H/(4*np.pi), s = 4, label= 'High', c = 'dodgerblue')
ax1.scatter(tfbL, Lum_L/(4*np.pi), s = 4, label= 'Low', c= 'b')
ax1.axhline(y=Ledd, c = 'k', linestyle = '--')
ax1.text(0.1, 1.3*Ledd, r'$L_{\rm Edd}$', fontsize = 18)
ax1.set_yscale('log')
ax1.set_ylim(1e37, 5e42)
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
ax1.grid()

ax2.scatter(tfbH, np.abs(diffH), color = 'dodgerblue', s = 4, label = 'High')
ax2.scatter(tfbL, np.abs(diffL), color = 'b', s = 4, label = 'Low')
ax2.set_yscale('log')
ax2.set_ylim(1e-2, 1e2)
ax2.set_xlabel(r'$t/t_{\rm fb}$', fontsize = 20)
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
ax2.grid()
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

# plt.savefig(f'/Users/paolamartire/shocks/Figs/multiple/fld.pdf')


