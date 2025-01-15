#%%
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
from matplotlib.collections import LineCollection
import Utilities.prelude as prel
import matplotlib.colors as colors
from Utilities.operators import sort_list
from src import orbits as orb

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
apocenter = orb.apocentre(Rstar, mstar, Mbh, beta)

dataL = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/LowResrich_red.csv', delimiter=',', dtype=float)
tfbL = dataL[:, 1]   
Lum_L = dataL[:, 2]   
Lum_L, tfbL = sort_list([Lum_L, tfbL], tfbL)
data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/rich_red.csv', delimiter=',', dtype=float)
tfb = data[:, 1]   
Lum = data[:, 2] 
Lum, tfb = sort_list([Lum, tfb], tfb)
# data8 = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/8_red.csv', delimiter=',', dtype=float)
# tfb8 = data8[:, 1]   
# Lum8 = data8[:, 2] 
dataH = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/HiResrich_red.csv', delimiter=',', dtype=float)
tfbH = dataH[:, 1]
Lum_H = dataH[:, 2]
tfbH, Lum_H = sort_list([tfbH, Lum_H], tfbH)
dataDoub = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}DoubleRad/DoubleRadrich_red.csv', delimiter=',', dtype=float)
tfbDou = dataDoub[:, 1]
Lum_Dou = dataDoub[:, 2]
tfbDou, Lum_Dou = sort_list([tfbDou, Lum_Dou], tfbDou)

# Photosphere data
ph_data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/photo_mean.txt')
tfbRph, Rph = ph_data[0], ph_data[3]
tfbRph, Rph = sort_list([tfbRph, Rph], tfbRph)
dataDissL = np.loadtxt(f'{abspath}data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/Rdiss_LowRes.txt')
LDissL = dataDissL[3]
dataDiss = np.loadtxt(f'{abspath}data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/Rdiss_.txt')
LDiss = dataDiss[3]
dataDissH = np.loadtxt(f'{abspath}data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/Rdiss_HiRes.txt')
LDissH = dataDissH[3]

diff8 = []
diffL = []
diffH = []
diffDou = []

# for i8, time in enumerate(tfb8):
#     i = np.argmin(np.abs(tfb-time))
#     diff8.append(1-Lum8[i8]/Lum[i])

for iL, time in enumerate(tfbL):
    i = np.argmin(np.abs(tfb-time))
    diffL.append(1-Lum_L[iL]/Lum[i])

for iH, time in enumerate(tfbH):
    i = np.argmin(np.abs(tfb-time))
    diffH.append(1-Lum_H[iH]/Lum[i])

for iDou, time in enumerate(tfbDou):
    i = np.argmin(np.abs(tfb-time))
    diffDou.append(1-Lum_Dou[iDou]/Lum[i])

# diff8 = np.array(diff8)
diffL = np.array(diffL)
diffH = np.array(diffH)
diffDou = np.array(diffDou)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
# ax1.scatter(tfb, Lum, s = 20, label= 'NSIDE = 4',  c= 'yellowgreen')
# ax1.scatter(tfb8, Lum8,  s = 4, label = 'NSIDE = 8', c ='darkgreen')
# ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
# ax1.legend(fontsize = 16)
# ax2.scatter(tfb8, np.abs(diff8), color = 'darkgreen', s = 4)
# ax2.set_xlabel(r'$t/t_{\rm fb}$', fontsize = 20)
# ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
# for ax in [ax1, ax2]:
#     ax.set_yscale('log')
#     ax.set_xlim(0,0.9)
#     ax.grid()
# plt.savefig(f'/Users/paolamartire/shocks/Figs/multiple/fld_NSIDE.pdf')
#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
img = ax.scatter(tfb, Lum, s = 5, c = Rph/apocenter, cmap = 'viridis', norm = colors.LogNorm(
                 vmin = 1e-2, vmax =3))
cbar = fig.colorbar(img)
cbar.set_label(r'$<R_{\rm ph}> [R_a]$', fontsize = 20)
ax.axhline(y=Ledd, c = 'k', linestyle = '--')
ax.text(0.1, 1.3*Ledd, r'$L_{\rm Edd}$', fontsize = 18)
ax.set_yscale('log')
ax.set_ylim(1e37, 5e42)
ax.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
ax.grid()
original_ticks = ax.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
ax.set_xticks(new_ticks)
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
ax.set_xticklabels(labels)
ax.tick_params(axis='x', which='major', width=0.7, length=7)
ax.tick_params(axis='x', which='minor', width=0.5, length=5)
ax.set_xlim(np.min(tfb), np.max(tfb))
plt.savefig(f'/Users/paolamartire/shocks/Figs/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/onefld.pdf')

#%% Plot the data
# Create segments for LineCollection
pointsL = np.array([tfbL, Lum_L]).T.reshape(-1, 1, 2)
segmentsL = np.concatenate([pointsL[:-1], pointsL[1:]], axis=1)
points = np.array([tfb, Lum]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
pointsH = np.array([tfbH, Lum_H]).T.reshape(-1, 1, 2)
segmentsH = np.concatenate([pointsH[:-1], pointsH[1:]], axis=1)


# Create a LineCollection with varying line widths
norm = np.mean(LDiss)
lcL = LineCollection(segmentsL, linewidths=LDissL/norm, color='C1', label='Low')
lc = LineCollection(segments, linewidths=LDiss/norm, color='yellowgreen', label='Fid')
lcH = LineCollection(segmentsH, linewidths=LDissH/norm, color='darkviolet', label='High')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
# ax1.scatter(tfbDou, Lum_Dou, s = 4, label = 'DoubleRad', c ='navy')
ax1.add_collection(lc)
ax1.add_collection(lcL)
ax1.add_collection(lcH)
# ax1.plot(tfbL, Lum_L, label= 'Low', c= 'C1')
# ax1.plot(tfb, Lum, label = 'Fid', c ='yellowgreen')
# ax1.plot(tfbH, Lum_H, label= 'High', c = 'darkviolet')
# ax1.axhline(y=Ledd, c = 'k', linestyle = '--')
# ax1.text(0.1, 1.3*Ledd, r'$L_{\rm Edd}$', fontsize = 18)
ax1.set_yscale('log')
ax1.set_ylim(1e37, 5e42)
ax1.set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
ax1.grid()

# ax2.plot(tfbDou, np.abs(diffDou), color = 'navy', label = 'DoubleRad')
ax2.plot(tfbL, np.abs(diffL), color = 'C1', label = 'Low')
ax2.plot(tfbH, np.abs(diffH), color = 'darkviolet', label = 'High')
ax2.set_yscale('log')
ax2.set_ylim(1e-2, 1e2)
ax2.set_xlabel(r'$t [t_{\rm fb}]$', fontsize = 20)
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$ from Fid', fontsize = 16)
ax2.grid()
ax1.legend(fontsize = 18, loc = 'lower right')   
ax2.tick_params(axis='y', which='minor', length = 3)
ax2.tick_params(axis='y', which='major', length = 5)

# Get the existing ticks on the x-axis
original_ticks = ax1.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2]:
    ax.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', width=0.7, length=7)
    ax.tick_params(axis='x', which='minor', width=0.5, length=5)
    ax.set_xlim(np.min(tfb), np.max(tfb))

plt.savefig(f'/Users/paolamartire/shocks/Figs/multiple/fld.pdf')
print('last errors L-fid', np.median(np.abs(diffL[-10:-5])))
print('last errors H-fid', np.median(np.abs(diffH[-10:-1])))


