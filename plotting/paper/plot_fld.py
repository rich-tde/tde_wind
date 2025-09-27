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
check = 'HiResNewAMR'
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.25/(prel.Rsol_cgs**2/prel.Msol_cgs), 0.004, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fall = things['t_fb_days']
t_fall_cgs = t_fall * 24 * 3600
Rt = things['Rt']
apo = things['apo']

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)
dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3] * prel.en_converter/prel.tsol_cgs

time_theory = tfb[210:-1]
Lum_theory = 5e41*time_theory**(-5/3)

medianRph = np.zeros(len(snaps))
for i, snap in enumerate(snaps):
    photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    xph, yph, zph = photo[0], photo[1], photo[2]
    rph = np.sqrt(xph**2 + yph**2 + zph**2)
    medianRph[i] = np.median(rph)
#     # Find the energy of the element at time t
#     energy = orb.keplerian_energy(Mbh, prel.G, tsol)
#     # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
#     i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
#     dMdE_t = dMdE_distr_tokeep[i_bin]
#     mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
#     mfall = mdot # code units
#     mdot_cgs[i] = mdot * prel.Msol_cgs / prel.tsol_cgs # [g/s]
#     Lacc[i] = 0.1 * np.abs(mdot_cgs[i]) * prel.c_cgs**2

# ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo_stat.txt')
# tfbRph, Rph = ph_data[0], ph_data[1]
# tfbRph, Rph = sort_list([tfbRph, Rph], tfbRph)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
img = ax.scatter(tfb, Lum, s = 12, c = medianRph/Rt, cmap = 'viridis', norm = colors.LogNorm(
                 vmin = 1, vmax = 7e1))
cbar = fig.colorbar(img)
cbar.set_label(r'median $R_{\rm ph} [R_{\rm t}]$')#, fontsize = 20)
cbar.ax.tick_params(which='major', length = 5)
cbar.ax.tick_params(which='minor', length = 3)
ax.plot(tfbdiss, LDiss, '--', c= 'gray')
ax.axhline(y=Ledd_cgs, c = 'k', linestyle = '-.', linewidth = 2)
ax.text(0.15, 1.4*Ledd_cgs, r'$L_{\rm Edd}$', fontsize = 20)
# ax.plot(time_theory, Lum_theory, c = 'k', linestyle = 'dotted', linewidth = 1)
# ax.text(1.4, 9e40, r'$L\propto t^{-5/3}$', fontsize = 20)
ax.set_yscale('log')
ax.set_ylim(9e37, 8e42)
ax.set_ylabel(r'Luminosity [erg/s]')#, fontsize = 20)
ax.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 20)
ax.grid()
original_ticks = ax.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
ax.set_xticks(new_ticks)
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
# ax.set_xticklabels(labels)
ax.tick_params(axis='both', which='major', width = 1.2, length = 8, color = 'k')
ax.tick_params(axis='y', which='minor', width = 1, length = 5, color = 'k')
ax.set_xlim(np.min(tfb), np.max(tfb))
plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/onefld.pdf', bbox_inches='tight')

# %%
