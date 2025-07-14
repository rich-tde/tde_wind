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
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
t_fall_cgs = t_fall * 24 * 3600
Rt = Rstar * (Mbh/mstar)**(1/3)
norm_dMdE = Mbh/Rt * (Mbh/Rstar)**(-1/3) # Normalisation (what on the x axis you call \Delta E). It's GM/Rt^2 * Rstar

data = np.loadtxt(f'{abspath}/data/{folder}/_red.csv', delimiter=',', dtype=float)
tfb = data[:, 1]   
Lum = data[:, 2] 
Lum, tfb = sort_list([Lum, tfb], tfb)
tfb_cgs = tfb * t_fall_cgs
dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_cutDen.txt')
tfbdiss, LDiss = dataDiss[0], dataDiss[2] * prel.en_converter/prel.tsol_cgs 
# data8 = np.loadtxt(f'{abspath}/data/{folder}/8_red.csv', delimiter=',', dtype=float)
# tfb8 = data8[:, 1]   
# Lum8 = data8[:, 2] 
# dataDoub = np.loadtxt(f'{abspath}/data/{folder}DoubleRad/DoubleRadrich_red.csv', delimiter=',', dtype=float)
# tfbDou = dataDoub[:, 1]
# Lum_Dou = dataDoub[:, 2]
# tfbDou, Lum_Dou = sort_list([tfbDou, Lum_Dou], tfbDou)

time_theory = tfb[210:-1]
Lum_theory = 5e41*time_theory**(-5/3)

bins = np.loadtxt(f'{abspath}data/{folder}/dMdE__bins.txt')
mid_points = (bins[:-1]+bins[1:])* norm_dMdE/2  # get rid of the normalization
# bins_cgs = bins * (prel.en_converter/prel.Msol_cgs) #  and convert to CGS (they are bins in SPECIFIC orbital energy)
dMdE_distr = np.loadtxt(f'{abspath}data/{folder}/dMdE_.txt')[0] # distribution just after the disruption
bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
plt.plot(bins_tokeep/norm_dMdE, 'o', markersize = 1)

mdot_cgs = np.zeros(len(tfb))
Lacc = np.zeros(len(tfb))
for i, t in enumerate(tfb_cgs):
    if i<5:
        continue
    tsol = t/prel.tsol_cgs # convert in code unit
    # Find the energy of the element at time t
    energy = orb.keplerian_energy(Mbh, prel.G, tsol)
    # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
    i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
    dMdE_t = dMdE_distr_tokeep[i_bin]
    mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
    mfall = mdot # code units
    mdot_cgs[i] = mdot * prel.Msol_cgs / prel.tsol_cgs # [g/s]
    Lacc[i] = 0.1 * np.abs(mdot_cgs[i]) * prel.c_cgs**2

ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo_stat.txt')
tfbRph, Rph = ph_data[0], ph_data[1]
tfbRph, Rph = sort_list([tfbRph, Rph], tfbRph)

#%%
plt.figure()
plt.plot(tfb, np.abs(mdot_cgs)*t_fall_cgs/prel.Msol_cgs, c= 'k')
plt.plot(time_theory, 1e-1*time_theory**(-5/3), c = 'k', linestyle = '--', linewidth = 1)
plt.ylabel(r'$\dot{M}_{\rm fb} [M_{\odot}/t_{\rm fb}]$')
plt.xlabel(r'$t [t_{\rm fb}]$')
plt.yscale('log')
plt.grid()
plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/mdot.png', bbox_inches='tight')

#%%
print(f'Apocenter in AU: {apocenter*prel.Rsol_AU}')
print(f'Last Rph value in Ra: {Rph[-1]/apocenter} \nLast Rph value in AU: {Rph[-1]*prel.Rsol_AU}')
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
img = ax.scatter(tfb, Lum, s = 12, c = Rph/apocenter, cmap = 'viridis', norm = colors.LogNorm(
                 vmin = 1e-2, vmax =3))
cbar = fig.colorbar(img)
cbar.set_label(r'median $R_{\rm ph} [R_{\rm a}]$')#, fontsize = 20)
ax.plot(tfbdiss, LDiss, '--', c= 'gray')
ax.axhline(y=Ledd, c = 'k', linestyle = '-.', linewidth = 2)
ax.plot(time_theory, Lum_theory, c = 'k', linestyle = 'dotted', linewidth = 1)
ax.text(0.1, 1.4*Ledd, r'$L_{\rm Edd}$', fontsize = 20)
ax.text(1.4, 9e40, r'$L\propto t^{-5/3}$', fontsize = 20)
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
ax.set_xticklabels(labels)
ax.tick_params(axis='y', which='major', width = 1.5, length = 5, color = 'k')
ax.tick_params(axis='y', which='minor', width = 1, length = 3, color = 'k')
ax.set_xlim(np.min(tfb), np.max(tfb))
plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/onefld.pdf', bbox_inches='tight')
plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/onefld.png', bbox_inches='tight')


# %%
