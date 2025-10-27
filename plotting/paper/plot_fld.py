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
import Utilities.prelude as prel
import matplotlib.colors as colors
from Utilities.operators import sort_list
from src import orbits as orb
from plotting.paper.IHopeIsTheLast import statistics_photo

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
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
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
dataDissIon = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}ionization.csv', delimiter=',', dtype=float, skiprows=1)
tfbdiss_split, LDissAb, LdissBl =  dataDissIon[:,1], dataDissIon[:,3] * prel.en_converter/prel.tsol_cgs, dataDissIon[:,5] * prel.en_converter/prel.tsol_cgs

time_theory = tfb[210:-1]
Lum_theory = 5e41*time_theory**(-5/3)

_, medianRph, percentile16, percentile84 = statistics_photo(snaps, check)
medianTemprad_ph = np.zeros(len(snaps))
f_ph = np.zeros(len(snaps))
for i, snap in enumerate(snaps):
    x_ph, y_ph, z_ph, vol_ph, den_ph, Temp_ph, RadDen_ph, Vx_ph, Vy_ph, Vz_ph, Press_ph, IE_den_ph, _, _, _, _ = \
        np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    Temprad_ph = (RadDen_ph*prel.en_den_converter/prel.alpha_cgs)**(1/4)  
    r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
    vel_ph = np.sqrt(Vx_ph**2 + Vy_ph**2 + Vz_ph**2)
    mass_ph = den_ph * vol_ph
    oe_ph = orb.orbital_energy(r_ph, vel_ph, mass_ph, params, prel.G)
    bern_ph = orb.bern_coeff(r_ph, vel_ph, den_ph, mass_ph, Press_ph, IE_den_ph, RadDen_ph, params)
    cond_un = bern_ph>=0 # oe_ph>=0
    f_ph[i] = len(oe_ph[np.logical_and(cond_un, r_ph!=0)]) / len(r_ph)  
    medianTemprad_ph[i] = np.median(Temprad_ph)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
img = ax.scatter(tfb, Lum, s = 12, c = medianRph/Rt, cmap = 'viridis', norm = colors.LogNorm(
                 vmin = 1, vmax = 7e1))
cbar = fig.colorbar(img)
cbar.set_label(r'median $R_{\rm ph} [R_{\rm t}]$')#, fontsize = 20)
cbar.ax.tick_params(which='major', length = 5)
cbar.ax.tick_params(which='minor', length = 3)
ax.plot(tfbdiss_split, LdissBl, ls = 'dotted', c= 'b', label = r'$T_{\rm{gas}} < 5\cdot 10^4 K$')
ax.plot(tfbdiss_split, LDissAb, '--', c= 'r', label = r'$T_{\rm{gas}} > 5\cdot 10^4 K$')
ax.axhline(y=Ledd_cgs, c = 'k', linestyle = '-.', linewidth = 2)
ax.text(0.15, 1.4*Ledd_cgs, r'$L_{\rm Edd}$', fontsize = 20)
# ax.plot(time_theory, Lum_theory, c = 'k', linestyle = 'dotted', linewidth = 1)
# ax.text(1.4, 9e40, r'$L\propto t^{-5/3}$', fontsize = 20)
ax.set_yscale('log')
ax.set_ylim(9e37, 8e42)
ax.set_ylabel(r'Luminosity [erg/s]')#, fontsize = 20)
ax.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 20)
ax.grid()
ax.legend(fontsize = 16)
original_ticks = ax.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
ax.set_xticks(new_ticks)
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
# ax.set_xticklabels(labels)
ax.tick_params(axis='both', which='major', width = 1.2, length = 9, color = 'k')
ax.tick_params(axis='y', which='minor', width = 1, length = 5, color = 'k')
ax.set_xlim(np.min(tfb), np.max(tfb))
plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/onefld_ioniz.pdf', bbox_inches='tight')

# %%
fig, (axR, axL) = plt.subplots(1, 2, figsize=(16, 7))
axR.plot(tfb, percentile84/Rt, c = 'k', alpha = 0.3, linestyle = '--')
axR.plot(tfb, percentile16/Rt, c = 'k', alpha = 0.3, linestyle = '--')
img = axR.scatter(tfb, medianRph/Rt, c = f_ph, s = 12, cmap = 'viridis', vmin = 0, vmax = 1)
cbar = fig.colorbar(img, orientation = 'horizontal')
cbar.set_label(r'f = $N_{\rm ph, unbound}/N_{\rm obs}$')
cbar.ax.tick_params(which='major', length = 5)
cbar.ax.tick_params(which='minor', length = 3) 
axR.set_ylabel(r'median $r_{\rm ph} [r_{\rm t}]$')
axR.axhline(apo/Rt, c = 'k', linestyle = '-.', linewidth = 2)
axR.text(0.11, 1.1*apo/Rt, r'$r_{\rm a}$', fontsize = 20)

# img = axL.scatter(tfb, Lum, s = 12, c = medianRph/Rt, cmap = 'viridis', norm = colors.LogNorm(
#                  vmin = 1, vmax = 7e1))
img = axL.scatter(tfb, Lum, s = 12, c = medianTemprad_ph*1e-4, cmap = 'viridis', vmin = 1, vmax = 5)
cbar = fig.colorbar(img, orientation = 'horizontal')
cbar.set_label(r'median $T_{\rm rad, ph} [10^4 K]$')#, fontsize = 20)
cbar.ax.tick_params(which='major', length = 5)
cbar.ax.tick_params(which='minor', length = 3)
axL.plot(tfbdiss, LDiss, '--', c= 'gray')
axL.axhline(y=Ledd_cgs, c = 'k', linestyle = '-.', linewidth = 2)
axL.text(0.15, 1.4*Ledd_cgs, r'$L_{\rm Edd}$', fontsize = 20)
# ax.plot(time_theory, Lum_theory, c = 'k', linestyle = 'dotted', linewidth = 1)
# ax.text(1.4, 9e40, r'$L\propto t^{-5/3}$', fontsize = 20)
original_ticks = axR.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
for ax in [axR, axL]:
    ax.set_yscale('log')
    ax.set_xticks(new_ticks)
    ax.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 20)
    ax.grid()
    # ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width = 1.2, length = 9, color = 'k')
    ax.tick_params(axis='y', which='minor', width = 1, length = 5, color = 'k')
    ax.set_xlim(np.min(tfb), np.max(tfb))
axR.set_ylim(1, 1.5e2)
axL.set_ylabel(r'Luminosity [erg/s]')#, fontsize = 20)
axL.set_ylim(9e37, 2e43)
plt.tight_layout()
plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/onefld.pdf', bbox_inches='tight')
# %%
