""" Plots for FLD light curve for 1.paperEdd and to check ionization"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/home/martirep/data1/postprocess/tde'
else:
    abspath = '/Users/paolamartire/shocks'
import csv
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
import matplotlib.colors as colors
from Utilities.operators import sort_list
from src import orbits as orb
from plotting.paperEdd.IHopeIsTheLast import statistics_photo, statistics_col

##
# PARAMETERS
#%%
m = 5
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
if m == 4:
    Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
else:
    Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.5/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, int(beta)]
things = orb.get_things_about(params)
apo = things['apo']
Rt = things['Rt']
t_fall = things['t_fb_days']
t_fall_cgs = t_fall * 24 * 3600
omega_minus1 = np.sqrt(Rt**3/(prel.G*Mbh))
# print('t_visc', prel.tsol_cgs/t_fall_cgs * omega_minus1 / 0.02)
print('orb period in t_fb: ', 2*np.pi*omega_minus1*prel.tsol_cgs/t_fall_cgs)
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)
idx_maxLum = np.argmax(Lum)

if m ==4:
    dataDiss = np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
    dataDissIon = np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/Rdiss_{check}ionizationHe.csv', delimiter=',', dtype=float, skiprows=1)
    tfbdiss_split, LDissAb, LdissBl =  dataDissIon[:,1], dataDissIon[:,3] * prel.en_converter/prel.tsol_cgs, dataDissIon[:,5] * prel.en_converter/prel.tsol_cgs
    _, tfbmdot, mfallH, _, _, _, _, _, tot_IE_H, tot_Rad_H = \
                np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/wind/Mdot_HiResNewAMR05aminmean.csv', 
                        delimiter = ',', 
                        skiprows=1, 
                        unpack=True) 
    Lum_mdot = 0.1 * np.abs(mfallH) * prel.csol_cgs**2 *prel.en_converter/prel.tsol_cgs 
    time_theory = tfb[210:-1]
    Lum_theory = 5e41*time_theory**(-5/3)
else:
    dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
snapdiss, tfbdiss, LDiss = dataDiss[:,0], dataDiss[:,1], dataDiss[:,3] * prel.en_converter/prel.tsol_cgs

meanRph, medianRph, percentile16, percentile84 = statistics_photo(snaps, np.concatenate((params, [n, compton, check])))
meancol, mediancol, _, _ = statistics_col(snaps, np.concatenate((params, [n, compton, check])))
medianTemprad_ph = np.zeros(len(snaps))
f_ph = np.zeros(len(snaps))
for i, snap in enumerate(snaps):
    # x_ph, y_ph, z_ph, vol_ph, den_ph, Temp_ph, RadDen_ph, Vx_ph, Vy_ph, Vz_ph, Press_ph, IE_den_ph, alpha_ph, _, _, _ = \
    #     np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    x_ph, y_ph, z_ph, den_ph, vol_ph, RadDen_ph, Temp_ph, Vx_ph, Vy_ph, Vz_ph, Press_ph, IE_den_ph, alpha_ph= \
        photo['x'], photo['y'], photo['z'], photo['den'], photo['vol'], photo['radden'], photo['temp'], photo['vx'], photo['vy'], photo['vz'], photo['P'], photo['ieden'], photo['alpha_rossland']

    Temprad_ph = (RadDen_ph*prel.en_den_converter/prel.alpha_cgs)**(1/4)  
    if i == idx_maxLum:
        print('max median T_rad_ph:', np.median(Temprad_ph))
        kappa_ph = alpha_ph / den_ph
        one_over_kappa_ph = (np.mean(1/kappa_ph))#**(-1)
        kappa = one_over_kappa_ph**(-1)
        print('kappa at max L:', kappa)
    r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
    vel_ph = np.sqrt(Vx_ph**2 + Vy_ph**2 + Vz_ph**2)
    mass_ph = den_ph * vol_ph
    oe_ph = orb.orbital_energy(r_ph, vel_ph, mass_ph, params, prel.G)
    bern_ph = orb.bern_coeff(r_ph, vel_ph, den_ph, mass_ph, Press_ph, IE_den_ph, RadDen_ph, params)
    cond_un = bern_ph>=0 # oe_ph>=0
    f_ph[i] = len(oe_ph[np.logical_and(cond_un, r_ph!=0)]) / len(r_ph)  
    medianTemprad_ph[i] = np.median(Temprad_ph)

#%%
if m ==4:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    img = ax.scatter(tfb, Lum, s = 12, c = medianRph/Rt, cmap = 'viridis', norm = colors.LogNorm(
                    vmin = 1, vmax = 7e1))
    cbar = fig.colorbar(img)
    cbar.set_label(r'median $(r_{\rm ph}/r_{\rm t})$')#, fontsize = 20)
    cbar.ax.tick_params(which='major', length = 7, width = 1)
    cbar.ax.tick_params(which='minor', length = 4, width = .6)
    ax.plot(tfbdiss, LDiss,c = 'gray', ls = '--', label = r'$L_{\rm diss}$')
    ax.plot(tfbdiss_split, LdissBl, ls = 'dotted', c= 'b', label = r'$T_{\rm{gas}} < 1\cdot 10^5 K$')
    ax.plot(tfbdiss_split, LDissAb, '--', c= 'r', label = r'$T_{\rm{gas}} > 1\cdot 10^5 K$')
    ax.plot(tfbmdot, Lum_mdot*1e-5, ls = 'dashdot', c= 'orange', label = r'from $\dot{M}_{\rm fb}$ (scaled by $10^{-5}$)')
    ax.axhline(y=Ledd_cgs, c = 'k', linestyle = '-.', linewidth = 2)
    ax.text(0.15, 1.4*Ledd_cgs, r'$L_{\rm Edd} (\kappa_{\rm p})$', fontsize = 20)
    # ax.plot(time_theory, Lum_theory, c = 'k', linestyle = 'dotted', linewidth = 1)
    # ax.text(1.4, 9e40, r'$L\propto t^{-5/3}$', fontsize = 20)
    ax.set_yscale('log')
    ax.set_ylim(9e37, 1e43)
    ax.set_ylabel(r'Luminosity (erg/s)')
    ax.set_xlabel(r'$t / t_{\rm fb}$')
    ax.grid()
    original_ticks = ax.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    ax.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width = 1.2, length = 9, color = 'k')
    ax.tick_params(axis='y', which='minor', width = 1, length = 5, color = 'k')
    ax.set_xlim(np.min(tfb), np.max(tfb))
    ax.legend(fontsize = 16)
    plt.savefig(f'/Users/paolamartire/shocks/Figs/1.paperEdd/onefld_ionizz.jpg', bbox_inches='tight')

#%%
fig, (axR, axL) = plt.subplots(1, 2, figsize=(16, 7))
axR.set_ylabel(r'median $(r_{\rm ph}/r_{\rm t})$')
axR.axhline(apo/Rt, c = 'k', linestyle = '-.', linewidth = 2)
axR.text(0.11, 1.1*apo/Rt, r'$r_{\rm a}$', fontsize = 20)
img = axL.scatter(tfb, Lum, s = 12, c = medianTemprad_ph*1e-4, cmap = 'viridis', vmin = 1, vmax = 10)
cbar = fig.colorbar(img, orientation = 'horizontal')
cbar.set_label(r'median $T_{\rm rad, ph} (10^4 K)$')#, fontsize = 20)
cbar.ax.tick_params(which='major', length = 5)
cbar.ax.tick_params(which='minor', length = 3)
axR.plot(tfb[2:], percentile84[2:]/Rt, c = 'k', alpha = 0.3, linestyle = '--')
axR.plot(tfb, percentile16/Rt, c = 'k', alpha = 0.3, linestyle = '--')
img = axR.scatter(tfb, medianRph/Rt, c = f_ph, s = 12, cmap = 'plasma', vmin = 0, vmax = 1)
cbar = fig.colorbar(img, orientation = 'horizontal')
cbar.set_label(r'$f\equiv N_{\rm ph, unbound}/N_{\rm obs}$')
cbar.ax.tick_params(which='major', length = 5)
cbar.ax.tick_params(which='minor', length = 3) 
axL.text(0.15, 1.4*Ledd_cgs, r'$L_{\rm Edd} (\kappa_{\rm p})$', fontsize = 20)
axL.set_ylim(9e37, 4e43)
axL.plot(tfbdiss, LDiss, '--', c= 'gray', label = r'$L_{\rm diss}$')
# else:
#     axR.plot(tfb, medianRph/Rt, c ='k')
#     axL.plot(tfb, Lum, c = 'k')
#     axL.set_ylim(9e37, 5e43)
#     axL.text(0.15, 1.4*Ledd_cgs, r'$L_{\rm Edd} (\sigma_{\rm T})$', fontsize = 20)
axL.axhline(y=Ledd_cgs, c = 'k', linestyle = '-.', linewidth = 2)
original_yticks = axR.get_yticks()
ymidpoints = (original_yticks[:-1] + original_yticks[1:]) / 2
yticks = np.sort(np.concatenate([original_yticks, ymidpoints]))
ylabels = [f"{y:.0e}" if y in original_yticks else "" for y in yticks]
cm_ticks = yticks * Rt * prel.Rsol_cgs
cm_labels = [f"{t:.0e}" if y in original_yticks else "" for y, t in zip(yticks, cm_ticks)]
axR.set_yticks(yticks, ylabels)
axR.set_ylim(1, 1.5e2)
ax_cm = axR.twinx()
ax_cm.set_yticks(cm_ticks, cm_labels)
ax_cm.set_ylim(1 * Rt * prel.Rsol_cgs, 1.5e2 * Rt * prel.Rsol_cgs)
original_ticks = axR.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
for ax in [axR, axL, ax_cm]:
    ax.set_yscale('log')
    ax.set_xticks(new_ticks)
    ax.set_xlabel(r't / t$_{\rm fb}$')#, fontsize = 20)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width = 1, length = 7, color = 'k')
    ax.tick_params(axis='y', which='minor', width = 1, length = 4, color = 'k')
    ax.set_xlim(-.1, np.max(tfb))
    ax.grid()
axL.set_ylabel(r'Luminosity (erg/s)')#, fontsize = 20)
plt.tight_layout()
if m == 4:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/1.paperEdd/onefld.pdf', bbox_inches='tight')
else:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/onefld.png', bbox_inches='tight')
print('L_max/L_Edd', np.max(Lum)/Ledd_cgs, 'at snap', snaps[np.argmax(Lum)], f'({tfb[np.argmax(Lum)]} t_fb)')
#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(tfb, meanRph/Rt, ls = '--', label = r'mean $r_{\rm ph}$')
ax.scatter(tfb, meancol/Rt, label = r'mean $r_{\rm th}$')
ax.plot(tfb, medianRph/Rt, ls = '--',label = r'median $r_{\rm ph}$')
ax.scatter(tfb, mediancol/Rt, label = r'median $r_{\rm th}$')
ax.set_ylabel(r' $r/r_{\rm t}$')
ax.axhline(apo/Rt, c = 'gray', linestyle = '-.', linewidth = 2)
ax.text(0.11, 1.1*apo/Rt, r'$r_{\rm a}$', fontsize = 20)
ax.set_yscale('log')
ax.set_xticks(new_ticks)
ax.set_xlabel(r't / t$_{\rm fb}$')#, fontsize = 20)
ax.set_xticklabels(labels)
ax.tick_params(axis='both', which='major', width = 1, length = 7, color = 'k')
ax.tick_params(axis='y', which='minor', width = 1, length = 4, color = 'k')
ax.set_xlim(-.1, np.max(tfb))
ax.grid()
ax.legend(fontsize = 18)

# %%
