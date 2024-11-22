""" Compute shock efficiency and Rshock from theoretical approximation.
Plot it and compare with Rdissipation."""
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
import src.orbits as orb

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
tfallback = 2.5777261297507925 #days 
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rt = Rstar * (Mbh/mstar)**(1/3)
norm_dMdE = Mbh/Rt * (Mbh/Rstar)**(-1/3) # Normalisation (what on the x axis you call \Delta E). It's GM/Rt^2 * Rstar
time_array_yr = np.linspace(1e-1,2, 100) # yr
time_array_cgs = time_array_yr * 365 * 24 * 3600 # converted to seconds

#
## FUNCTIONS
#
def efficiency_shock(Lum, Mdot, const_c):
    eta_sh = Lum/(np.abs(Mdot) * const_c**2) 
    return eta_sh

def R_shock(Mbh, eta_sh, const_G, const_c):
    R_sh = const_G * Mbh / (const_c**2 * eta_sh)
    return R_sh

##
# MAIN
##

checks = ['DoubleRad', 'LowRes', '', 'HiRes' ]
checkslegend = ['DoubleRad', 'Low', 'Middle', 'High']
colors = ['navy', 'b', 'darkorange', 'dodgerblue']

# Fallback rate for long times
mfall_all_yr = []
tfb_all = []
mfall_all = []
eta_shL_all = []
R_sh_all = []
for j, check in enumerate(checks):
    # Load data
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    datadays = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_days.txt')
    snaps, tfb = datadays[0], datadays[1]
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    databins = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_bins.txt')
    bins = databins[:-1] * norm_dMdE # get rid of the normalization
    # bins_cgs = bins * (prel.en_converter/prel.Msol_cgs) #  and convert to CGS (they are bins in SPECIFIC orbital energy)
    dMdE_distr = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = bins[bins<0], dMdE_distr[bins<0] # keep only the bound energies
    dataLum = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}/{check}_red.csv', delimiter=',', dtype=float)
    snapsLum = dataLum[:, 0]
    tfbLum = dataLum[:, 1]   
    Lum = dataLum[:, 2]  

    # dataLum = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}/{check}_red.csv', delimiter=',', dtype=float)
    # tfbLum, Lum = dataLum[0], dataLum[1] 
  
    mfall_yr = np.zeros(len(time_array_cgs))
    mfall = np.zeros(len(tfb_cgs))
    eta_sh = np.zeros(len(tfb_cgs))
    R_sh = np.zeros(len(tfb_cgs))
    
    for i, t in enumerate(time_array_cgs):
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol)
        # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
        mfall_yr[i] = mdot # code units

    mfall_all_yr.append(mfall_yr)

    # compute Rshock and eta_shock
    for i, t in enumerate(tfb_cgs):
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol)
        # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
        mfall[i] = mdot # code units
        mdot_cgs = mdot * prel.Msol_cgs / prel.tsol_cgs # [CGS]
        # Find the luminosity at time t
        idx = np.argmin(np.abs(tfb[i]-tfbLum)) # just to be sure that you match the data
        # idx = np.argmin(np.abs(snaps[i]-snapsLum)) # just to be sure that you match the data
        Lum_t = Lum[idx]
        eta_sh[i] = efficiency_shock(Lum_t, mdot_cgs, prel.c_cgs) # [CGS]
        R_sh[i] = R_shock(Mbh_cgs, eta_sh[i], prel.G_cgs, prel.c_cgs) # [CGS]

    tfb_all.append(tfb)
    mfall_all.append(mfall)
    eta_shL_all.append(eta_sh)    
    R_sh_all.append(R_sh)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(15, 10))
for i, check in enumerate(checks):
    mfall_toplot = mfall_all_yr[i] / (prel.tsol_cgs / (3600*24*365)) # convert to Msol/yr
    ax0.plot(time_array_yr, np.abs(mfall_toplot), label = checkslegend[i], color = colors[i])
    ax0.plot(time_array_yr, time_array_yr**(-5/3)/100, label = r'$t^{-5/3}$', color = 'k', linestyle = '--')

for i, check in enumerate(checks):
    # Load data RDiss
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    dataDiss = np.loadtxt(f'{abspath}data/{folder}/Rdiss_{check}.txt')
    timeDiss, RDiss = dataDiss[0], dataDiss[1]
    # Plot
    mfall_toplot_days = mfall_all[i] / (prel.tsol_cgs / (3600*24)) # convert to Msol/days
    mfall_toplot = mfall_toplot_days / tfallback
    ax2.plot(tfb_all[i], np.abs(mfall_toplot), label = checkslegend[i], color = colors[i])

    ax1.plot(tfb_all[i], eta_shL_all[i], label = checkslegend[i], color = colors[i])

    ax3.plot(tfb_all[i], R_sh_all[i], label = checkslegend[i], color = colors[i])
    ax3.axhline(y=Rt*prel.Rsol_cgs, color = 'k', linestyle = 'dotted')
    ax3.plot(timeDiss, np.abs(RDiss) * prel.Rsol_cgs, linestyle = '--', color = colors[i])#, label = f'Rdiss {checkslegend[i]}')

ax0.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot$/yr]', fontsize = 15)
ax2.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot/t_{\rm fb}$]', fontsize = 15)
ax1.set_ylabel(r'$\eta_{\rm sh}$', fontsize = 18)
ax1.set_ylim(1e-8, 1e-3)
ax3.set_ylabel(r'$R$ [cm]', fontsize = 15)
ax3.set_ylim(1e11, 1e17)

for ax in [ax0, ax1, ax2, ax3]:
    ax.grid()
    if ax == ax0:
        ax.set_xlabel(r't [yr]', fontsize = 20)
        ax.loglog()
    else:
        ax.set_yscale('log')
        ax.set_xlabel(r't [t$_{\rm fb}$]', fontsize = 20)
        ax.set_xlim(0, 1.75)
        # ax.minorticks_on()  # Enable minor ticks
        original_ticks = ax.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', width=0.7, length=7)
    ax.tick_params(axis='x', which='minor', width=0.5, length=5)

ax2.legend(fontsize = 18)
plt.tight_layout()
