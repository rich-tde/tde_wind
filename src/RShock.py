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
Ledd = 1.26e38 * Mbh # erg/s
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

checks = ['LowRes', '', 'HiRes', 'DoubleRad']
checkslegend = ['LowRes', 'Middle', 'HiRes', 'DoubleRad']
colors = ['darkorange', 'r', 'dodgerblue', 'navy']

mfall_all_yr = []
for j, check in enumerate(checks):
    # Load data
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    datadays = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_days.txt')
    snaps, tfb = datadays[0], datadays[1]
    databins = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_bins.txt')
    bins = databins[:-1] * norm_dMdE # get rid of the normalization
    # bins_cgs = bins * (prel.en_converter/prel.Msol_cgs) #  and convert to CGS (they are bins in SPECIFIC orbital energy)
    dMdE_distr = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = bins[bins<0], dMdE_distr[bins<0] # keep only the bound energies
    dataLum = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}/{check}_red.csv', delimiter=',', dtype=float)
    snapsLum = dataLum[:, 0]
    tfbLum = dataLum[:, 1]   
    Lum = dataLum[:, 2]   
    
    mfall_yr = np.zeros(len(time_array_cgs))
    # compute Rshock and eta_shock
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

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(8, 6))
for i, check in enumerate(checks):
    mfall_toplot = mfall_all_yr[i] / (prel.tsol_cgs / (3600*24*365)) # convert to Msol/yr
    ax0.plot(time_array_yr, np.abs(mfall_toplot), label = checkslegend[i], color = colors[i])
    ax0.plot(time_array_yr, time_array_yr**(-5/3), label = r'$t^{-5/3}$', color = 'k', linestyle = '--')
ax0.set_xlabel(r't [yr]', fontsize = 15)
ax0.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot$/yr]', fontsize = 15)
ax0.loglog()
ax0.grid()
ax0.tick_params(axis='both', which='minor', length=4, width=.5)  # Make minor ticks larger
ax0.tick_params(axis='both', which='major', length=5, width=1)  # Make minor ticks larger


#%%
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
    
    eta_sh = np.zeros(len(tfb_cgs))
    R_sh = np.zeros(len(tfb_cgs))
    mfall = np.zeros(len(tfb_cgs))
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
        idx = np.argmin(np.abs(snaps[i]-snapsLum)) # just to be sure that you match the data
        Lum_t = Lum[idx]
        eta_sh[i] = efficiency_shock(Lum_t, mdot_cgs, prel.c_cgs) # [CGS]
        R_sh[i] = R_shock(Mbh_cgs, eta_sh[i], prel.G_cgs, prel.c_cgs) # [CGS]

    tfb_all.append(tfb)
    mfall_all.append(mfall)
    eta_shL_all.append(eta_sh)    
    R_sh_all.append(R_sh)

for i, check in enumerate(checks):
    # Load data RDiss
    if check == '':
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        dataDiss = np.loadtxt(f'{abspath}data/{folder}/Rdiss_{check}.txt')
        timeDiss, RDiss = dataDiss[0], dataDiss[1]
    # Plot
    mfall_toplot_days = mfall_all[i] / (prel.tsol_cgs / (3600*24)) # convert to Msol/days
    mfall_toplot = mfall_toplot_days / tfallback
    ax1.plot(tfb_all[i], np.abs(mfall_toplot), label = checkslegend[i], color = colors[i])
    ax1.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot/t_{\rm fb}$]', fontsize = 15)
    ax1.grid()

    ax2.plot(tfb_all[i], eta_shL_all[i], label = checkslegend[i], color = colors[i])
    ax2.set_ylabel(r'$\eta_{\rm sh}$', fontsize = 18)
    ax2.set_ylim(1e-8, 1e-3)

    ax3.plot(tfb_all[i], R_sh_all[i], label = checkslegend[i], color = colors[i])
    if check == '':
        ax3.plot(timeDiss, np.abs(RDiss)   * prel.Rsol_cgs, linestyle = '--', color = colors[i])#, label = f'Rdiss {checkslegend[i]}')
    ax3.set_ylabel(r'$R$ [cm]', fontsize = 15)
    ax3.set_ylim(1e12, 1e17)

for ax in [ax1, ax2, ax3]:
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel(r't [t$_{\rm fb}$]', fontsize = 15)
    ax.minorticks_on()  # Enable minor ticks
    ax.tick_params(axis='both', which='minor', length=4, width=.5)  # Make minor ticks larger
    ax.tick_params(axis='both', which='major', length=5, width=1)  # Make minor ticks larger

ax2.legend(fontsize = 10)
plt.tight_layout()
