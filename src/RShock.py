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
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import sort_list, find_ratio

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
cut = '' # '' or 'cutCoord' or 'cutDenCoord'

tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt/beta
norm_dMdE = Mbh/Rt * (Mbh/Rstar)**(-1/3) # Normalisation (what on the x axis you call \Delta E). It's GM/Rt^2 * Rstar
apo = orb.apocentre(Rstar, mstar, Mbh, beta) 
amin = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)

#
## FUNCTIONS
#
def efficiency_shock(Lum, Mdot, const_c):
    eta_sh = Lum/(np.abs(Mdot) * const_c**2) 
    return eta_sh

def R_shock(Mbh, eta_sh, const_G, const_c):
    R_sh = const_G * Mbh / (const_c**2 * eta_sh)
    return R_sh

def eta_from_R(Mbh, R_sh, const_G, const_c):
    eta = const_G * Mbh / (R_sh * const_c**2)
    return eta

##
# MAIN
#%%
ph_data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/photo_mean.txt')
tfbRph, Rph = ph_data[0], ph_data[1]
time_array_yr = np.linspace(1e-1,2, 100) # yr
time_yr_cgs = time_array_yr * 365 * 24 * 3600 # converted to seconds

checks = ['LowRes', '', 'HiRes' ]
checkslegend = ['Low', 'Fid', 'High']
colorslegend = ['C1', 'yellowgreen', 'darkviolet']

# list of arrays. Each line contains the data for one resolution
tfb_all = []
mfall_all = []
eta_shL_all = []
R_sh_all = []
eta_shL_diss_all = []
R_shDiss_all = []
RDiss_all = []
timeRDiss_all = []
Eradtot_all = []
LDiss_all = []

for j, check in enumerate(checks):
    # Load data
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    datadays = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_days.txt')
    snaps, tfb = datadays[0], datadays[1]
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_bins.txt')
    mid_points = (bins[:-1]+bins[1:])* norm_dMdE/2  # get rid of the normalization
    # bins_cgs = bins * (prel.en_converter/prel.Msol_cgs) #  and convert to CGS (they are bins in SPECIFIC orbital energy)
    dMdE_distr = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
    dataLum = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}/{check}rich_red.csv', delimiter=',', dtype=float)
    snapsLum, tfbLum, Lum = dataLum[:, 0], dataLum[:, 1], dataLum[:, 2] 
    dataDiss = np.loadtxt(f'{abspath}data/{folder}/Rdiss_{check}{cut}.txt')
    timeRDiss, RDiss, Eradtot, LDiss = dataDiss[0], dataDiss[1], dataDiss[2], dataDiss[3] 
    if check == 'LowRes':
        timeRDiss = np.concatenate([timeRDiss[:127], timeRDiss[129:]])
        RDiss = np.concatenate([RDiss[:127], RDiss[129:]])

    timeRDiss_all.append(timeRDiss)
    RDiss_all.append(RDiss)
    Eradtot_all.append(Eradtot)
    LDiss_all.append(LDiss)
  
    mfall = np.zeros(len(tfb_cgs))
    eta_sh = np.zeros(len(tfb_cgs))
    R_sh = np.zeros(len(tfb_cgs))
    eta_sh_diss = np.zeros(len(tfb_cgs))
    R_shDiss = np.zeros(len(tfb_cgs))

    # compute Rshock and eta_shock in the simualtion time range
    for i, t in enumerate(tfb_cgs):
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol)
        # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        if energy/bins_tokeep[i_bin] > 2:
            print('You do not match the data in your time range')
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
        mfall[i] = mdot # code units
        mdot_cgs = mdot * prel.Msol_cgs / prel.tsol_cgs # [g/s]
        # Find the luminosity at time t
        idx = np.argmin(np.abs(tfb[i]-tfbLum)) # just to be sure that you match the data
        # idx = np.argmin(np.abs(snaps[i]-snapsLum)) # just to be sure that you match the data
        Lum_t = Lum[idx]
        eta_sh[i] = efficiency_shock(Lum_t, mdot_cgs, prel.c_cgs) # [CGS]
        R_sh[i] = R_shock(Mbh_cgs, eta_sh[i], prel.G_cgs, prel.c_cgs) # [CGS]
        # idxDiss = np.argmin(np.abs(tfb[i]-timeRDiss)) # just to be sure that you match the data
        # Lum_diss = np.abs(LDiss[idxDiss]) * prel.en_converter / prel.tsol_cgs # [CGS]
        # eta_sh_diss[i] = efficiency_shock(Lum_diss, mdot_cgs, prel.c_cgs) # [CGS]
        # R_shDiss[i] = R_shock(Mbh_cgs, eta_sh_diss[i], prel.G_cgs, prel.c_cgs) # [CGS]

    tfb_all.append(tfb)
    mfall_all.append(mfall)
    eta_shL_all.append(eta_sh)    
    R_sh_all.append(R_sh/(prel.Rsol_cgs))
    # eta_shL_diss_all.append(eta_sh_diss)
    # R_shDiss_all.append(R_shDiss)

# ratioDissL = find_ratio(RDiss_all[0], RDiss_all[1][:len(RDiss_all[0])])
# ratioDissL = find_ratio(RDiss_all[1][1:len(RDiss_all[2])+1], RDiss_all[2])
Rlim_min = 1e11/(apo*prel.Rsol_cgs)
Rlim_max = 1e16/(apo*prel.Rsol_cgs) 
#
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
# fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
# ax0.plot(tfb_all[1][-90:], 1e-2*(tfb_all[1][-90:])**(-5/3), c = 'k', alpha = 0.5, linestyle = '--')
for i, check in enumerate(checks):
    # Plot
    mfall_toplot_days = mfall_all[i] / (prel.tsol_cgs / (3600*24)) # convert to Msol/days
    mfall_toplot = mfall_toplot_days / tfallback
    # ax0.plot(tfb_all[i], np.abs(mfall_toplot), label = checkslegend[i], color = colorslegend[i])

    # ax1.plot(tfb_all[i], R_shDiss_all[i], linestyle = 'dotted', color = colorslegend[i])#, label = f'Rdiss {checkslegend[i]}')
    if i == 2:
        ax1.plot(tfb_all[i], R_sh_all[i]/apo, color = colorslegend[i], label = r'$R_{\rm sh}$')
        ax1.plot(timeRDiss_all[i], RDiss_all[i]/apo, linestyle = '--', color = colorslegend[i], label = r'$R_{\rm diss}$')
    else:
        ax1.plot(tfb_all[i], R_sh_all[i]/apo, color = colorslegend[i])
        ax1.plot(timeRDiss_all[i], RDiss_all[i]/apo, linestyle = '--', color = colorslegend[i])

    # ax1.scatter(tfb_all[i], R_sh_all[i], c = np.log10(eta_shL_all[i]), marker = markerslegend[i], cmap = 'viridis', vmin = -7, vmax = -1, label = checkslegend[i])
    # # ax1.scatter(tfb_all[i], R_shDiss_all[i], linestyle = 'dotted', color = eta_shL_diss_all[i])#, label = f'Rdiss {checkslegend[i]}')
    # ax1.axhline(y=Rt*prel.Rsol_cgs, color = 'k', linestyle = 'dotted')
    # ax1.text(1.8, .4* Rt*prel.Rsol_cgs, r'$R_{\rm t}$', fontsize = 20, color = 'k')
    # ax1.plot(timeRDiss_all[i], RDiss_all[i] * prel.Rsol_cgs, linestyle = '--', color = colorslegend[i])#, label = f'Rdiss {checkslegend[i]}')

ax1.axhline(y=Rp/apo, color = 'k', linestyle = 'dotted')
ax1.text(1.85, .4* Rp/apo, r'$R_{\rm p}$', fontsize = 20, color = 'k')
ax1.axhline(y=amin/apo, color = 'k', linestyle = '--')
ax1.text(1.85, .4* amin/apo, r'$a_{\rm mb}$', fontsize = 20, color = 'k')
ax1.plot(tfbRph, Rph/apo, color = 'k', label = r'$R_{\rm ph}$')
# ax1.axhline(y=apo/apo, color = 'r', linestyle = 'dotted')
# ax0.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot/t_{\rm fb}$]', fontsize = 18)
ax1.set_ylabel(r'$R [R_{\rm a}$]')#, fontsize = 18)
ax1.set_ylim(Rlim_min, Rlim_max)
# Set primary y-axis ticks
R_ticks = np.logspace(np.log10(Rlim_min), np.log10(Rlim_max), num=5)
ax1.set_yticks(R_ticks)
ax1.set_yticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in R_ticks])

ax2 = ax1.twinx()
eta_ticks = eta_from_R(Mbh_cgs, R_ticks[::-1]*apo*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_min = eta_from_R(Mbh_cgs, Rlim_max*apo*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_max = eta_from_R(Mbh_cgs, Rlim_min*apo*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
ax2.set_yticks(eta_ticks)
ax2.set_ylim(etalim_max, etalim_min)
ax2.set_ylabel(r'$\eta_{\rm sh}$')#, fontsize = 18)

original_ticks = ax1.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.set_xlabel(r't [$t_{\rm fb}$]')#, fontsize = 20)
    if ax!=ax2:
        ax.grid()
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', width=0.7, length=7)
        ax.tick_params(axis='x', which='minor', width=0.5, length=5)
    ax.set_xlim(0, 2)
ax1.legend(fontsize = 16, loc = 'upper right')

plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/Reta{cut}.pdf')
plt.savefig(f'{abspath}/Figs/multiple/Reta{cut}.png')

