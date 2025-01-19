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
from Utilities.operators import sort_list

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

tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rt = Rstar * (Mbh/mstar)**(1/3)
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
##
ph_data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/photo_mean.txt')
tfbRph, Rph = ph_data[0], ph_data[3]
time_array_yr = np.linspace(1e-1,2, 100) # yr
time_yr_cgs = time_array_yr * 365 * 24 * 3600 # converted to seconds

checks = ['LowRes', '', 'HiRes' ]
checkslegend = ['Low', 'Fid', 'High']
colorslegend = ['C1', 'yellowgreen', 'darkviolet']

# list of arrays. Each line contains the data for one resolution
mfall_all_yr = []
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
    dataDiss = np.loadtxt(f'{abspath}data/{folder}/Rdiss_{check}.txt')
    timeRDiss, RDiss, Eradtot, LDiss = dataDiss[0], dataDiss[1], dataDiss[2], dataDiss[3] 

    timeRDiss_all.append(timeRDiss)
    RDiss_all.append(RDiss)
    Eradtot_all.append(Eradtot)
    LDiss_all.append(LDiss)
  
    mfall_yr = np.zeros(len(time_yr_cgs))
    mfall = np.zeros(len(tfb_cgs))
    eta_sh = np.zeros(len(tfb_cgs))
    R_sh = np.zeros(len(tfb_cgs))
    eta_sh_diss = np.zeros(len(tfb_cgs))
    R_shDiss = np.zeros(len(tfb_cgs))
    
    # Fallback rate in a longer range of time
    for i, t in enumerate(time_yr_cgs):
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t [all in code units]
        energy = orb.keplerian_energy(Mbh, prel.G, tsol)
        # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        if energy/bins_tokeep[i_bin] > 2:
            print('You do not match the data')
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
        mfall_yr[i] = mdot # code units

    mfall_all_yr.append(mfall_yr)

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
    R_sh_all.append(R_sh)
    # eta_shL_diss_all.append(eta_sh_diss)
    # R_shDiss_all.append(R_shDiss)


Rlim_min = 1e11
Rlim_max = 1e16   
#%%
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
# ax0.plot(tfb_all[1][-90:], 1e-2*(tfb_all[1][-90:])**(-5/3), c = 'k', alpha = 0.5, linestyle = '--')
for i, check in enumerate(checks):
    # Plot
    mfall_toplot_days = mfall_all[i] / (prel.tsol_cgs / (3600*24)) # convert to Msol/days
    mfall_toplot = mfall_toplot_days / tfallback
    # ax0.plot(tfb_all[i], np.abs(mfall_toplot), label = checkslegend[i], color = colorslegend[i])

    # ax1.plot(tfb_all[i], R_shDiss_all[i], linestyle = 'dotted', color = colorslegend[i])#, label = f'Rdiss {checkslegend[i]}')
    if i == 2:
        ax1.plot(tfb_all[i], R_sh_all[i], color = colorslegend[i], label = r'$R_{\rm sh}$')
        ax1.plot(timeRDiss_all[i], RDiss_all[i] * prel.Rsol_cgs, linestyle = '--', color = colorslegend[i], label = r'$R_{\rm diss}$')
    else:
        ax1.plot(tfb_all[i], R_sh_all[i], color = colorslegend[i])
        ax1.plot(timeRDiss_all[i], RDiss_all[i] * prel.Rsol_cgs, linestyle = '--', color = colorslegend[i])

    # ax1.scatter(tfb_all[i], R_sh_all[i], c = np.log10(eta_shL_all[i]), marker = markerslegend[i], cmap = 'viridis', vmin = -7, vmax = -1, label = checkslegend[i])
    # # ax1.scatter(tfb_all[i], R_shDiss_all[i], linestyle = 'dotted', color = eta_shL_diss_all[i])#, label = f'Rdiss {checkslegend[i]}')
    # ax1.axhline(y=Rt*prel.Rsol_cgs, color = 'k', linestyle = 'dotted')
    # ax1.text(1.8, .4* Rt*prel.Rsol_cgs, r'$R_{\rm t}$', fontsize = 20, color = 'k')
    # ax1.plot(timeRDiss_all[i], RDiss_all[i] * prel.Rsol_cgs, linestyle = '--', color = colorslegend[i])#, label = f'Rdiss {checkslegend[i]}')

ax1.axhline(y=Rt*prel.Rsol_cgs, color = 'k', linestyle = 'dotted')
ax1.text(1.85, .4* Rt*prel.Rsol_cgs, r'$R_{\rm t}$', fontsize = 20, color = 'k')
ax1.axhline(y=amin*prel.Rsol_cgs, color = 'k', linestyle = '--')
ax1.text(1.85, .4* amin*prel.Rsol_cgs, r'$a_{\rm mb}$', fontsize = 20, color = 'k')
ax1.plot(tfbRph, Rph * prel.Rsol_cgs, color = 'k', label = r'$R_{\rm ph}$')
# ax1.axhline(y=apo*prel.Rsol_cgs, color = 'r', linestyle = 'dotted')
# ax0.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot/t_{\rm fb}$]', fontsize = 18)
ax1.set_ylabel(r'$R$ [cm]', fontsize = 18)
ax1.set_ylim(Rlim_min, Rlim_max)
# Set primary y-axis ticks
R_ticks = np.logspace(np.log10(Rlim_min), np.log10(Rlim_max), num=5)
ax1.set_yticks(R_ticks)
ax1.set_yticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in R_ticks])

ax2 = ax1.twinx()
eta_ticks = eta_from_R(Mbh_cgs, R_ticks[::-1], prel.G_cgs, prel.c_cgs)
etalim_min = eta_from_R(Mbh_cgs, Rlim_max, prel.G_cgs, prel.c_cgs)
etalim_max = eta_from_R(Mbh_cgs, Rlim_min, prel.G_cgs, prel.c_cgs)
ax2.set_yticks(eta_ticks)
ax2.set_ylim(etalim_max, etalim_min)
ax2.set_ylabel(r'$\eta_{\rm sh}$', fontsize = 18)

original_ticks = ax1.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.set_xlabel(r't [$t_{\rm fb}$]', fontsize = 20)
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
plt.savefig(f'{abspath}/Figs/multiple/Reta.pdf')

#%% Which light 
Lc = []
tLc = []
dataL = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/LowResrich_red.csv', delimiter=',', dtype=float)
data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/rich_red.csv', delimiter=',', dtype=float)
dataH = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/HiResrich_red.csv', delimiter=',', dtype=float)
tLc.append(dataL[:, 1])
tLc.append(data[:, 1])
tLc.append(dataH[:, 1]) 
Lc.append(dataL[:, 2])
Lc.append(data[:, 2])
Lc.append(dataH[:, 2])
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
t_fall_cgs = t_fall * 24 * 3600

fig, ax = plt.subplots(3,1,figsize=(12, 10))
for i in range(len(checks)):
    Lumrad_i = Eradtot_all[i]*prel.en_converter/(timeRDiss_all[i]*t_fall_cgs) # [erg/s]
    Lc[i], tLc[i] = sort_list([Lc[i], tLc[i]], tLc[i])
    if i == 2:
        ax[i].plot(timeRDiss_all[i], LDiss_all[i]*prel.en_converter/prel.tsol_cgs, color = colorslegend[i], label = r'E$_{\rm diss}$')
        ax[i].plot(tLc[i], Lc[i], label = 'Light curve', color = colorslegend[i], linestyle = '--') 
        ax[i].plot(timeRDiss_all[i], Lumrad_i, color = colorslegend[i], linestyle = '-.', label = r'$\Sigma (u_{rad}V)/t$')
    else:
        ax[i].plot(timeRDiss_all[i], LDiss_all[i]*prel.en_converter/prel.tsol_cgs, color = colorslegend[i])
        ax[i].plot(tLc[i], Lc[i], color = colorslegend[i], linestyle = '--') 
        ax[i].plot(timeRDiss_all[i], Lumrad_i, color = colorslegend[i], linestyle = '-.')
ax[2].set_xlabel(r't [$t_{\rm fb}$]', fontsize = 20)
for i in range(3):
    ax[i].set_ylabel(r'Luminosity [erg/s]', fontsize = 20)
    ax[i].set_yscale('log')
    ax[i].set_title(checkslegend[i], fontsize = 20)
    ax[i].set_xlim(0, 1.8)
plt.legend(fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/LCDiss.png')

# %% Investigate what happens in the drop of Low res (snap 208)
from Utilities.sections import make_slices
from Utilities.time_extractor import days_since_distruption
snapsLow = [207, 208, 209]
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes'

fig, ax = plt.subplots(len(snapsLow), 2, figsize=(30, 15))
for i, snap in enumerate(snapsLow):
    path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
    time = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    X = np.load(f'{path}/CMx_{snap}.npy')
    Y = np.load(f'{path}/CMy_{snap}.npy')
    Z = np.load(f'{path}/CMz_{snap}.npy')
    Vol = np.load(f'{path}/Vol_{snap}.npy')
    Den = np.load(f'{path}/Den_{snap}.npy')
    Diss_den = np.load(f'{path}/Diss_{snap}.npy')
    Diss = np.abs(Diss_den) * Vol
    # make cut in density
    cutden = Den > 1e-19
    x_cut, y_cut, z_cut, den_cut, vol_cut, diss_cut = \
        make_slices([X, Y, Z, Den, Vol,  Diss], cutden)
    mid = np.abs(z_cut) < vol_cut**(1/3)
    x_mid, y_mid, z_mid, den_mid, vol_mid, diss_mid = \
        make_slices([x_cut, y_cut, z_cut, den_cut, vol_cut, diss_cut], mid)
    
    img = ax[i][0].scatter(x_mid/apo, y_mid/apo, c = den_mid, cmap = 'viridis', s = 2,
                        norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-3))
    cb = plt.colorbar(img)
    cb.set_label(r'Density$ [M_\odot/R_\odot^3]$', fontsize = 20)

    img = ax[i][1].scatter(x_mid/apo, y_mid/apo, c = diss_mid*prel.en_converter/prel.tsol_cgs, cmap = 'cet_fire', s = 2,
                        norm = colors.LogNorm(vmin = 1e36, vmax = 1e38))
    cb = plt.colorbar(img)
    cb.set_label(r'E$_{diss}$ [erg/s]', fontsize = 20)
    ax[i][0].text(-1.15, -0.4, f't = {np.round(time,3)}' + r't$_{fb}$', fontsize = 25)
    ax[i][0].set_ylabel('Y [R$_a$]')
    for j in range(2):
        ax[i][j].set_xlim(-1.2, 0.2)
        ax[i][j].set_ylim(-0.5, 0.5)
        ax[2][j].set_xlabel('X [$R_a$]') 
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/{folder}/checkDiss.png')

# %%
