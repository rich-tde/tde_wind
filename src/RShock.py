""" Compute shock efficiency and Rshock from theoretical approximation.
Plot it and compare with Rdissipation."""
import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
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
params = [Mbh, Rstar, mstar, beta]

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
norm_dMdE = things['E_mb']
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

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

def Rtr_out(params, Mdot, fout, fv):
    Mbh, Rstar, mstar, beta = params
    Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
    Medd = Ledd/(0.1*prel.c_cgs**2)
    Medd_code = Medd * prel.tsol_cgs / prel.Msol_cgs  
    things = orb.get_things_about(params)
    Rp = things['Rp']
    Rs = things['Rs']
    r = 4 * fout/fv * (np.abs(Mdot)/Medd_code) * np.sqrt(Rp/(3*Rs)) * Rs
    return r

##
# MAIN
#%%
# time_array_yr = np.linspace(1e-1,2, 100) # yr
# time_yr_cgs = time_array_yr * 365 * 24 * 3600 # converted to seconds

check = 'HiResNewAMR' #['LowResNewAMR', 'NewAMR', 'HiResNewAMR' ]


# Load data
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
# datadays = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}_days.txt')
# snaps, tfb = datadays[0], datadays[1]

# tfb_cgs = tfb * tfallback_cgs #converted to seconds
# bins = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}_bins.txt')
# max_bin_negative = np.abs(np.min(bins))
# mid_points = (bins[:-1]+bins[1:]) * norm_dMdE/2  # get rid of the normalization
# dMdE_distr = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}.txt')[0] # distribution just after the disruption
# bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
snaps, tfb, mfall, _, _, _, _, _, _, _ = \
        np.loadtxt(f'{abspath}/data/{folder}/paper1/wind/Mdot_{check}05aminmean.csv', 
                delimiter = ',', 
                skiprows=1, 
                unpack=True) 
tfb_cgs = tfb * tfallback_cgs #converted to seconds

dataLum = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snapsLum, tfbLum, Lum = dataLum[:, 0], dataLum[:, 1], dataLum[:, 2] 
tfbLum, Lum, snapsLum = sort_list([tfbLum, Lum, snapsLum], snapsLum) # becuase Lum data are not ordered
dataDiss = np.loadtxt(f'{abspath}/data/{folder}/paper1/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
timeRDiss, RDiss, Ldisstot_posH = dataDiss[:,1], dataDiss[:,2], dataDiss[:,3]

eta_sh = np.zeros(len(tfb_cgs))
R_sh = np.zeros(len(tfb_cgs))
eta_sh_diss = np.zeros(len(tfb_cgs))
R_ph = np.zeros(len(tfb_cgs))
R_tr = np.zeros(len(tfb_cgs))
Ldiss_Mdotc2 = np.zeros(len(tfb_cgs))

# compute Rshock and eta_shock in the simulation time range
for i, t in enumerate(tfb_cgs):
    # convert to code units
    # tsol = t / prel.tsol_cgs
    # # Find the energy of the element at time t
    # energy = orb.keplerian_energy(Mbh, prel.G, tsol) # it'll give it positive
    # i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
    # if energy-max_bin_negative*norm_dMdE > 0:
    #     print(f'You overcome the maximum negative bin ({max_bin_negative*norm_dMdE}). You required {energy}')
    #     continue
    # dMdE_t = dMdE_distr_tokeep[i_bin]
    # mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
    # mfall[i] = mdot # code units
    mdot = mfall[i] # code units
    mdot_cgs = mdot * prel.Msol_cgs / prel.tsol_cgs # [g/s]
    # Find the luminosity at time t
    idx = np.argmin(np.abs(tfb[i]-tfbLum)) # just to be sure that you match the data
    Lum_t = Lum[idx]
    eta_sh[i] = efficiency_shock(Lum_t, mdot_cgs, prel.c_cgs) # [CGS]
    R_sh[i] = R_shock(Mbh_cgs, eta_sh[i], prel.G_cgs, prel.c_cgs) # [CGS]

    snapR = int(snaps[i])
    photo = \
        np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snapR}.txt')
    xph, yph, zph, volph, denph = photo[0], photo[1], photo[2], photo[3], photo[4]
    mass_ph = volph * denph
    r_ph_all = np.sqrt(xph**2 + yph**2 + zph**2)
    R_ph[i] = np.median(r_ph_all)

    data_tr= \
        np.load(f'{abspath}/data/{folder}/trap/{check}_Rtr{snapR}.npz')
    x_tr, y_tr, z_tr = data_tr['x_tr'], data_tr['y_tr'], data_tr['z_tr']

    r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    R_tr[i] = np.median(r_tr_all[r_tr_all!=0])


    Ldiss_Mdotc2[i] = Ldisstot_posH[np.argmin(np.abs(tfb[i]-timeRDiss))] / (np.abs(mfall[i]) * prel.csol_cgs**2)

# nan = np.logical_or(np.isnan(R_sh), R_sh==0)
# R_sh = R_sh[~nan]
# eta_sh = eta_sh[~nan]
# R_ph = R_ph[~nan]
# R_tr = R_tr[~nan]
# tfb = tfb[~nan]
# print(tfb)
R_sh = R_sh / prel.Rsol_cgs # in solar radii
Rlim_min = 7e11/(Rt*prel.Rsol_cgs)
Rlim_max = 1e16/(Rt*prel.Rsol_cgs) 

#
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.plot(tfb, R_sh/Rt, color = 'k', linestyle = '--', label = r'$r_{\rm sh}$')
ax1.plot(timeRDiss, RDiss/Rt,  color = 'magenta', label = r'$r_{\rm diss}$')
ax1.plot(tfb, R_ph/Rt, color = 'darkviolet', label = r'$r_{\rm ph}$')

ax1.axhline(y=apo/Rt, color = 'gray', linestyle = '-.')
ax1.text(0.92*np.max(tfb), 1.18* apo/Rt, r'$r_{\rm a}$', fontsize = 20, color = 'k')
ax1.set_ylabel(r'$r [r_{\rm t}$]')#, fontsize = 18)
ax1.set_ylim(Rlim_min, Rlim_max)
# Set primary y-axis ticks
R_ticks = np.logspace(np.log10(Rlim_min), np.log10(Rlim_max), num=5)
ax1.set_yticks(R_ticks)
ax1.set_yticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in R_ticks])

ax2 = ax1.twinx()
eta_ticks = eta_from_R(Mbh_cgs, R_ticks[::-1]*Rt*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_min = eta_from_R(Mbh_cgs, Rlim_max*Rt*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_max = eta_from_R(Mbh_cgs, Rlim_min*Rt*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
ax2.set_yticks(eta_ticks)
ax2.set_ylim(etalim_max, etalim_min)
ax2.set_ylabel(r'$\eta_{\rm num}$')#, fontsize = 18)

original_ticks = ax1.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.set_xlabel(r't [$t_{\rm fb}$]')#, fontsize = 20)
    ax.grid()
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width=1.1, length=9)
    ax.tick_params(axis='both', which='minor', width=1, length=6)
    ax.set_xlim(np.min(tfb), np.max(tfb))
ax1.legend(fontsize = 20, loc = 'upper right')

plt.tight_layout()
plt.savefig(f'{abspath}/Figs/paper/Reta.pdf', bbox_inches = 'tight')

#%% eta
eta_checkRsh = [eta_from_R(Mbh, R_sh[i], prel.G, prel.csol_cgs) for i in range(len(R_sh))] # i.e. you imagine that all the kinetic energy of returning material is converted into energy and released at Rshock
eta_checkRdiss = [eta_from_R(Mbh, RDiss[i], prel.G, prel.csol_cgs) for i in range(len(RDiss))] # same as above but at Rdiss
plt.figure(figsize=(9, 6))
plt.plot(tfb, eta_sh, color = 'dodgerblue', label = r'$\eta_{\rm sh} = L_{\rm FLD}/(|\dot{M}_{\rm fb}|c^2)$') 
plt.plot(tfb, eta_checkRsh, color = 'k', ls = '--', label = r'$\eta_{\rm sh} = GM/(r_{\rm sh}c^2)=r_{\rm g}/(r_{\rm sh})$')
plt.plot(tfb, 0.5*Rs/R_sh, color = 'gold', ls = ':', label = r'$r_{\rm g}/(r_{\rm sh})$')
plt.plot(timeRDiss, eta_checkRdiss, color = 'magenta', label = r'$\eta_{\rm diss}= GM/(r_{\rm diss}c^2)=r_{\rm g}/(r_{\rm diss})$')
plt.yticks(eta_ticks)
plt.ylim(etalim_max, etalim_min)
plt.ylabel(r'$\eta_{\rm sh}$')#, fontsize = 18)
plt.yscale('log')
plt.xlim(np.min(tfb), np.max(tfb))
plt.legend(fontsize = 18)
plt.suptitle(r'$\dot{M}_{\rm fb}$ is computed at $0.5a_{\rm mb}$', fontsize = 24)
plt.grid()

# %%
horiz_line = 0.5 / (600 * 13**2)
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.plot(tfb, Ldiss_Mdotc2, color = 'dodgerblue', label = r'$\eta_{\rm diss} = L_{\rm diss}/(\dot{M}_{\rm fb} c^2)$')
ax.set_ylabel(r'$\eta$', fontsize = 30)
ax.set_xlabel(r't [$t_{\rm fb}$]', fontsize = 30)
ax.set_yscale('log')
ax.axhline(y=horiz_line, color = 'gray', linestyle = '--', label = r'$\eta_{\rm nz} = 0.5(r_\star/r_{\rm p})^2r_{\rm g}/r_{\rm p}$')
ax.set_xlim(0.5, np.max(tfb))
ax.set_ylim(1e-6, 1e-4)
ax.legend(fontsize = 22)
ax.tick_params(axis='both', which='major', width=1.1, length=9)
ax.tick_params(axis='both', which='minor', width=1, length=6)
ax.grid()


# %%
