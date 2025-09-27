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
params = [Mbh, Rstar, mstar, beta]

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
t_fb_days_cgs = things['t_fb_days'] * 24 * 3600 # in seconds
norm_dMdE = Mbh/Rt * (Mbh/Rstar)**(-1/3) # Normalisation (what on the x axis you call \Delta E). It's GM/Rt^2 * Rstar
v_esc = np.sqrt(2*prel.G*Mbh/Rp)

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
time_array_yr = np.linspace(1e-1,2, 100) # yr
time_yr_cgs = time_array_yr * 365 * 24 * 3600 # converted to seconds

checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR' ]
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
    print(check)
    # Load data
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    datadays = np.loadtxt(f'{abspath}data/{folder}/wind/dMdE_{check}_days.txt')
    snaps, tfb = datadays[0], datadays[1]
    tfb_cgs = tfb * t_fb_days_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}data/{folder}/wind/dMdE_{check}_bins.txt')
    mid_points = (bins[:-1]+bins[1:])* norm_dMdE/2  # get rid of the normalization
    # bins_cgs = bins * (prel.en_converter/prel.Msol_cgs) #  and convert to CGS (they are bins in SPECIFIC orbital energy)
    dMdE_distr = np.loadtxt(f'{abspath}data/{folder}/wind/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
    dataLum = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snapsLum, tfbLum, Lum = dataLum[:, 0], dataLum[:, 1], dataLum[:, 2] 
    tfbLum, Lum, snapsLum = sort_list([tfbLum, Lum, snapsLum], snapsLum) # becuase Lum data are not ordered
    dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
    timeRDiss, RDiss = dataDiss[:,1], dataDiss[:,2] 
    # if check == 'LowRes':
    #     timeRDiss = np.concatenate([timeRDiss[:127], timeRDiss[129:]])
    #     RDiss = np.concatenate([RDiss[:127], RDiss[129:]])

    timeRDiss_all.append(timeRDiss)
    RDiss_all.append(RDiss)
  
    mfall = np.zeros(len(tfb_cgs))
    eta_sh = np.zeros(len(tfb_cgs))
    R_sh = np.zeros(len(tfb_cgs))
    eta_sh_diss = np.zeros(len(tfb_cgs))
    R_shDiss = np.zeros(len(tfb_cgs))
    if check == 'HiResNewAMR':
        R_ph = np.zeros(len(tfb_cgs))
        R_tr = np.zeros(len(tfb_cgs))

    # compute Rshock and eta_shock in the simulation time range
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
        Lum_t = Lum[idx]
        eta_sh[i] = efficiency_shock(Lum_t, mdot_cgs, prel.c_cgs) # [CGS]
        R_sh[i] = R_shock(Mbh_cgs, eta_sh[i], prel.G_cgs, prel.c_cgs) # [CGS]

        if check == 'HiResNewAMR': # compute the other special radii
            snapR = int(snaps[i])
            photo = \
                np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snapR}.txt')
            xph, yph, zph = photo[0], photo[1], photo[2]
            r_ph_all = np.sqrt(xph**2 + yph**2 + zph**2)
            R_ph[i] = np.median(r_ph_all)

            data_tr= \
                np.load(f'{abspath}/data/{folder}/trap/{check}_Rtr{snapR}.npz')
            x_tr, y_tr, z_tr = data_tr['x_tr'], data_tr['y_tr'], data_tr['z_tr']

            r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
            R_tr[i] = np.median(r_tr_all[r_tr_all!=0])
 
    nan = np.isnan(R_sh)
    R_sh = R_sh[~nan]
    tfb = tfb[~nan]
    tfb_all.append(tfb)
    mfall_all.append(mfall)
    eta_shL_all.append(eta_sh)    
    R_sh_all.append(R_sh/(prel.Rsol_cgs))

Rlim_min = 7e11/(Rp*prel.Rsol_cgs)
Rlim_max = 1e16/(Rp*prel.Rsol_cgs) 
#
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
for i, check in enumerate(checks):
    ax1.plot(tfb_all[i], R_sh_all[i]/Rp, color = colorslegend[i], label = r'$R_{\rm sh}$' if i==2 else None)
    ax1.plot(timeRDiss_all[i], RDiss_all[i]/Rp, linestyle = '--', color = colorslegend[i], label = r'$R_{\rm diss}$' if i==2 else None)

ax1.axhline(y=amin/Rp, color = 'k', linestyle = 'dotted')
ax1.text(1.85, .5* amin/Rp, r'$a_{\rm mb}$', fontsize = 20, color = 'k')
ax1.set_ylabel(r'$R [R_{\rm p}$]')#, fontsize = 18)
ax1.set_ylim(Rlim_min, Rlim_max)
# Set primary y-axis ticks
R_ticks = np.logspace(np.log10(Rlim_min), np.log10(Rlim_max), num=5)
ax1.set_yticks(R_ticks)
ax1.set_yticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in R_ticks])

ax2 = ax1.twinx()
eta_ticks = eta_from_R(Mbh_cgs, R_ticks[::-1]*Rp*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_min = eta_from_R(Mbh_cgs, Rlim_max*Rp*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_max = eta_from_R(Mbh_cgs, Rlim_min*Rp*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
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
        # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        # ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', width=0.7, length=7)
        ax.tick_params(axis='x', which='minor', width=0.5, length=5)
    ax.set_xlim(0, 2)
ax1.legend(fontsize = 16, loc = 'upper right')

plt.tight_layout()
# plt.savefig(f'{abspath}/Figs/paper/Reta{cut}conv.pdf', bbox_inches = 'tight')


# %% compare with other speecial radii
_, tfb_fall, mfall, \
_, _, _, _, \
_, _, _, _, \
_, _, _, _, \
_, _, _, _, = \
    np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiResNewAMR/wind/Mdot_HiResNewAMR.csv', 
                delimiter = ',', 
                skiprows=1, 
                unpack=True)
Rtr_th = Rtr_out(params, mfall, fout=0.1, fv=1)
Redge = v_esc / prel.tsol_cgs * tfb_fall*t_fb_days_cgs

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.plot(tfb_all[2], R_sh_all[2]/Rp, color = 'navy', label = r'$R_{\rm sh}$')
ax1.plot(timeRDiss_all[2], RDiss_all[2]/Rp, color = 'deepskyblue', label = r'$R_{\rm diss}$')
ax1.plot(tfb_all[2], R_ph/Rp, color = colorslegend[2], label = r'$R_{\rm ph}$')
ax1.plot(tfb_all[2], R_tr/Rp, color = 'orchid', label = r'$R_{\rm tr}$')
# ax1.plot(tfb_fall, Redge/Rp, color = 'darkviolet', ls = '--', label = r'$v_{\rm esc}(R_p)$t')
# ax1.plot(tfb_fall, Rtr_th/Rp, color = 'darkviolet', ls = '--', label = r'$R_{\rm tr}$ theory')

ax1.axhline(y=amin/Rp, color = 'k', linestyle = 'dotted')
ax1.text(1.65, 1.2* amin/Rp, r'$a_{\rm mb}$', fontsize = 20, color = 'k')
# ax1.axhline(y=Rp/Rp, color = 'k', linestyle = '--')
# ax1.text(1.85, .5* Rp/Rp, r'$R_{\rm a}$', fontsize = 20, color = 'k')
# ax1.axhline(y=Rp/Rp, color = 'r', linestyle = 'dotted')
# ax0.set_ylabel(r'$|\dot{M}_{\rm fb}| [M_\odot/t_{\rm fb}$]', fontsize = 18)
ax1.set_ylabel(r'$R [R_{\rm p}$]')#, fontsize = 18)
ax1.set_ylim(Rlim_min, Rlim_max)
# Set primary y-axis ticks
R_ticks = np.logspace(np.log10(Rlim_min), np.log10(Rlim_max), num=5)
ax1.set_yticks(R_ticks)
ax1.set_yticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in R_ticks])

ax2 = ax1.twinx()
eta_ticks = eta_from_R(Mbh_cgs, R_ticks[::-1]*Rp*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_min = eta_from_R(Mbh_cgs, Rlim_max*Rp*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
etalim_max = eta_from_R(Mbh_cgs, Rlim_min*Rp*prel.Rsol_cgs, prel.G_cgs, prel.c_cgs)
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
        # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        # ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width=1.1, length=9)
    ax.tick_params(axis='both', which='minor', width=1, length=6)
    ax.set_xlim(0.01, 1.8)
ax1.legend(fontsize = 15, loc = 'upper right')

plt.tight_layout()
plt.savefig(f'{abspath}/Figs/paper/Reta.pdf', bbox_inches = 'tight')

# %%
plt.figure(figsize=(9, 6))
plt.plot(tfb_all[2], eta_shL_all[2], color = colorslegend[2], label = r'$\eta_{\rm sh}$')
plt.yticks(eta_ticks)
plt.ylim(etalim_max, etalim_min)
plt.ylabel(r'$\eta_{\rm sh}$')#, fontsize = 18)
plt.yscale('log')
plt.xlim(0, 2)
plt.grid()
# %%
