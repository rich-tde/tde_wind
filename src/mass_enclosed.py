""" If alice: Find the mass enclosed in a sphere of radius R0, Rt, a_mb, apo for all the snapshots.ù
If local: check energy dissipation."""
import sys
sys.path.append('/Users/paolamartire/shocks/')

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
from Utilities.operators import make_tree, calc_deriv
from Utilities.selectors_for_snap import select_snap
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
check = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

##
# MAIN
##
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rcheck = np.array([R0, Rt, a_mb, apo])
tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

    Mass_encl = np.zeros((len(snaps), len(Rcheck)))
    Diss_encl = np.zeros((len(snaps), len(Rcheck)))
    Mass_encl_cut = np.zeros((len(snaps), len(Rcheck)))
    for i,snap in enumerate(snaps):
        print(snap)
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        mass, den, vol, diss = data.Mass, data.Den, data.Vol, data.Diss
        Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
        cut = den > 1e-19
        mass_cut, Rsph_cut = mass[cut], Rsph[cut]
        for j,R in enumerate(Rcheck):
            Mass_encl[i,j] = np.sum(mass[Rsph < R])
            Diss_encl[i,j] = np.sum(diss[Rsph < R] * vol[Rsph < R])
            Mass_encl_cut[i,j] = np.sum(mass_cut[Rsph_cut < R])

    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl.txt', Mass_encl)
    np.savetxt(f'{abspath}/data/{folder}/{check}Diss_encl.txt', Diss_encl)
    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl_cut.txt', Mass_encl_cut)
    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl_time.txt', tfb)

else:
    checks = ['LowRes', '', 'HiRes']
    checklab = ['Low', 'Fid', 'High']
    colorcheck = ['C1', 'yellowgreen', 'darkviolet']

    ## Plot as Extended figure 3 in SteinbergStone24
    fig, ax = plt.subplots(1,3, figsize = (18, 5))
    for i, check in enumerate(checks):
        tfb = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_time.txt')
        tfb_cgs = tfb * tfallback_cgs
        Mass_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl.txt')
        Mass_encl = np.transpose(Mass_encl)
        # Mass_encl_cut = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_cut.txt')
        # Mass_encl_cut = np.transpose(Mass_encl_cut)
        ax[i].plot(tfb, Mass_encl[0]/mstar, c = 'deepskyblue', linewidth = 2, label = r'$R = R_0$')
        ax[i].plot(tfb, Mass_encl[1]/mstar, c = 'coral', linewidth = 2, label = r'$R = R_{\rm t}$')
        ax[i].plot(tfb, Mass_encl[2]/mstar, c = 'mediumseagreen', linewidth = 2, label = r'$R = a_{\rm mb}$')
        ax[i].plot(tfb, Mass_encl[3]/mstar, c = 'm', linewidth = 2, label = r'$R = R_{\rm a}$')
        ax[i].set_xlabel(r'$\rm t [t_{fb}]$')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e-7, 2)
        ax[i].grid()
        ax[i].set_title(f'{checklab[i]}', fontsize = 20)
    ax[0].legend(fontsize = 15)
    ax[0].set_ylabel(r'Mass enclosed $[M_\star]$', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/multiple/mass_encl_all.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(1,3, figsize = (20, 5))
    for i, check in enumerate(checks):
        tfb = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_time.txt')
        tfb_cgs = tfb * tfallback_cgs
        Mass_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl.txt')
        Mass_encl = np.transpose(Mass_encl)
        Mdot0 = calc_deriv(tfb_cgs, Mass_encl[0]) * prel.Msol_cgs 
        Lacc0 = 0.05 * Mdot0 * prel.c_cgs**2
        Diss_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Diss_encl.txt')
        Diss_encl = (np.transpose(Diss_encl)) * prel.en_converter/prel.tsol_cgs
        # Mass_encl_cut = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_cut.txt')
        # Mass_encl_cut = np.transpose(Mass_encl_cut)
        ax[i].scatter(tfb, Lacc0, c = 'deepskyblue', s = 5, label = r'$\eta \dot{M}_{\rm encl}c^2$')
        ax[i].scatter(tfb, Diss_encl[0], c = 'coral', s = 5, label = r'Diss')
        ax[i].set_xlabel(r'$\rm t [t_{fb}]$')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e37, 1e44)
        ax[i].set_title(r'Inside $R_0$, ' + f'res: {checklab[i]}', fontsize = 20)
        ax[i].grid()

    ax[0].set_ylabel(r'L$_{\rm acc}$ [erg/s]')#, fontsize = 25)
    ax[0].legend(fontsize = 15)
    plt.savefig(f'{abspath}/Figs/multiple/Maccr_encl.png', bbox_inches='tight')

    #%% Resolutions test for R0
    plt.figure()
    for i, check in enumerate(checks):
        tfb = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_time.txt')
        tfb_cgs = tfb * tfallback_cgs
        Mass_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl.txt')
        Mass_encl = np.transpose(Mass_encl)
        Mass_encl_diff = np.diff(Mass_encl, axis = 1)
        Mdot = (Mass_encl_diff * prel.Msol_cgs) / np.diff(tfb_cgs)
        Lacc = 0.05 * Mdot * prel.c_cgs**2
        # Mass_encl_cut = np.loadtxt(f'{abspath}/data/{folder}/{check}Mass_encl_cut.txt')
        # Mass_encl_cut = np.transpose(Mass_encl_cut)
        plt.plot(tfb, Mass_encl[0]/mstar, c = colorcheck[i], linewidth = 2, label = f'{checklab[i]}')
    
    plt.ylabel(r'Mass enclosed $[M_\star]$ inside $R_0$')#, fontsize = 25)
    plt.xlabel(r'$\rm t [t_{fb}]$')
    plt.yscale('log')
    plt.ylim(1e-9, 1e-4)
    plt.legend(fontsize = 15)
    plt.grid()
    plt.savefig(f'{abspath}/Figs/multiple/Mass_encl.png', bbox_inches='tight')

# %%
