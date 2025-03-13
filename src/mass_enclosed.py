""" If alice: Compute and save the mass enclosed and the total diss rate in a sphere of radius R0, Rt, a_mb, apo for all the snapshots.
If local: plots"""
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
from Utilities.sections import make_slices
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

    with open(f'{abspath}/data/{folder}/{check}Mass_encl.txt','w') as file:
        file.write('# just cells with den > 1e-19. Different lines are for quantities enclosed in R0, Rt, a_mb, apo\n')
        file.close()
    with open(f'{abspath}/data/{folder}/{check}Diss_pos_encl.txt','w') as file:
        file.write('# just cells with den > 1e-19. Different lines are for quantities enclosed in R0, Rt, a_mb, apo\n')
        file.close()
    with open(f'{abspath}/data/{folder}/{check}Diss_neg_encl.txt','w') as file:
        file.write('# just cells with den > 1e-19. Different lines are for quantities enclosed in R0, Rt, a_mb, apo\n')
        file.close()
    Mass_encl = np.zeros((len(snaps), len(Rcheck)))
    Diss_pos_encl = np.zeros((len(snaps), len(Rcheck)))
    Diss_neg_encl = np.zeros((len(snaps), len(Rcheck)))
    for i,snap in enumerate(snaps):
        print(snap)
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        mass, den, vol, diss = data.Mass, data.Den, data.Vol, data.Diss
        Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
        cut = den > 1e-19
        mass, Rsph, diss, vol = \
            make_slices([mass, Rsph, diss, vol], cut)
        diss_pos = diss >= 0
        diss_neg = diss < 0
        for j,R in enumerate(Rcheck):
            Mass_encl[i,j] = np.sum(mass[Rsph < R])
            Diss_pos_encl[i,j] = np.sum(diss[np.logical_and(diss_pos, Rsph < R)] * vol[np.logical_and(diss_pos, Rsph < R)])
            Diss_neg_encl[i,j] = np.sum(diss[np.logical_and(diss_neg, Rsph < R)] * vol[np.logical_and(diss_neg, Rsph < R)])
    
    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl_time.txt', tfb)
    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl.txt', Mass_encl)
    np.savetxt(f'{abspath}/data/{folder}/{check}Diss_pos_encl.txt', Diss_pos_encl)
    np.savetxt(f'{abspath}/data/{folder}/{check}Diss_neg_encl.txt', Diss_neg_encl)

else:
    checks = ['LowRes', '', 'HiRes']
    checklab = ['Low', 'Fid', 'High']
    colorcheck = ['C1', 'yellowgreen', 'darkviolet']

    ## Plot as Extended figure 3 in SteinbergStone24
    fig, ax = plt.subplots(1,3, figsize = (18, 5))
    for i, check in enumerate(checks):
        tfb = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_time.txt')
        tfb_cgs = tfb * tfallback_cgs
        Mass_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl.txt', comments="#")
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
    
    fig, ax = plt.subplots(1,3, figsize = (22, 5))
    for i, check in enumerate(checks):
        tfb = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_time.txt')
        tfb_cgs = tfb * tfallback_cgs
        Mass_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl.txt')
        Mass_encl = np.transpose(Mass_encl)
        Mdot0 = calc_deriv(tfb_cgs, Mass_encl[0]) * prel.Msol_cgs 
        Lacc0 = 0.05 * Mdot0 * prel.c_cgs**2
        Diss_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Diss_pos_encl.txt')
        Diss_encl = (np.transpose(Diss_encl)) * prel.en_converter/prel.tsol_cgs
        Diss_encl0 = Diss_encl[0]
        nan = np.isnan(Diss_encl0)
        Diss_encl0 = Diss_encl0[~nan]
        tfbDis = tfb[~nan]
        # exclude the 127 because old data for LowRes
        if check == 'LowRes':
            Diss_encl0 = np.delete(Diss_encl0, 127)
            tfbDis = np.delete(tfbDis, 127)
        # find where Lacc0 give nan and remove it from Lacc0 and tfb
        nan = np.isnan(Lacc0)
        Lacc0 = Lacc0[~nan]
        tfbL = tfb[~nan]
        ax[i].plot(tfbDis, np.abs(Diss_encl0), c = 'cornflowerblue', label = r'L$_{\rm diss}$')
        ax[i].plot(tfbL, np.abs(Lacc0), c = 'chocolate', label = r'L$_{\rm acc}$')
        ax[i].set_xlabel(r'$\rm t [t_{fb}]$')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e37, 1e44)
        ax[i].text(np.max(tfbL)-0.4, 2e37, f'{checklab[i]} res', fontsize = 25)
        ax[i].grid()
        original_ticks = ax[i].get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        ax[i].set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax[i].set_xticklabels(labels)
        ax[i].set_xlim(0, np.max(tfbL))


    ax[0].set_ylabel(r'Luminosity [erg/s]')#, fontsize = 25)
    ax[0].legend(fontsize = 18)
    plt.savefig(f'{abspath}/Figs/paper/Maccr_encl.pdf', bbox_inches='tight')

    #%% Resolutions test for R0
    fig, ax = plt.subplots(1,1, figsize= (10,7))
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
        ax.plot(tfb, Mass_encl[0]/mstar, c = colorcheck[i], linewidth = 2, label = f'{checklab[i]}')
    
    original_ticks = ax.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    ax.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'Mass enclosed $[M_\star]$ inside $R_0$')#, fontsize = 25)
    ax.set_xlabel(r'$\rm t [t_{fb}]$')
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 1e-4)
    ax.set_xlim(0, 1.75)
    ax.legend(fontsize = 20)
    ax.grid()
    plt.savefig(f'{abspath}/Figs/paper/Mass_encl.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,3, figsize = (22, 5))
    for i, check in enumerate(checks):
        tfb_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Mass_encl_time.txt')
        dataDiss_tot = np.loadtxt(f'{abspath}/data/{folder}{check}/Rdiss_{check}cutDen.txt')
        tdiss_tot, Ldiss_tot = dataDiss_tot[0], dataDiss_tot[2]
        Ldiss_tot *= prel.en_converter/prel.tsol_cgs
        Diss_encl = np.loadtxt(f'{abspath}/data/{folder}{check}/{check}Diss_pos_encl.txt')
        Diss_encl = (np.transpose(Diss_encl)) * prel.en_converter/prel.tsol_cgs
        Diss_encl0 = Diss_encl[0]
        nan = np.isnan(Diss_encl0)
        Diss_encl0 = Diss_encl0[~nan]
        tfb_encl = tfb_encl[~nan]
        # exclude the 127 because old data for LowRes
        if check == 'LowRes':
            Diss_encl0 = np.delete(Diss_encl0, 127)
            tfb_encl = np.delete(tfb_encl, 127)
            Ldiss_tot = np.delete(Ldiss_tot, 127)
            tdiss_tot = np.delete(tdiss_tot, 127)
        ax[i].plot(tfb_encl, Diss_encl0, c = 'cornflowerblue', label = r'L$_{\rm diss} (R<R_0)$')
        ax[i].plot(tdiss_tot, Ldiss_tot, c = 'firebrick', label = r'total L$_{\rm diss}$')
        ax[i].set_xlabel(r'$\rm t [t_{fb}]$')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e37, 1e44)
        ax[i].grid()
        original_ticks = ax[i].get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        ax[i].set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax[i].set_xticklabels(labels)
        ax[i].set_xlim(0,np.max(tdiss_tot))
        ax[i].text(0.1, 1e43, f'{checklab[i]} res', fontsize = 25)
    ax[0].set_ylabel(r'Luminosity [erg/s]')
    ax[0].legend(fontsize = 16, loc = 'lower right')
    plt.savefig(f'{abspath}/Figs/paper/Ldiss_totVSencl.pdf', bbox_inches='tight')

# %%
