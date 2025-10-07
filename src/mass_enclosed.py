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
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
import numpy as np
import os
import csv
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

##
# MAIN
##
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
a_mb = things['a_mb']
t_fb_days = things['t_fb_days']
t_fb_days_cgs = t_fb_days * 24 * 3600 

if alice:
    Rcheck = np.array([R0, Rt, a_mb])
    check = 'NewAMR'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
    csv_path = f'{abspath}/data/{folder}/{check}Mass_encl.csv'

    for i,snap in enumerate(snaps):
        print(snap,  flush = True)
        # Load data
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        mass, den, vol, diss_den = data.Mass, data.Den, data.Vol, data.Diss
        diss = diss_den * vol
        Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
        cut = den > 1e-19
        mass, Rsph, diss = \
            make_slices([mass, Rsph, diss], cut)
        diss_pos = diss >= 0
        diss_neg = diss < 0
        Mass_encl = np.zeros(len(Rcheck))
        Diss_pos_encl = np.zeros(len(Rcheck))
        Diss_neg_encl = np.zeros(len(Rcheck))
        for j, R in enumerate(Rcheck):
            enclosed = Rsph < R
            Mass_encl[j] = np.sum(mass[enclosed])
            Diss_pos_encl[j] = np.sum(diss[np.logical_and(diss_pos, enclosed)])
            Diss_neg_encl[j] = np.sum(diss[np.logical_and(diss_neg, enclosed)])
        
        data = np.concatenate([[snap, tfb], Mass_encl, Diss_pos_encl, Diss_neg_encl])
        # save in code units
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                header = ['snap', ' tfb']
                header += [f' Mass at {r} ' for r in Rcheck]
                header += [f' Diss_pos at {r} ' for r in Rcheck]
                header += [f' Diss_neg at {r} ' for r in Rcheck]
                writer.writerow(header)
            writer.writerow(data)
    file.close() 

else:
    checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']
    checklab = ['Low', 'Middle', 'High']
    colorcheck = ['C1', 'yellowgreen', 'darkviolet']
    commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

    # Plot mass and dissipation rate enclosed in a sphere of radius Rcheck: 
    fig1, ((axM1, axMerr1), (axDiss1, axDisserr1)) = plt.subplots(2,2, figsize = (16,14)) 
    fig2, ((axM2, axMerr2), (axDiss2, axDisserr2)) = plt.subplots(2,2, figsize = (16,14)) 
    fig3, ((axM3, axMerr3), (axDiss3, axDisserr3)) = plt.subplots(2,2, figsize = (16,14)) 
    for i, check in enumerate(checks):
        # load and covert
        data = np.loadtxt(f'{abspath}/data/{folder}/{check}Mass_encl.csv', comments="#")
        tfb = data[:, 1]
        Mass_enclR0 = data[:, 2]
        Mass_enclRt = data[:, 3]
        Mass_encla_mb = data[:, 4]
        Diss_pos_enclR0 = data[:, 5] * prel.en_converter/prel.tsol_cgs
        Diss_pos_enclRt = data[:, 6] * prel.en_converter/prel.tsol_cgs
        Diss_pos_encla_mb = data[:, 7] * prel.en_converter/prel.tsol_cgs
        
        axM1.plot(tfb, Mass_enclR0/mstar, c = colorcheck[i], label = checklab[i])
        axM2.plot(tfb, Mass_enclRt/mstar, c = colorcheck[i], label = checklab[i])
        axM3.plot(tfb, Mass_encla_mb/mstar, c = colorcheck[i], label = checklab[i])
        
        axDiss1.plot(tfb, Diss_pos_enclR0, c = colorcheck[i])
        axDiss2.plot(tfb, Diss_pos_enclRt, c = colorcheck[i])
        axDiss3.plot(tfb, Diss_pos_encla_mb, c = colorcheck[i])

    ax1.set_ylabel(r'Mass enclosed $[M_\star]$')
    axDiss1.set_ylabel(r'Dissipation rate enclosed [erg/s]')
    ax2.legend(fontsize = 20)
    axDiss2.legend(fontsize = 20)
    ax1.set_title(f'Low Res', fontsize = 20)
    ax2.set_title(f'Fid Res', fontsize = 20)
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    # y 
    original_ticks_y = ax1.get_yticks()
    midpoints_y = (original_ticks_y[:-1] + original_ticks_y[1:]) / 2
    new_ticks_y = np.sort(np.concatenate((original_ticks_y, midpoints_y)))
    for a in [ax1, ax2, axDiss1, axDiss2]:
        a.set_xticks(new_ticks)
        a.tick_params(axis='both', which='major', length=7, width=1)
        a.tick_params(axis='both', which='minor', length=4, width=1)
        a.grid()
        # a.axvline(tfb[136], c = 'k', ls = '--', linewidth = 2, label = r'$t_{\rm fb}$')
        # print(tfb[136], tfb[137])
        a.set_xlim(0.01, 1.8)
        if a in [ax1, ax2]:
            a.set_yticks(new_ticks_y)
            a.set_ylim(1e-9, 9e-2)
        else:
            a.set_ylim(2e37, 1e43)
        a.set_yscale('log')

    fig.suptitle(f'New AMR runs', fontsize = 25)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/Test/MazeOfRuns/sink/massDiss_encl.png', bbox_inches='tight')
