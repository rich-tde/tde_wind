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
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import find_ratio

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
Rcheck = np.array([R0, Rt, a_mb])
Rchecklab = [r'$R_0$', r'$R_t$', r'$a_{\rm mb}$']
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 

if alice:
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
    to_plot = 'single_res' # 'compare_res', 'single_res'

    if to_plot == 'compare_res':
        checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']
        checklab = ['Low', 'Middle', 'High']
        colorcheck = ['C1', 'yellowgreen', 'darkviolet']
        commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

        # Plot mass and dissipation rate enclosed in a sphere of radius Rcheck: 
        fig1, (axM1, axDiss1) = plt.subplots(1,2, figsize = (18,6)) 
        fig2, (axM2, axDiss2) = plt.subplots(1,2, figsize = (18,6)) 
        fig3, (axM3, axDiss3) = plt.subplots(1,2, figsize = (18,6)) 
        for i, check in enumerate(checks):
            # load and covert
            data = np.loadtxt(f'{abspath}/data/{commonfolder}{check}/{check}Mass_encl.csv', comments="#", delimiter=',', skiprows = 1)
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

        original_ticks = axM1.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        for ax in [axM1, axM2, axM3,axDiss1, axDiss2, axDiss3]:
            ax.set_xticks(new_ticks)
            ax.tick_params(axis='both', which='major', length=7, width=1)
            ax.tick_params(axis='both', which='minor', length=4, width=1)
            ax.set_xlim(0.01, 2.5)
            ax.set_xlabel(r't [t$_{\rm fb}$]')  
            ax.grid()
            if ax in [axM1, axM2, axM3]:
                ax.set_ylim(1e-6, 1e-3)
                ax.set_ylabel(r'Mass enclosed $[M_\star]$')
                ax.legend(fontsize = 16)
            if ax in [axDiss1, axDiss2, axDiss3]:
                ax.set_ylim(1e38, 2e43)
                ax.set_ylabel(r'Dissipation rate enclosed [erg/s]')
            ax.set_yscale('log')

        fig1.suptitle(Rchecklab[0], fontsize = 20)
        fig2.suptitle(Rchecklab[1], fontsize = 20)
        fig3.suptitle(Rchecklab[2], fontsize = 20)
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig1.savefig(f'{abspath}/Figs/paper/Maccr_encl.pdf', bbox_inches='tight')

    if to_plot == 'single_res':
        check = 'HiResNewAMR'
        Rcheck = np.array([R0, Rt, a_mb])
        labelcheck = [r'$r<r_0$', r'$r<r_{\rm t}$', r'$r<a_{\rm mb}$']
        colorcheck = ['magenta', 'darkviolet', 'k']
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

        # Plot mass and dissipation rate enclosed in a sphere of radius Rcheck: 
        # load and covert
        dataencl = np.loadtxt(f'{abspath}/data/{folder}/{check}Mass_encl.csv', delimiter=',', skiprows = 1)
        tfb_encl = dataencl[:, 1]
        Mass_encl = dataencl[:, 2:5]
        Diss_pos_encl = dataencl[:, 5:8] * prel.en_converter/prel.tsol_cgs
        dMdt0 = np.diff(Mass_encl[:,0])/np.diff(tfb_encl)
        dMdt0_cgs = dMdt0 * prel.Msol_cgs/t_fb_days_cgs
        Lacc = 0.05 * dMdt0_cgs * prel.c_cgs**2
        print(f'L = {np.max(Lacc)} at t = {tfb_encl[np.argmax(Lacc)]}')
        # print(0.05 * 1e-4*mstar*prel.Msol_cgs/t_fb_days_cgs * prel.c_cgs**2)
        plt.figure(figsize = (10,6))
        plt.plot(tfb_encl[:-1], dMdt0/Medd_sol, c = 'k')
        plt.yscale('log') 
        plt.xlabel(r't [t$_{\rm fb}$]')
        plt.ylabel(r'dM/dt $(r\leq r_0)[M_{\rm Edd}$]')
        plt.grid()
        plt.savefig(f'{abspath}/Figs/paper/dMdt0.pdf', bbox_inches='tight')
        dataall = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows = 1)
        tfb_all, Ldisstot_pos, Ldisstot_neg = dataall[:,1], dataall[:,3], dataall[:,5]
        Ldisstot_pos *= prel.en_converter/prel.tsol_cgs
        Ldisstot_neg *= prel.en_converter/prel.tsol_cgs
    
        fig , (axM, axDiss) = plt.subplots(1, 2, figsize = (21,7)) 
        for j in range(3):
            axM.plot(tfb_encl, Mass_encl[:,j]/mstar, c = colorcheck[j], label = labelcheck[j])
            axDiss.plot(tfb_encl, Diss_pos_encl[:,j], c = colorcheck[j])#, label = labelcheck[j])

        axDiss.plot(tfb_all, Ldisstot_pos, c = 'gray', ls = '--', label = 'Total')
        axM.set_ylabel(r'Mass enclosed $[M_\star]$')
        axM.set_ylim(1e-6, 1e-1)
        axDiss.set_ylabel(r'Dissipation rate enclosed [erg/s]')
        axDiss.set_ylim(1e38, 2e43)
        original_ticks = axM.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [f'{t:.2f}' if t in original_ticks else '' for t in new_ticks]
        for ax in [axM, axDiss]:
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels)
            ax.tick_params(axis='both', which='major', length=7, width=1)
            ax.tick_params(axis='both', which='minor', length=4, width=1)
            ax.set_xlim(0.01, np.max(tfb_encl))
            ax.grid()
            ax.set_yscale('log')
            ax.set_xlabel(r't [t$_{\rm fb}$]')  
            ax.legend(fontsize = 18)
        
        fig.savefig(f'{abspath}/Figs/paper/ME_encl.pdf', bbox_inches='tight')
