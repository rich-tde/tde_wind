""" Compute Rdissipation. Written to be run on alice."""
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
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree
from Utilities.sections import make_slices
from Utilities.selectors_for_snap import select_snap
import Utilities.sections as sec
import csv
import os

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
compton = 'Compton'
check = 'HiResNewAMR'
do_cut = 'ionizationHE' # '' or 'ionization' or 'ionizationHE'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
print(f'we are in {check}', flush=True)
things = orb.get_things_about(params)
Rt = things['Rt']
apo = things['apo']

if alice:
    snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 

    for i, snap in enumerate(snaps):
        print(snap, flush=False) 
        sys.stdout.flush()
        
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        csv_path = f'{abspath}/data/{folder}/Rdiss_{check}{do_cut}.csv'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        X, Y, Z, vol, den, Temp, Rad_den, Ediss_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Temp, data.Rad, data.Diss
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)

        # cut fluff
        cut = den > 1e-19
        Rsph, vol, Temp, Rad_den, Ediss_den = \
            make_slices([Rsph, vol, Temp, Rad_den, Ediss_den], cut)
        Ediss = Ediss_den * vol # energy dissipation rate [energy/time] in code units

        if do_cut == '':
            Ldisstot_pos = np.sum(Ediss[Ediss_den >= 0])
            Rdiss_pos = np.sum(Rsph[Ediss_den >= 0] * Ediss[Ediss_den >= 0]) / np.sum(Ediss[Ediss_den >= 0])
            Ldisstot_neg = np.sum(Ediss[Ediss_den < 0])
            Rdiss_neg = np.sum(Rsph[Ediss_den < 0] * Ediss[Ediss_den < 0]) / np.sum(Ediss[Ediss_den < 0])
            data = [snap, tfb, Rdiss_pos, Ldisstot_pos, Rdiss_neg, Ldisstot_neg]
            with open(csv_path,'a', newline='') as file:
                writer = csv.writer(file)           
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    header = ['snap', ' tfb', ' Rdiss_pos', ' Ldisstot_pos', ' Rdiss_neg', ' Ldisstot_neg']
                    writer.writerow(header)
                writer.writerow(data)
            file.close()

        if do_cut == 'ionization' or do_cut == 'ionizationHE': 
            # split above and belowe 5e4K
            if do_cut == 'ionizationHE':
                above = np.logical_and(Temp >= 1e5, Ediss_den >= 0)
                below = np.logical_and(Temp < 1e5, Ediss_den >= 0)
            if do_cut == 'ionization':
                above = np.logical_and(Temp >= 5e4, Ediss_den >= 0)
                below = np.logical_and(Temp < 5e4, Ediss_den >= 0)
            Ldiss_above = np.sum(Ediss[above])
            Rdiss_above = np.sum(Rsph[above] * Ediss[above]) / np.sum(Ediss[above])
            Ldiss_below = np.sum(Ediss[below])
            Rdiss_below = np.sum(Rsph[below] * Ediss[below]) / np.sum(Ediss[below])

            data = [snap, tfb, Rdiss_above, Ldiss_above, Rdiss_below, Ldiss_below]
            with open(csv_path,'a', newline='') as file:
                writer = csv.writer(file)           
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    header = ['snap', ' tfb', ' Rdiss (pos) above 5e4K', ' Ldiss pos above', ' Rdiss pos below', ' Ldiss pos below']
                    writer.writerow(header)
                writer.writerow(data)
            file.close()

else:
    from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall
    commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    dataL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/Rdiss_LowResNewAMR.csv', delimiter=',', dtype=float)
    snapL, tfbL, Rdiss_posL, Ldisstot_posL, Rdiss_negL, Ldisstot_negL =  dataL[:, 0], dataL[:, 1], dataL[:, 2], dataL[:, 3], dataL[:, 4], dataL[:, 5]
    data = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/Rdiss_NewAMR.csv', delimiter=',', dtype=float)
    snap, tfb, Rdiss_pos, Ldisstot_pos, Rdiss_neg, Ldisstot_neg = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
    dataH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/Rdiss_HiResNewAMR.csv', delimiter=',', dtype=float)
    snapH, tfbH, Rdiss_posH, Ldisstot_posH, Rdiss_negH, Ldisstot_negH = dataH[:,0], dataH[:,1], dataH[:,2], dataH[:,3], dataH[:,4], dataH[:,5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    ax1.plot(tfbL, Ldisstot_posL * prel.en_converter/prel.tsol_cgs, color = 'C1', label = 'Low')
    ax1.plot(tfb, Ldisstot_pos * prel.en_converter/prel.tsol_cgs, color = 'yellowgreen', label = 'Fid')
    ax1.plot(tfbH, Ldisstot_posH * prel.en_converter/prel.tsol_cgs, color = 'darkviolet', label = 'High')
    # ax1.plot(tfb, np.abs(Ldisstot_neg) * prel.en_converter/prel.tsol_cgs, label = r'$|$L negative, cut den$|$', color = 'grey')
    ax1.set_ylabel('Dissipation rate [erg/s]')
    ax1.legend(fontsize = 16)

    ax2.plot(tfbL, Rdiss_posL/apo, c = 'C1')
    ax2.plot(tfb, Rdiss_pos/apo, c = 'yellowgreen')
    ax2.plot(tfbH, Rdiss_posH/apo, c = 'darkviolet')
    ax2.axhline(Rt/apo, color = 'k', linestyle = '--', label = r'$R_{\rm t}$')


    ax2.set_ylabel(r'$R_{\rm diss}$ [R$_{\rm a}$]')
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', length = 7, width = 1.2)
        ax.tick_params(axis='both', which='minor', length = 3, width = 1)
    ax2.set_xlabel(r't [t$_{\rm fb}$]')

    # with ratios
    tfb_ratioDiss_L, ratio_DissL = ratio_BigOverSmall(tfb, Ldisstot_pos, tfbL, Ldisstot_posL)
    tfb_ratioDiss_H, ratio_DissH = ratio_BigOverSmall(tfb, Ldisstot_pos, tfbH, Ldisstot_posH)
    tfb_ratioRdiss_L, ratio_RdissL = ratio_BigOverSmall(tfb, Rdiss_pos, tfbL, Rdiss_posL)
    tfb_ratioRdiss_H, ratio_RdissH = ratio_BigOverSmall(tfb, Rdiss_pos, tfbH, Rdiss_posH)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 16))
    ax1.plot(tfbL, Ldisstot_posL * prel.en_converter/prel.tsol_cgs, color = 'C1', label = 'Low')
    ax1.plot(tfb, Ldisstot_pos * prel.en_converter/prel.tsol_cgs, color = 'yellowgreen', label = 'Fid')
    ax1.plot(tfbH, Ldisstot_posH * prel.en_converter/prel.tsol_cgs, color = 'darkviolet', label = 'High')
    ax1.set_ylabel('Dissipation rate [erg/s]')
    ax1.legend(fontsize = 16)

    ax3.plot(tfb_ratioDiss_L, ratio_DissL, color = 'C1')
    ax3.plot(tfb_ratioDiss_L, ratio_DissL, '--', color = 'yellowgreen')
    ax3.plot(tfb_ratioDiss_H, ratio_DissH, color = 'yellowgreen')
    ax3.plot(tfb_ratioDiss_H, ratio_DissH, '--', color = 'darkviolet')

    ax2.plot(tfbL, Rdiss_posL/apo, c = 'C1')
    ax2.plot(tfb, Rdiss_pos/apo, c = 'yellowgreen')
    ax2.plot(tfbH, Rdiss_posH/apo, c = 'darkviolet')
    ax2.set_ylabel(r'$R_{\rm diss}$ [R$_{\rm a}$]')
    ax2.axhline(Rt/apo, color = 'k', linestyle = '--', label = r'$R_{\rm t}$')
    ax3.set_ylabel('Ratio of dissipation rate')

    ax4.plot(tfb_ratioRdiss_L, ratio_RdissL, color = 'C1')
    ax4.plot(tfb_ratioRdiss_L, ratio_RdissL, '--', color = 'yellowgreen')
    ax4.plot(tfb_ratioRdiss_H, ratio_RdissH, color = 'yellowgreen')
    ax4.plot(tfb_ratioRdiss_H, ratio_RdissH, '--', color = 'darkviolet')
    ax4.set_ylabel(r'ratio $R_{\rm diss}$')

    for ax in [ax1, ax2, ax3, ax4]:
        if ax != ax4:
            ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', length = 7, width = 1.2)
        ax.tick_params(axis='both', which='minor', length = 3, width = 1)
        ax.set_xlim(0, 1.7)
    ax3.set_xlabel(r't [t$_{\rm fb}$]')
    ax4.set_xlabel(r't [t$_{\rm fb}$]')


