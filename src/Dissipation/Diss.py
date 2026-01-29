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
from Utilities.operators import make_tree, choose_sections
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
do_cut = 'sections' # '' or 'ionization' or 'ionizationHe' or 'sections'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
print(f'we are in {check}', flush=True)
things = orb.get_things_about(params)
Rt = things['Rt']
apo = things['apo']
t_fb_days = things['t_fb_days']
t_fb_days_cgs = t_fb_days * 24 * 3600 

if alice:
    snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 

    for i, snap in enumerate(snaps):
        # if snap > 30:
        #     continue
        print(snap, flush=True) 
        
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        csv_path = f'{abspath}/data/{folder}/Rdiss_{check}{do_cut}.csv'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        X, Y, Z, vol, den, Temp, Rad_den, Ediss_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Temp, data.Rad, data.Diss
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)

        # cut fluff
        if do_cut == 'nocut':
            cut = den > 0
        else:
            cut = den > 1e-19
        Rsph, vol, Temp, Rad_den, Ediss_den = \
            make_slices([Rsph, vol, Temp, Rad_den, Ediss_den], cut)
        Ediss = Ediss_den * vol # energy dissipation rate [energy/time] in code units
        
        if do_cut == 'sections':
            X, Y, Z = make_slices([X, Y, Z], cut)
            sections = choose_sections(X, Y, Z, choice = 'dark_bright_z')
            cond_sec = []
            label_obs = []
            for key in sections.keys():
                cond_sec.append(sections[key]['cond'])
                label_obs.append(sections[key]['label'])

            Rdiss_list = np.zeros(len(cond_sec))
            Ldiss_list = np.zeros(len(cond_sec))
            for k, cond in enumerate(cond_sec):
                mask = np.logical_and(Ediss_den >= 0, cond)

                Rdiss_pos = np.sum(Rsph[mask] * Ediss[mask]) / np.sum(Ediss[mask])
                Ldisstot_pos = np.sum(Ediss[mask])

                Rdiss_list[k] = Rdiss_pos
                Ldiss_list[k] = Ldisstot_pos
    
            data = np.concatenate(([snap], [tfb], Rdiss_list, Ldiss_list))  
            with open(csv_path,'a', newline='') as file:
                writer = csv.writer(file)           
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    header = ['snap', ' tfb', [f' Rdiss_pos {lab}' for lab in label_obs], [f' Ldisstot_pos {lab}' for lab in label_obs]]
                writer.writerow(data)
            file.close()


        if do_cut == '' or do_cut == 'nocut':
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

        elif do_cut == 'ionization' or do_cut == 'ionizationHe': 
            # split above and belowe 5e4K
            if do_cut == 'ionizationHe':
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
        
    if do_cut == 'sections':
        np.savez( f'{abspath}/data/{folder}/wind/{check}_RdissSecTest.npz', diss_list=diss_list)

else:
    from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall
    commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    dataL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/Rdiss_LowResNewAMR.csv', delimiter=',', dtype=float, skiprows=1)
    snapL, tfbL, Rdiss_posL, Ldisstot_posL, Rdiss_negL, Ldisstot_negL =  dataL[:, 0], dataL[:, 1], dataL[:, 2], dataL[:, 3], dataL[:, 4], dataL[:, 5]
    data = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/Rdiss_NewAMR.csv', delimiter=',', dtype=float, skiprows=1)
    snap, tfb, Rdiss_pos, Ldisstot_pos, Rdiss_neg, Ldisstot_neg = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
    dataH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/paper1/Rdiss_HiResNewAMR.csv', delimiter=',', dtype=float, skiprows=1)
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
    # tfb_ratioDiss_L, ratio_DissL = ratio_BigOverSmall(tfb, Ldisstot_pos, tfbL, Ldisstot_posL)
    # tfb_ratioDiss_H, ratio_DissH = ratio_BigOverSmall(tfb, Ldisstot_pos, tfbH, Ldisstot_posH)
    # tfb_ratioRdiss_L, ratio_RdissL = ratio_BigOverSmall(tfb, Rdiss_pos, tfbL, Rdiss_posL)
    # tfb_ratioRdiss_H, ratio_RdissH = ratio_BigOverSmall(tfb, Rdiss_pos, tfbH, Rdiss_posH)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 16))
    ax1.plot(tfbL, Ldisstot_posL * prel.en_converter/prel.tsol_cgs, color = 'C1', label = 'Low')
    ax1.plot(tfb, Ldisstot_pos * prel.en_converter/prel.tsol_cgs, color = 'yellowgreen', label = 'Fid')
    ax1.plot(tfbH, Ldisstot_posH * prel.en_converter/prel.tsol_cgs, color = 'darkviolet', label = 'High')
    ax1.set_ylabel('Dissipation rate [erg/s]')
    ax1.legend(fontsize = 16)

    # ax3.plot(tfb_ratioDiss_L, ratio_DissL, color = 'C1')
    # ax3.plot(tfb_ratioDiss_L, ratio_DissL, '--', color = 'yellowgreen')
    # ax3.plot(tfb_ratioDiss_H, ratio_DissH, color = 'yellowgreen')
    # ax3.plot(tfb_ratioDiss_H, ratio_DissH, '--', color = 'darkviolet')

    ax2.plot(tfbL, Rdiss_posL/apo, c = 'C1')
    ax2.plot(tfb, Rdiss_pos/apo, c = 'yellowgreen')
    ax2.plot(tfbH, Rdiss_posH/apo, c = 'darkviolet')
    ax2.set_ylabel(r'$R_{\rm diss}$ [R$_{\rm a}$]')
    ax2.axhline(Rt/apo, color = 'k', linestyle = '--', label = r'$R_{\rm t}$')
    ax3.set_ylabel('Ratio of dissipation rate')

    # ax4.plot(tfb_ratioRdiss_L, ratio_RdissL, color = 'C1')
    # ax4.plot(tfb_ratioRdiss_L, ratio_RdissL, '--', color = 'yellowgreen')
    # ax4.plot(tfb_ratioRdiss_H, ratio_RdissH, color = 'yellowgreen')
    # ax4.plot(tfb_ratioRdiss_H, ratio_RdissH, '--', color = 'darkviolet')
    ax4.set_ylabel(r'ratio $R_{\rm diss}$')

    for ax in [ax1, ax2, ax3, ax4]:
        if ax != ax4:
            ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', length = 7, width = 1.2)
        ax.tick_params(axis='both', which='minor', length = 3, width = 1)
        ax.set_xlim(-0.1, 1.7)
    ax3.set_xlabel(r't [t$_{\rm fb}$]')
    ax4.set_xlabel(r't [t$_{\rm fb}$]')

    # compute the integral of Ldisstot_posH
    from scipy.integrate import trapezoid 
    idx_time_before_sqitch = np.argmin(np.abs(snapH - 21))
    # print(tfbH[:idx_time_before_sqitch])
    integral = trapezoid(Ldisstot_posH[:idx_time_before_sqitch] * prel.en_converter/prel.tsol_cgs, tfbH[:idx_time_before_sqitch] * t_fb_days_cgs)
    print(f'Integral of Ldisstot_posH up to snap 21: {integral:.2e} erg')
    idx_time25 = np.argmin(np.abs(tfbH - 0.25))
    # print(tfbH[:idx_time25])
    integral = trapezoid(Ldisstot_posH[:idx_time25] * prel.en_converter/prel.tsol_cgs, tfbH[:idx_time25] * t_fb_days_cgs)
    print(f'Integral of Ldisstot_posH up to t=0.25 tfb: {integral:.2e} erg')


