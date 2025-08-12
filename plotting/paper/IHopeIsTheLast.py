"""Bunch of plots for the paper"""
#%%
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
from matplotlib.collections import LineCollection
import Utilities.prelude as prel
import matplotlib.colors as colors
from Utilities.operators import sort_list, find_ratio
from src import orbits as orb

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
tfallback = 2.5777261297507925 * 24 * 3600 #2.5 days
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Rt = Rstar * (Mbh/mstar)**(1/3)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
#
# FUNCTIONS
##
def L_Edd_k(k): # cgs
    L_e = 1.38*1e42 * (0.34/k)
    return L_e

def split_data_red(check):
    """Split the data in the file into two lists: time and luminosity."""
    if check in ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']:
         filename = f'{abspath}/data/{commonfold}{check}/{check}_red.csv'
    else:
        filename = f'{abspath}/data/opacity_tests/{commonfold}{check}/{check}_red.csv'
    data = np.loadtxt(filename, delimiter=',', dtype=float)
    snap = np.array([int(s) for s in data[:,0]])
    time = data[:, 1]
    luminosity = data[:, 2]
    snap, luminosity, time = sort_list([snap, luminosity, time], time, unique=True)
    return snap, luminosity, time

def ratio_BigOverSmall(tfb1, Lum1, tfb2, Lum2):
    """Calculate the ratio of luminosities between two datasets."""
    shorter_tfb = tfb1
    shorter_Lum = Lum1
    longer_tfb = tfb2
    longer_Lum = Lum2
    if len(tfb2) < len(tfb1):
        shorter_tfb = tfb2
        shorter_Lum = Lum2
        longer_tfb = tfb1
        longer_Lum = Lum1
    ratio = []
    time_ratio = []
    step_tfb = max(np.mean(np.diff(longer_tfb)), 
                   np.mean(np.diff(shorter_tfb)))
    for i, time in enumerate(shorter_tfb):
        if time < np.min(longer_tfb):
            continue
        # Find the closest time in the longer dataset
        idx = np.argmin(np.abs(longer_tfb - time))
        if np.abs(longer_tfb[idx] - time) > step_tfb:
            continue
        # Calculate the ratio of luminosities
        ratio.append(find_ratio(shorter_Lum[i], longer_Lum[idx]))
        time_ratio.append(time)
    return time_ratio, ratio

def statistics_photo(snaps, check):
    """Calculate the statistics of the photosphere for a given set of snapshots."""
    mean_ph = np.zeros(len(snaps))
    median_ph = np.zeros(len(snaps))
    percentile16 = np.zeros(len(snaps))
    percentile84 = np.zeros(len(snaps))
    for i, snapi in enumerate(snaps):
        if check in ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']:
            photo = np.loadtxt(f'{abspath}/data/{commonfold}{check}/photo/{check}_photo{snapi}.txt')
        else:
            photo = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}{check}/photo/{check}_photo{snapi}.txt')
        xph_i, yph_i, zph_i = photo[0], photo[1], photo[2]
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_ph[i] = np.mean(rph_i)
        median_ph[i] = np.median(rph_i)
        percentile16[i] = np.percentile(rph_i, 16)
        percentile84[i] = np.percentile(rph_i, 84)
    return mean_ph, median_ph, percentile16, percentile84

if __name__ == '__main__':
    # Luminosity 
    # Low
    snapL, LumL, tfbL = split_data_red('LowResNewAMR')
    snap, Lum, tfb = split_data_red('NewAMR')
    snapH, LumH, tfbH = split_data_red('HiResNewAMR')
    # Rph 
    mean_phL, median_phL, percentile16L, percentile84L = statistics_photo(snapL, 'LowResNewAMR')
    mean_ph, median_ph, percentile16, percentile84 = statistics_photo(snap, 'NewAMR')
    mean_phH, median_phH, percentile16H, percentile84H = statistics_photo(snapH, 'HiResNewAMR')

    # Dissipation (positive sign, which is the one of pericenter)
    dataDissL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/Rdiss_LowResNewAMRcutDen.txt')
    tfbdissL, LDissL = dataDissL[0], dataDissL[2] *  prel.en_converter/prel.tsol_cgs 
    dataDiss = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/Rdiss_NewAMRcutDen.txt')
    tfbdiss, LDiss = dataDiss[0], dataDiss[2] * prel.en_converter/prel.tsol_cgs 
    dataDissH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/Rdiss_HiResNewAMRcutDen.txt')
    tfbdissH, LDissH = dataDissH[0], dataDissH[2] *  prel.en_converter/prel.tsol_cgs 

    # dataDissOld = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}/Rdiss_cutDen.txt')
    # tfbdissOld, LDissOld = dataDissOld[0], dataDissOld[2] * prel.en_converter/prel.tsol_cgs 

    ######## Energy density 
    # Rad_den_R_snapL = np.load(f'{abspath}/data/{commonfold}LowRes/LowRes267_Radden.npy')
    # r_arrL = np.load(f'{abspath}/data/{commonfold}LowRes/LowRes267_R_Radden.npy')
    # Rad_den_R_snap = np.load(f'{abspath}/data/{commonfold}/267_Radden.npy')
    # r_arr = np.load(f'{abspath}/data/{commonfold}/267_R_Radden.npy') # it's the same for all resolutions
    # Rad_den_R_snapH = np.load(f'{abspath}/data/{commonfold}HiRes/HiRes267_Radden.npy')
    # r_arrH = np.load(f'{abspath}/data/{commonfold}HiRes/HiRes267_R_Radden.npy')

    ######## Plot #######
    # Luminosity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    ax1.plot(tfbL, LumL, c = 'C1', label = 'Low')
    # ax1.plot(tfbdissL, LDissL, ls = '--', c = 'C1')
    ax1.plot(tfb, Lum, c = 'yellowgreen', label = 'Fid')
    # ax1.plot(tfbdiss, LDiss, ls = '--', c = 'yellowgreen')
    # ax1.plot(tfbdissOld, LDissOld, ls = '--', c = 'forestgreen', label = 'Old')
    ax1.plot(tfbH, LumH, c = 'darkviolet', label = 'High')
    # ax1.plot(tfbdissH, LDissH, ls = '--', c = 'darkviolet')

    ax1.axhline(y=Ledd, c = 'k', linestyle = '-.', linewidth = 2)
    ax1.axhline(y=Ledd, c = 'k', linestyle = '-.', linewidth = 2)
    ax1.text(0.1, 1.4*Ledd, r'$L_{\rm Edd}$', fontsize = 20)
    ax1.set_ylabel(r'Luminosity [erg/s]')

    original_ticks = ax1.get_yticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    ax1.set_yticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax1.set_yticklabels(labels)
    ax1.tick_params(axis='y', which='major', width = 1.5, length = 5, color = 'k')
    ax1.tick_params(axis='y', which='minor', width = 1, length = 3, color = 'k')
    ax1.set_yscale('log')
    ax1.set_ylim(1e37, 8e42)
    ax1.grid()
    ax1.legend(fontsize = 18, loc = 'lower right')

    tfb_ratioL, ratioL  = ratio_BigOverSmall(tfb, Lum, tfbL, LumL)
    ax2.plot(tfb_ratioL, ratioL, linewidth = 2, color = 'yellowgreen')
    ax2.plot(tfb_ratioL, ratioL, linewidth = 2, color = 'C1', linestyle = (0, (5, 10)), label = 'Low,Fid')
    tfb_ratioH, ratioH  = ratio_BigOverSmall(tfb, Lum, tfbH, LumH)
    ax2.plot(tfb_ratioH, ratioH, linewidth = 2, color = 'yellowgreen')
    ax2.plot(tfb_ratioH, ratioH, linewidth = 2, color = 'darkviolet', linestyle = (0, (5, 10)), label = 'Fid,High')
    ax2.set_ylim(.8, 15)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', which='major', width = 1.5, length = 5, color = 'k')
    ax2.tick_params(axis='y', which='minor', width = 1, length = 3, color = 'k')
    ax2.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 20)
    ax2.set_ylabel(r'$\mathcal{R}$ Luminosity')#, fontsize = 20)
    ax2.grid()
    ax2.tick_params(axis='y', which='minor', length = 3)
    ax2.tick_params(axis='y', which='major', length = 5)

    # Get the existing ticks on the x-axis
    original_ticks = ax2.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in [ax1, ax2]:
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', width=0.7, length=7)
        ax.tick_params(axis='x', which='minor', width=0.7, length=5)
        ax.set_xlim(0, np.max(tfbL))
    ax1.legend(fontsize = 18, loc = 'lower right')
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/paper/fld.pdf', bbox_inches='tight')

    # Photosphere
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    ax1.plot(tfbL, median_phL/apo, c = 'C1', label = 'Low')
    ax1.plot(tfbL, percentile84L/apo, c = 'C1', alpha = 0.4, linestyle = '--')
    ax1.plot(tfbL, percentile16L/apo, c = 'C1', alpha = 0.4, linestyle = '--')
    ax1.fill_between(tfbL, percentile16L/apo, percentile84L/apo, color = 'C1', alpha = 0.4)
    # Fid
    ax1.plot(tfb, median_ph/apo, c = 'yellowgreen', label = 'Fid')
    ax1.plot(tfb, percentile84/apo, c = 'yellowgreen', alpha = 0.3, linestyle = '--')
    ax1.plot(tfb, percentile16/apo, c = 'yellowgreen', alpha = 0.3, linestyle = '--')
    ax1.fill_between(tfb, percentile16/apo, percentile84/apo, color = 'yellowgreen', alpha = 0.3)
    # High
    ax1.plot(tfbH, median_phH/apo, c = 'darkviolet', label = 'High')
    ax1.plot(tfbH, percentile84H/apo, c = 'darkviolet', alpha = 0.2, linestyle = '--')
    ax1.plot(tfbH, percentile16H/apo, c = 'darkviolet', alpha = 0.2, linestyle = '--')
    ax1.fill_between(tfbH, percentile16H/apo, percentile84H/apo, color = 'darkviolet', alpha = 0.2)
    ax1.axhline(Rt/apo, c = 'k', linestyle = '--', linewidth = 2)
    ax1.text(1.5, 1.2*Rt/apo, r'$R_{\rm t}$', fontsize = 18)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'median $R_{\rm ph} [R_{\rm a}]$')
    ax1.legend(fontsize = 18)
    ax1.tick_params(axis='y', which='major', width = 1.5, length = 5, color = 'k')
    ax1.tick_params(axis='y', which='minor', width = 1, length = 3, color = 'k')

    tfb_ratioL_Rph, ratio_medianRphL = ratio_BigOverSmall(tfb, median_ph, tfbL, median_phL)
    ax2.plot(tfb_ratioL_Rph, ratio_medianRphL, linewidth = 2, color = 'yellowgreen')
    ax2.plot(tfb_ratioL_Rph, ratio_medianRphL, linestyle = (0, (5, 10)), linewidth = 2, color = 'C1')
    tfb_ratioH_Rph, ratio_medianRphH = ratio_BigOverSmall(tfb, median_ph, tfbH, median_phH)
    ax2.plot(tfb_ratioH_Rph, ratio_medianRphH, linewidth = 2, color = 'yellowgreen')
    ax2.plot(tfb_ratioH_Rph, ratio_medianRphH, linestyle = (0, (5, 10)), linewidth = 2, color = 'darkviolet')
    ax2.set_xlabel(r't [$t_{\rm fb}$]')
    ax2.set_ylabel(r'$\mathcal{R}$ $R_{\rm ph}$')
    # original_ticksy = ax2.get_yticks()
    # midpointsy = (original_ticksy[:-1] + original_ticksy[1:]) / 2
    # new_ticksy = np.sort(np.concatenate((original_ticksy, midpointsy)))
    # ax2.set_yticks(new_ticksy)
    ax2.set_ylim(.8, 3.5)

    original_ticks = ax2.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in [ax1, ax2]:
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        ax.grid()
        ax.set_xlim(0.01, np.max(tfbL))
    ax2.set_xlabel(r't [$t_{fb}$]')
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/paper/Rph_ratio.pdf', bbox_inches='tight')

    #%% Radiation energy density 
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    # ax1.plot(r_arr/apo, Rad_den_R_snapL * prel.en_den_converter, c = 'C1', label = 'Low')
    # ax1.plot(r_arr/apo, Rad_den_R_snap * prel.en_den_converter, c = 'yellowgreen', label = 'Fid')
    # ax1.plot(r_arr/apo, Rad_den_R_snapH * prel.en_den_converter, c = 'darkviolet', label = 'High')
    # ax1.set_ylabel(r'$u$ [erg/cm$^3$s]')
    # ax1.legend(fontsize = 18)

    # ax2.plot(r_arr/apo, diffuL, color = 'C1')
    # ax2.plot(r_arr/apo, diffuH, color = 'darkviolet')
    # ax2.set_xlabel(r'$R [R_\odot$]')
    # ax2.set_ylabel(r'$2|u_{\rm x}-u_{\rm Fid}|/(u_{\rm x}+u_{\rm Fid})$')
    # for ax in [ax1, ax2]:
    #     ax.set_yscale('log')
    #     ax.grid()
    # plt.suptitle('Radiation energy density u from the line of sight of the observers')
    # plt.tight_layout()
    # ## 
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    # ax1.plot(r_arr/apo, Rad_den_R_snapL * prel.en_den_converter, c = 'C1', label = 'Low')
    # ax1.plot(r_arr/apo, Rad_den_R_snap * prel.en_den_converter, c = 'yellowgreen', label = 'Fid')
    # ax1.plot(r_arr/apo, Rad_den_R_snapH * prel.en_den_converter, c = 'darkviolet', label = 'High')
    # ax1.set_ylabel(r'$u_{\rm rad}$ [erg/(cm$^3$s)]')#, fontsize = 18)
    # ax1.legend(fontsize = 18)
    # ax1.set_yscale('log')

    # ax2.plot(r_arr/apo, ratiouL, linewidth = 2, color = 'yellowgreen')
    # ax2.plot(r_arr/apo, ratiouL, linestyle = (0, (5, 10)), linewidth = 2, color = 'C1')
    # ax2.plot(r_arr/apo, ratiouH, linewidth = 2, color = 'yellowgreen')
    # ax2.plot(r_arr/apo, ratiouH, linestyle = (0, (5, 10)), linewidth = 2, color = 'darkviolet')
    # ax2.set_xlabel(r'$R [R_{\rm a}$]')#, fontsize = 18)
    # ax2.set_ylabel(r'$\mathcal{R} u_{\rm rad}$')#, fontsize = 18)
    # ax2.set_ylim(1, 3)
    # # Get the existing ticks on the x-axis
    # original_ticks = np.arange(1, 11, 2, dtype=int)
    # midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    # new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    # for ax in [ax1, ax2]:
    #     ax.set_xticks(new_ticks)
    #     labels = [str(int(tick)) if tick in original_ticks else "" for tick in new_ticks]       
    #     ax.set_xticklabels(labels)
    #     ax.tick_params(axis='y', which='minor', length = 3, width = 0.7)
    #     ax.tick_params(axis='x', which='minor', length = 3, width = 0.7)
    #     ax.tick_params(axis='y', which='major', length = 5, width = 1)
    #     ax.set_xlim(0, np.max(r_arr)/apo)
    #     ax.grid()
    # # plt.suptitle('Radiation energy density u from the line of sight of the observers')
    # plt.tight_layout()
    # plt.savefig(f'{abspath}/Figs/multiple/Rad.png', bbox_inches='tight')
    # plt.savefig(f'{abspath}/Figs/paper/Rad.pdf', bbox_inches='tight')

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    # ax1.plot(r_arr/apo, Rad_den_R_snapL * r_arr**3 * prel.en_converter, c = 'C1', label = 'Low')
    # ax1.plot(r_arr/apo, Rad_den_R_snap * r_arr**3 * prel.en_converter, c = 'yellowgreen', label = 'Fid')
    # ax1.plot(r_arr/apo, Rad_den_R_snapH * r_arr**3 * prel.en_converter, c = 'darkviolet', label = 'High')
    # ax1.set_ylabel(r'Radiation energy [erg/s]')#, fontsize = 18)
    # ax1.legend(fontsize = 18)
    # ax1.set_yscale('log')

    # ax2.plot(r_arr/apo, ratiouL, linewidth = 2, color = 'yellowgreen')
    # ax2.plot(r_arr/apo, ratiouL, linestyle = (0, (5, 10)), linewidth = 2, color = 'C1')
    # ax2.plot(r_arr/apo, ratiouH, linewidth = 2, color = 'yellowgreen')
    # ax2.plot(r_arr/apo, ratiouH, linestyle = (0, (5, 10)), linewidth = 2, color = 'darkviolet')
    # ax2.set_xlabel(r'$R [R_{\rm a}$]')#, fontsize = 18)
    # ax2.set_ylabel(r'$\mathcal{R} u_{\rm rad}$')#, fontsize = 18)
    # ax2.set_ylim(1, 3)
    # # Get the existing ticks on the x-axis
    # original_ticks = np.arange(1, 11, 2, dtype=int)
    # midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    # new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    # for ax in [ax1, ax2]:
    #     ax.set_xticks(new_ticks)
    #     labels = [str(int(tick)) if tick in original_ticks else "" for tick in new_ticks]       
    #     ax.set_xticklabels(labels)
    #     ax.tick_params(axis='y', which='minor', length = 3, width = 0.7)
    #     ax.tick_params(axis='x', which='minor', length = 3, width = 0.7)
    #     ax.tick_params(axis='y', which='major', length = 5, width = 1)
    #     ax.set_xlim(0, np.max(r_arr)/apo)
    #     ax.grid()
    # # plt.suptitle('Radiation energy density u from the line of sight of the observers')
    # plt.tight_layout()


    #%% OE and IE
    # Load data
    col_ie, col_orb_en_pos, col_orb_en_neg, col_rad = \
        np.load(f'{abspath}/data/{commonfold}NewAMR/convE_NewAMR.npy') #shape (3, len(tfb), len(radii))
    col_orb_en = np.abs(col_orb_en_pos + col_orb_en_neg)
    _, tfb_oe = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/convE_NewAMR_days.txt')

    col_ieL, col_orb_en_posL, col_orb_en_negL, col_radL = \
        np.load(f'{abspath}/data/{commonfold}LowResNewAMR/convE_LowResNewAMR.npy') #shape (3, len(tfb), len(radii))
    col_orb_enL = np.abs(col_orb_en_posL + col_orb_en_negL)
    _, tfb_oeL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/convE_LowResNewAMR_days.txt')

    col_ieH, col_orb_en_posH, col_orb_en_negH, col_radH = \
        np.load(f'{abspath}/data/{commonfold}HiResNewAMR/convE_HiResNewAMR.npy')
    col_orb_enH = np.abs(col_orb_en_posH + col_orb_en_negH)
    _, tfb_oeH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/convE_HiResNewAMR_days.txt')

    # relative differences 
    tfb_ratio_orbL, ratio_orbL = ratio_BigOverSmall(tfb_oe, col_orb_en, tfb_oeL, col_orb_enL)
    _, ratio_ieL = ratio_BigOverSmall(tfb_oe, col_ie, tfb_oeL, col_ieL)
    _, ratio_radL = ratio_BigOverSmall(tfb_oe, col_rad, tfb_oeL, col_radL)

    # relative difference 
    tfb_ratio_orbH, ratio_orbH = ratio_BigOverSmall(tfb_oe, col_orb_en, tfb_oeH, col_orb_enH)
    _, ratio_ieH = ratio_BigOverSmall(tfb_oe, col_ie, tfb_oeH, col_ieH)
    _, ratio_radH = ratio_BigOverSmall(tfb_oe, col_rad, tfb_oeH, col_radH)

    # Plot OE and IE
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 7), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    ax1.plot(tfb_oeL, col_orb_enL*prel.en_converter, label = r'Low', c = 'darkorange')
    ax1.plot(tfb_oe, col_orb_en*prel.en_converter, label = r'Fid', c = 'yellowgreen')
    ax1.plot(tfb_oeH, col_orb_enH*prel.en_converter, label = r'High', c = 'darkviolet')
    ax1.set_ylabel(r'$|$Orbital energy$|$ [erg]')#, fontsize = 18)
    ax1.legend(fontsize = 15)
    ax1.set_ylim(7e47, 2e48)

    ax2.plot(tfb_oeL, col_ieL*prel.en_converter, label = r'Low', c = 'darkorange')
    ax2.plot(tfb_oe, col_ie*prel.en_converter, label = r'Fid', c = 'yellowgreen')
    ax2.plot(tfb_oeH, col_ieH*prel.en_converter, label = r'High', c = 'darkviolet')
    ax2.set_ylabel(r'Internal energy [erg]')#, fontsize = 18)

    ax3.plot(tfb_ratio_orbL, ratio_orbL, linewidth = 2.5, c = 'darkorange',  label = r'Low and Fid',)
    ax3.plot(tfb_ratio_orbL, ratio_orbL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax3.plot(tfb_ratio_orbH, ratio_orbH, label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax3.plot(tfb_ratio_orbH, ratio_orbH,linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax3.set_ylabel(r'$\mathcal{R}$ orbital energy', fontsize = 22)
    ax3.set_ylim(.95, 1.2)

    ax4.plot(tfb_ratio_orbL, ratio_ieL, label = r'Low and Fid', linewidth = 2, c = 'darkorange')
    ax4.plot(tfb_ratio_orbL, ratio_ieL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax4.plot(tfb_ratio_orbH, ratio_ieH, label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax4.plot(tfb_ratio_orbH, ratio_ieH, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    # ax4.legend(fontsize = 18)
    ax4.set_ylabel(r'$\mathcal{R}$ internal energy', fontsize = 22)
    original_ticksy = ax4.get_yticks()
    midpointsy = (original_ticksy[:-1] + original_ticksy[1:]) / 2
    new_ticksy = np.sort(np.concatenate((original_ticksy, midpointsy)))
    ax4.set_yticks(new_ticksy)
    ax4.set_ylim(.95, 1.5)

    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        if ax in [ax3, ax4]:
            ax.set_xlabel(r'$t [t_{\rm fb}]$', fontsize = 18)
        else:
            ax.set_yscale('log')
        ax.grid()
        ax.set_xlim(.05, 1.7) 
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/paper/OeIe.pdf', bbox_inches='tight')


    # %% all three energies
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(25, 9), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    ax1.plot(tfb_oeL, col_orb_enL*prel.en_converter*1e-47, label = r'Low', c = 'darkorange')
    ax1.plot(tfb_oe, col_orb_en*prel.en_converter*1e-47, label = r'Fid', c = 'yellowgreen')
    ax1.plot(tfb_oeH, col_orb_enH*prel.en_converter*1e-47, label = r'High', c = 'darkviolet')
    ax1.set_ylabel(r'Energy')#, fontsize = 18)
    ax1.set_title(r'$|$Orbital energy$| [10^{47}$ erg]', fontsize = 27)
    ax1.legend(fontsize = 15)

    ax2.plot(tfb_oeL, col_ieL*prel.en_converter*1e-46, label = r'Low', c = 'darkorange')
    ax2.plot(tfb_oe, col_ie*prel.en_converter*1e-46, label = r'Fid', c = 'yellowgreen')
    ax2.plot(tfb_oeH, col_ieH*prel.en_converter*1e-46, label = r'High', c = 'darkviolet')
    ax2.set_title(r'Internal energy [$10^{46}$ erg]', fontsize = 27)

    ax3.plot(tfb_oeL, col_radL*prel.en_converter, label = r'Low', c = 'darkorange')
    ax3.plot(tfb_oe, col_rad*prel.en_converter, label = r'Fid', c = 'yellowgreen')
    ax3.plot(tfb_oeH, col_radH*prel.en_converter, label = r'High', c = 'darkviolet')
    ax3.set_title(r'Radiation energy [erg]', fontsize = 27)
    ax3.set_yscale('log')

    ax4.plot(tfb_ratio_orbL, ratio_orbL, linewidth = 2.5, c = 'darkorange',  label = r'Low and Fid',)
    ax4.plot(tfb_ratio_orbL, ratio_orbL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax4.plot(tfb_ratio_orbH, ratio_orbH, label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax4.plot(tfb_ratio_orbH, ratio_orbH,linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax4.set_ylabel(r'$\mathcal{R}$', fontsize = 26)
    ax4.set_ylim(.99, 1.1)

    ax5.plot(tfb_ratio_orbL, ratio_ieL, label = r'Low and Fid', linewidth = 2, c = 'darkorange')
    ax5.plot(tfb_ratio_orbL, ratio_ieL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax5.plot(tfb_ratio_orbH, ratio_ieH, label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax5.plot(tfb_ratio_orbH, ratio_ieH, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    # original_ticksy = ax5.get_yticks()
    # midpointsy = (original_ticksy[:-1] + original_ticksy[1:]) / 2
    # new_ticksy = np.sort(np.concatenate((original_ticksy, midpointsy)))
    # ax5.set_yticks(new_ticksy)
    # ax5.set_ylim(.95, 2)

    ax6.plot(tfb_ratio_orbL, ratio_radL, linewidth = 2.5, c = 'darkorange',  label = r'Low and Fid',)
    ax6.plot(tfb_ratio_orbL, ratio_radL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax6.plot(tfb_ratio_orbH, ratio_radH, label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax6.plot(tfb_ratio_orbH, ratio_radH,linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    # original_ticksy = ax6.get_yticks()
    # midpointsy = (original_ticksy[:-1] + original_ticksy[1:]) / 2
    # new_ticksy = np.sort(np.concatenate((original_ticksy, midpointsy)))
    # ax6.set_yticks(new_ticksy)
    ax6.set_ylim(.95, 3)

    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xticks(new_ticks)
        # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        # ax.set_xticklabels(labels)
        ax.grid()
        ax.set_xlim(.05, 1.7) 
        if ax in [ax4, ax5, ax6]:
            ax.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 18)
    plt.tight_layout()
    # plt.savefig(f'{abspath}/Figs/paper/OeIeRad.pdf', bbox_inches='tight')
    # %% bound unbound
    fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2, figsize=(18, 9), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    ax1.plot(tfb_oeL, col_orb_en_negL*prel.en_converter*1e-48, label = r'Low', c = 'darkorange')
    ax1.plot(tfb_oe, col_orb_en_neg*prel.en_converter*1e-48, label = r'Fid', c = 'yellowgreen')
    ax1.plot(tfb_oeH, col_orb_en_negH*prel.en_converter*1e-48, label = r'High', c = 'darkviolet')
    ax1.set_ylabel(r'Orbital Energy [$10^{48}$ erg]')#, fontsize = 18)
    ax1.set_title(r'Bound material ', fontsize = 27)
    ax1.legend(fontsize = 15)
    ax1.set_ylim(-13, -12)

    ax2.plot(tfb_oeL, col_orb_en_posL*prel.en_converter*1e-48, label = r'Low', c = 'darkorange')
    ax2.plot(tfb_oe, col_orb_en_pos*prel.en_converter*1e-48, label = r'Fid', c = 'yellowgreen')
    ax2.plot(tfb_oeH, col_orb_en_posH*prel.en_converter*1e-48, label = r'High', c = 'darkviolet')
    ax2.set_title(r'Unbound material', fontsize = 27)
    ax2.set_ylim(11, 12)

    tfb_ratio_orbnegL, ratio_orbnegL = ratio_BigOverSmall(tfb_oe, col_orb_en_neg, tfb_oeL, col_orb_en_negL)
    tfb_ratio_orbnegH, ratio_orbnegH = ratio_BigOverSmall(tfb_oe, col_orb_en_neg, tfb_oeH, col_orb_en_negH)
    _, ratio_orbposL = ratio_BigOverSmall(tfb_oe, col_orb_en_pos, tfb_oeL, col_orb_en_posL)
    _, ratio_orbposH = ratio_BigOverSmall(tfb_oe, col_orb_en_pos, tfb_oeH, col_orb_en_posH)

    ax4.plot(tfb_ratio_orbnegL, ratio_orbnegL, linewidth = 2.5, c = 'darkorange',  label = r'Low and Fid',)
    ax4.plot(tfb_ratio_orbnegL, ratio_orbnegL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax4.plot(tfb_ratio_orbnegH, ratio_orbnegH,  label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax4.plot(tfb_ratio_orbnegH, ratio_orbnegH, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax4.set_ylabel(r'$\mathcal{R}$', fontsize = 26)
    ax4.set_ylim(.99, 1.05)

    ax5.plot(tfb_ratio_orbnegL, ratio_orbposL, label = r'Low and Fid', linewidth = 2, c = 'darkorange')
    ax5.plot(tfb_ratio_orbnegL, ratio_orbposL, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax5.plot(tfb_ratio_orbnegH, ratio_orbposH, label = r'Fid and High', linewidth = 2, c = 'darkviolet')
    ax5.plot(tfb_ratio_orbnegH, ratio_orbposH, linestyle = (0, (5, 10)), linewidth = 2, c = 'yellowgreen')
    ax5.set_ylim(.99, 1.05)
    
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_xticks(new_ticks)
        ax.grid()
        ax.set_xlim(.05, 1.7) 
        if ax in [ax4, ax5]:
            ax.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 18)
    plt.tight_layout()
    # %%
