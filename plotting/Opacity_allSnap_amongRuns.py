"""Compare global quantities among runs with different opacity and extrapolation"""
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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import Utilities.prelude as prel
import matplotlib.colors as colors
from Utilities.operators import sort_list, find_ratio
from src import orbits as orb
from plotting.paper.IHopeIsTheLast import statistics_photo, split_data_red, ratio_BigOverSmall

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

# Luminosity and Rph
# Low
snapL, LumL, tfbL = split_data_red('LowRes')
snapLQ, LumLQ, tfbLQ = split_data_red('LowResOpacityNew')
snapL_AMR, LumL_AMR, tfbL_AMR = split_data_red('LowResNewAMR')
snapL_AMRRC, LumL_AMRRC, tfbL_AMRRC = split_data_red('LowResNewAMRRemoveCenter')
median_phL, percentile16L, percentile84L = statistics_photo(snapL, 'LowRes')
median_phLQ, _, _ = statistics_photo(snapLQ, 'LowResOpacityNew')
median_phL_AMR, _, _ = statistics_photo(snapL_AMR, 'LowResNewAMR')
median_phL_AMRRC, _, _ = statistics_photo(snapL_AMRRC, 'LowResNewAMRRemoveCenter')
# Fiducial 
snap, Lum, tfb = split_data_red('')
snapQ, LumQ, tfbQ = split_data_red('OpacityNew')
snapAMR, LumAMR, tfbAMR = split_data_red('OpacityNewNewAMR')
snapAMR_RC, LumAMR_RC, tfbAMR_RC = split_data_red('NewAMRRemoveCenter')
median_ph, percentile16, percentile84 = statistics_photo(snap, '')
median_phQ, _, _ = statistics_photo(snapQ, 'OpacityNew')
median_phAMR, _, _ = statistics_photo(snapAMR, 'OpacityNewNewAMR')
median_phAMR_RC, _, _ = statistics_photo(snapAMR_RC, 'NewAMRRemoveCenter')
# High
snapH, LumH, tfbH = split_data_red('HiRes')
median_phH, percentile16H, percentile84H = statistics_photo(snapH, 'HiRes')

# Dissipation (positive sign, which is the one of pericenter)
dataDissL = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}LowRes/Rdiss_LowRescutDen.txt')
tfbdissL, LDissL = dataDissL[0], dataDissL[2] *  prel.en_converter/prel.tsol_cgs 
dataDiss = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}/Rdiss_cutDen.txt')
tfbdiss, LDiss = dataDiss[0], dataDiss[2] * prel.en_converter/prel.tsol_cgs 
dataDissH = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}HiRes/Rdiss_HiRescutDen.txt')
tfbdissH, LDissH = dataDissH[0], dataDissH[2] *  prel.en_converter/prel.tsol_cgs 

dataDissAMR = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}OpacityNewNewAMR/Rdiss_OpacityNewNewAMRcutDen.txt')
tfbdissAMR, LDissAMR = dataDissAMR[0], dataDissAMR[2] * prel.en_converter/prel.tsol_cgs
dataDissAMR_RC = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}NewAMRRemoveCenter/Rdiss_NewAMRRemoveCentercutDen.txt')
tfbdissAMR_RC, LDissAMR_RC = dataDissAMR_RC[0], dataDissAMR_RC[2] * prel.en_converter/prel.tsol_cgs
dataDissL_AMR = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/Rdiss_LowResNewAMRcutDen.txt')
tfbdissL_AMR, LDissL_AMR = dataDissL_AMR[0], dataDissL_AMR[2] * prel.en_converter/prel.tsol_cgs
dataDissL_AMRRC = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}LowResNewAMRRemoveCenter/Rdiss_LowResNewAMRRemoveCentercutDen.txt')
tfbdissL_AMRRC, LDissL_AMRRC = dataDissL_AMRRC[0], dataDissL_AMRRC[2] * prel.en_converter/prel.tsol_cgs

# Luminosity
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.set_title('Fiducial', fontsize = 18)
ax1.plot(tfb, Lum, c = 'yellowgreen', label = 'OldExtr, OldAMR')
ax1.plot(tfbQ, LumQ, c = 'forestgreen', label = 'NewExtr, OldAMR')
ax1.plot(tfbAMR, LumAMR, c = 'royalblue', label = 'NewExtr, NewAMR')
ax1.plot(tfbAMR_RC, LumAMR_RC, c = 'k', label = r'Old$^*$Extr, NewAMR, RemoveCenter')
ax1.axhline(y=Ledd, c = 'k', linestyle = '-.', linewidth = 2)
ax1.text(0.1, 1.4*Ledd, r'$L_{\rm Edd}$', fontsize = 20)
ax1.set_ylabel(r'Luminosity [erg/s]')#, fontsize = 20)
# Low
ax3.set_title('Low', fontsize = 18)
ax3.plot(tfbL, LumL, c = 'C1', label = 'OldExtr, OldAMR')
ax3.plot(tfbLQ, LumLQ, c = 'goldenrod', label = 'NewExtr, OldAMR')
ax3.plot(tfbL_AMR, LumL_AMR, c = 'plum', label = r'Old$^*$Extr, NewAMR')
ax3.plot(tfbL_AMRRC, LumL_AMRRC, c = 'maroon', label = r'Old$^*$Extr, NewAMR, RemoveCenter')
ratioQ, tfb_ratioQ = ratio_BigOverSmall(tfb, Lum, tfbQ, LumQ)
ax2.plot(tfb_ratioQ, ratioQ, linewidth = 2, color = 'forestgreen', label = 'New vs Old Extr, Old AMR')
ax2.plot(tfb_ratioQ, ratioQ, linewidth = 2, color = 'yellowgreen', linestyle = (0, (5, 10)))
ratioAMR, tfb_ratioAMR = ratio_BigOverSmall(tfbQ, LumQ, tfbAMR, LumAMR)
ax2.plot(tfb_ratioAMR, ratioAMR, linewidth = 2, color = 'k', label = 'New Extr, New vs Old AMR')
ax2.plot(tfb_ratioAMR, ratioAMR, linewidth = 2, color = 'forestgreen', linestyle = (0, (5, 10)))
ratioLQ, tfb_ratioLQ = ratio_BigOverSmall(tfbL, LumL, tfbLQ, LumLQ)
ax4.plot(tfb_ratioLQ, ratioLQ, linewidth = 2, color = 'goldenrod', label = 'New vs Old Extr, Old AMR')
ax4.plot(tfb_ratioLQ, ratioLQ, linewidth = 2, color = 'C1', linestyle = (0, (5, 10)))
ratioAMR_RC, tfb_ratioAMR_RC = ratio_BigOverSmall(tfbAMR, LumAMR, tfbAMR_RC, LumAMR_RC)
ax2.plot(tfb_ratioAMR_RC, ratioAMR_RC, linewidth = 2, color = 'k', label = 'New vs Old*+RC Extr, New AMR')
ax2.plot(tfb_ratioAMR_RC, ratioAMR_RC, linewidth = 2, color = 'royalblue', linestyle = (0, (5, 10)))
ratioL_AMR, tfb_ratioL_AMR = ratio_BigOverSmall(tfbL, LumL, tfbL_AMR, LumL_AMR)
ax4.plot(tfb_ratioL_AMR, ratioL_AMR, linewidth = 2, color = 'plum', label = 'New Extr, New vs Old AMR')
ax4.plot(tfb_ratioL_AMR, ratioL_AMR, linewidth = 2, color = 'C1', linestyle = (0, (5, 10)))
ratioL_AMRRC, tfb_ratioL_AMRRC = ratio_BigOverSmall(tfbL_AMR, LumL_AMR, tfbL_AMRRC, LumL_AMRRC)
ax4.plot(tfb_ratioL_AMRRC, ratioL_AMRRC, linewidth = 2, color = 'maroon', label = 'New Extr,RC')
ax4.plot(tfb_ratioL_AMRRC, ratioL_AMRRC, linewidth = 2, color = 'plum', linestyle = (0, (5, 10)))

# Get the existing ticks 
original_ticks_y = ax1.get_yticks()
midpoints_y = (original_ticks_y[:-1] + original_ticks_y[1:]) / 2
new_ticks_y = np.sort(np.concatenate((original_ticks_y, midpoints_y)))
original_ticks = ax2.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(new_ticks)
    # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    # ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', width = 0.7, length=7)
    ax.tick_params(axis='x', which='minor', width = 0.7, length=5)
    ax.tick_params(axis='y', which='major', width = 1.5, length = 6.5, color = 'k')
    ax.tick_params(axis='y', which='minor', width = 1, length = 4, color = 'k')
    ax.set_xlim(0.01, np.max(tfb))
    ax.grid()
    # ax.axvline(tfb[np.argmin(np.abs(snap-248))], c = 'k', linestyle = '--', alpha = .5)
    # ax.axvline(tfb[np.argmin(np.abs(snap-161))], c = 'k', linestyle = '--', alpha = .5)
    # ax.axvline(tfb[np.argmin(np.abs(snap-164))], c = 'k', linestyle = '--', alpha = .5)
    # ax.axvline(tfb[np.argmin(np.abs(snap-115))], c = 'k', linestyle = '--', alpha = .5)
    # ax.axvline(tfb[np.argmin(np.abs(snap-240))], c = 'k', linestyle = '--', alpha = .5)
    if ax in [ax2, ax4]:
        ax.set_yscale('log')
        ax.set_ylim(.8, 15)
        ax.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 20)
    for ax in [ax1, ax3]:
        ax.set_yticks(new_ticks_y)
        labels = [str(np.round(tick,2)) if tick in original_ticks_y else "" for tick in new_ticks_y]       
        ax.set_yticklabels(labels)
        ax.set_yscale('log')
        ax.set_ylim(2e37, 4e42)
        ax.legend(fontsize = 16)
# ax3.axvline(tfbLQ[np.argmin(np.abs(snapL_AMR-225))], c = 'k', linestyle = '--', alpha = .5)
# ax3.axvline(tfbLQ[np.argmin(np.abs(snapL_AMR-227))], c = 'k', linestyle = '--', alpha = .5)
# ax3.axvline(tfbLQ[np.argmin(np.abs(snapL_AMR-229))], c = 'k', linestyle = '--', alpha = .5)

ax2.set_ylabel(r'$\mathcal{R}$ Luminosity')#, fontsize = 20)
# ax2.axhline(5, c = 'k', linestyle = '--', alpha = .5)
# ax2.axhline(1.5, c = 'k', linestyle = '--', alpha = .5)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/MazeOfRuns/fld_OpAMR.png', bbox_inches='tight')

#%% Low and Fid with removing center
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL_AMRRC, LumL_AMRRC, c = 'C1', label = r'L$_{\rm FLD}$ Low')
ax1.plot(tfbdissL_AMRRC, LDissL_AMRRC, ls = '--', c = 'C1', label = r'Diss tot Low')
ax1.plot(tfbAMR_RC, LumAMR_RC, c = 'yellowgreen', label = r'L$_{\rm FLD}$ Fid')
ax1.plot(tfbdissAMR_RC, LDissAMR_RC, ls = '--', c = 'yellowgreen', label = r'Diss tot Fid')
ax1.set_ylabel(r'Luminosity [erg/s]')#, fontsize = 20)
ax1.set_ylim(9e37, 8e42)
ax1.legend(fontsize = 16, loc = 'lower right')

ratioRC, tfb_ratioRC = ratio_BigOverSmall(tfbL_AMRRC, LumL_AMRRC, tfbAMR_RC, LumAMR_RC)
ax2.plot(tfb_ratioRC, ratioRC, linewidth = 2, color = 'k')
ax2.set_ylim(.9, 10)
ax2.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 20)
ax2.set_ylabel(r'$\mathcal{R}$ Luminosity')#, fontsize = 20)
ax2.tick_params(axis='y', which='minor', length = 3)
ax2.tick_params(axis='y', which='major', length = 5)

# Get the existing ticks on the x-axis
original_ticks = ax2.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2]:
    ax.grid()
    ax.set_xticks(new_ticks)
    # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    # ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', width=0.7, length=7)
    ax.tick_params(axis='x', which='minor', width=0.7, length=5)
    ax.tick_params(axis='y', which='major', width=0.7, length=7)
    ax.tick_params(axis='y', which='minor', width=0.7, length=5)
    ax.set_xlim(1e-2, np.max(tfbL_AMRRC))
    ax.set_yscale('log')
ax1.legend(fontsize = 18, loc = 'lower right')
plt.suptitle('Luminosity with new AMR+Removing Center', fontsize = 20)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/MazeOfRuns/fld_resRemoveCenter.png', bbox_inches='tight')

# %% Photosphere
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6)) #, gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, median_phL/apo, c = 'C1', label = 'Low old')
ax1.plot(tfbLQ, median_phLQ/apo, c = 'goldenrod', label = 'Low NewExtr, OldAMR')
ax1.plot(tfb, median_ph/apo, c = 'yellowgreen', label = 'Fid old')
ax1.plot(tfbL_AMR, median_phL_AMR/apo, c ='plum', label = r'Low Old$^*$Extr, NewAMR')
ax1.plot(tfbL_AMRRC, median_phL_AMRRC/apo, c = 'maroon', label = r'Low Old$^*$Extr, NewAMR, RemoveCenter')
ax1.plot(tfbQ, median_phQ/apo, c = 'forestgreen', label = 'Fid, NewExtr, OldAMR')
ax1.plot(tfbAMR, median_phAMR/apo, c = 'royalblue', label = 'Fid NewExtr, NewAMR')
ax1.plot(tfbH, median_phH/apo, c = 'darkviolet', label = 'High old')
ax1.axhline(Rt/apo, c = 'k', linestyle = '--', linewidth = 2)
ax1.text(1.6, 1.2*Rt/apo, r'$R_{\rm t}$', fontsize = 18)
ax1.set_yscale('log')
ax1.set_ylabel(r'median $R_{\rm ph} [R_{\rm a}]$')
ax1.legend(fontsize = 12)
ax1.tick_params(axis='y', which='major', width = 1.5, length = 5, color = 'k')
ax1.tick_params(axis='y', which='minor', width = 1, length = 3, color = 'k')

# ax2.plot(tfbL, ratio_medianRphL, linewidth = 2, color = 'yellowgreen')
# ax2.plot(tfbL, ratio_medianRphL, linestyle = (0, (5, 10)), linewidth = 2, color = 'C1')
# ax2.plot(tfbH, ratio_medianRphH, linewidth = 2, color = 'yellowgreen')
# ax2.plot(tfbH, ratio_medianRphH, linestyle = (0, (5, 10)), linewidth = 2, color = 'darkviolet')
# ax2.plot(tfbLQ, ratio_medianRphLQ, linewidth = 2, color = 'plum')
# ax2.plot(tfbQ, ratio_medianRphQ, linewidth = 2, color = 'forestgreen', label = 'Fid')
# ax2.plot(tfbAMR, ratio_medianRphAMR, linewidth = 2, color = 'k', label = 'Fid AMR')
# ax2.set_xlabel(r't [$t_{\rm fb}$]')
# ax2.set_ylabel(r'$\mathcal{R}$ $R_{\rm ph}$')
# original_ticksy = ax2.get_yticks()
# midpointsy = (original_ticksy[:-1] + original_ticksy[1:]) / 2
# new_ticksy = np.sort(np.concatenate((original_ticksy, midpointsy)))
# ax2.set_yticks(new_ticksy)
# ax2.set_ylim(.8, 3.5)

# original_ticks = ax2.get_xticks()
# midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
# new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
# for ax in [ax1, ax2]:
ax1.set_xticks(new_ticks)
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
ax1.set_xticklabels(labels)
ax1.grid()
ax1.set_xlim(0, np.max(tfb))
ax1.set_xlabel(r't [$t_{fb}$]')
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/MazeOfRuns/Rph_OpAMR.png', bbox_inches='tight')
# %%
