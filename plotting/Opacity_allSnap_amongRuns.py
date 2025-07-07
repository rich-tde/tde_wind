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
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
import matplotlib.colors as colors
from Utilities.sections import make_slices
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

#%% Luminosity and Rph
# Low
snapL, LumL, tfbL = split_data_red('LowRes')
snapLQ, LumLQ, tfbLQ = split_data_red('LowResOpacityNew')
snapL_AMR, LumL_AMR, tfbL_AMR = split_data_red('LowResNewAMR')
snapL_AMRRC, LumL_AMRRC, tfbL_AMRRC = split_data_red('LowResNewAMRRemoveCenter')
_, median_phL, percentile16L, percentile84L = statistics_photo(snapL, 'LowRes')
_, median_phLQ, _, _ = statistics_photo(snapLQ, 'LowResOpacityNew')
_, median_phL_AMR, _, _ = statistics_photo(snapL_AMR, 'LowResNewAMR')
_, median_phL_AMRRC, _, _ = statistics_photo(snapL_AMRRC, 'LowResNewAMRRemoveCenter')
# Fiducial 
snap, Lum, tfb = split_data_red('')
snapQ, LumQ, tfbQ = split_data_red('OpacityNew')
# snapAMR_Q, LumAMR_Q, tfbAMR_Q = split_data_red('OpacityNewNewAMR')
snapAMR, LumAMR, tfbAMR = split_data_red('NewAMR')
snapAMR_RC, LumAMR_RC, tfbAMR_RC = split_data_red('NewAMRRemoveCenter')
_, median_ph, percentile16, percentile84 = statistics_photo(snap, '')
_, median_phQ, _, _ = statistics_photo(snapQ, 'OpacityNew')
# _, median_phAMR_Q, _, _ = statistics_photo(snapAMR, 'OpacityNewNewAMR')
_, median_phAMR, _, _ = statistics_photo(snapAMR, 'NewAMR')
_, median_phAMR_RC, _, _ = statistics_photo(snapAMR_RC, 'NewAMRRemoveCenter')
# High
snapH, LumH, tfbH = split_data_red('HiRes')
_, median_phH, percentile16H, percentile84H = statistics_photo(snapH, 'HiRes')

# Dissipation (positive sign, which is the one of pericenter)
dataDissL = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}LowRes/Rdiss_LowRescutDen.txt')
tfbdissL, LDissL = dataDissL[0], dataDissL[2] *  prel.en_converter/prel.tsol_cgs 
dataDiss = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}/Rdiss_cutDen.txt')
tfbdiss, LDiss = dataDiss[0], dataDiss[2] * prel.en_converter/prel.tsol_cgs 
dataDissH = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}HiRes/Rdiss_HiRescutDen.txt')
tfbdissH, LDissH = dataDissH[0], dataDissH[2] *  prel.en_converter/prel.tsol_cgs 

# dataDissAMR_Q = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}OpacityNewNewAMR/Rdiss_OpacityNewNewAMRcutDen.txt')
# tfbdissAMR_Q, LDissAMR_Q = dataDissAMR_Q[0], dataDissAMR_Q[2] * prel.en_converter/prel.tsol_cgs
dataDissAMR = np.loadtxt(f'{abspath}/data/opacity_tests/{commonfold}NewAMR/Rdiss_NewAMRcutDen.txt')
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
tfb_ratioQ, ratioQ = ratio_BigOverSmall(tfb, Lum, tfbQ, LumQ)
ax2.plot(tfb_ratioQ, ratioQ, linewidth = 2, color = 'forestgreen', label = 'New vs Old Extr, Old AMR')
ax2.plot(tfb_ratioQ, ratioQ, linewidth = 2, color = 'yellowgreen', linestyle = (0, (5, 10)))
tfb_ratioAMR, ratioAMR = ratio_BigOverSmall(tfbQ, LumQ, tfbAMR, LumAMR)
ax2.plot(tfb_ratioAMR, ratioAMR, linewidth = 2, color = 'k', label = 'New vs Old AMR')
ax2.plot(tfb_ratioAMR, ratioAMR, linewidth = 2, color = 'forestgreen', linestyle = (0, (5, 10)))

tfb_ratioLQ, ratioLQ = ratio_BigOverSmall(tfbL, LumL, tfbLQ, LumLQ)
ax4.plot(tfb_ratioLQ, ratioLQ, linewidth = 2, color = 'goldenrod', label = 'New vs Old Extr, Old AMR')
ax4.plot(tfb_ratioLQ, ratioLQ, linewidth = 2, color = 'C1', linestyle = (0, (5, 10)))
tfb_ratioAMR_RC, ratioAMR_RC = ratio_BigOverSmall(tfbAMR, LumAMR, tfbAMR_RC, LumAMR_RC)
ax2.plot(tfb_ratioAMR_RC, ratioAMR_RC, linewidth = 2, color = 'k', label = 'New vs Old*+RC Extr, New AMR')
ax2.plot(tfb_ratioAMR_RC, ratioAMR_RC, linewidth = 2, color = 'royalblue', linestyle = (0, (5, 10)))
tfb_ratioL_AMR, ratioL_AMR = ratio_BigOverSmall(tfbL, LumL, tfbL_AMR, LumL_AMR)
ax4.plot(tfb_ratioL_AMR, ratioL_AMR, linewidth = 2, color = 'plum', label = 'New Extr, New vs Old AMR')
ax4.plot(tfb_ratioL_AMR, ratioL_AMR, linewidth = 2, color = 'C1', linestyle = (0, (5, 10)))
tfb_ratioL_AMRRC, ratioL_AMRRC = ratio_BigOverSmall(tfbL_AMR, LumL_AMR, tfbL_AMRRC, LumL_AMRRC)
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

ax2.set_ylabel(r'$\mathcal{R}$ Luminosity')
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

tfb_ratioRC, ratioRC = ratio_BigOverSmall(tfbL_AMRRC, LumL_AMRRC, tfbAMR_RC, LumAMR_RC)
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
# ax1.plot(tfbL, median_phL/apo, c = 'C1', label = 'Low old')
# ax1.plot(tfbLQ, median_phLQ/apo, c = 'goldenrod', label = 'Low NewExtr, OldAMR')
ax1.plot(tfb, median_ph/apo, c = 'yellowgreen', label = 'Fid old')
# ax1.plot(tfbL_AMR, median_phL_AMR/apo, c ='plum', label = r'Low Old$^*$Extr, NewAMR')
# ax1.plot(tfbL_AMRRC, median_phL_AMRRC/apo, c = 'maroon', label = r'Low Old$^*$Extr, NewAMR, RemoveCenter')
# ax1.plot(tfbQ, median_phQ/apo, c = 'forestgreen', label = 'Fid, NewExtr, OldAMR')
ax1.plot(tfbAMR, median_phAMR/apo, c = 'royalblue', label = 'NewAMR')
# ax1.plot(tfbH, median_phH/apo, c = 'darkviolet', label = 'High old')
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
ax1.axvline(0.5, c = 'k', linestyle = '--', alpha = .5)
ax1.axvline(0.6, c = 'k', linestyle = '--', alpha = .5)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/MazeOfRuns/Rph_OpAMR.png', bbox_inches='tight')

# %% Check photosphere at different times
snapsOld, _, tfbsOld = split_data_red('')
folderOld = f'{abspath}/data/opacity_tests/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/photo'

snapsNew, _, tfbsNew = split_data_red('NewAMR')
folderNew = f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}NewAMR/photo'

snapsLNew, _, tfbsLNew = split_data_red('LowResNewAMR')
folderLNew = f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowResNewAMR/photo'

# geometri things for plotting
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 
indecesorbital = np.concatenate(np.where(latitude_moll==0))
first_idx, last_idx = np.min(indecesorbital), np.max(indecesorbital)

for i, tfb in enumerate(tfbsNew):
    snapNew = snapsNew[i]
    if snapNew not in [115, 164, 175, 237, 280, 318]:
        continue
    print(snapNew)
    dataphNew = np.loadtxt(f'{folderNew}/NewAMR_photo{snapNew}.txt')
    xphNew, yphNew, zphNew, volphNew = dataphNew[0], dataphNew[1], dataphNew[2], dataphNew[3]
    # yz plane and midplane
    yz_phNew = np.abs(xphNew-Rt) < volphNew**(1/3)
    yph_yzNew, zph_yzNew = make_slices([yphNew, zphNew], yz_phNew)
    falselong_ph_yzNew = np.arctan2(zph_yzNew, yph_yzNew)
    sorted_indices_yzNew = np.argsort(falselong_ph_yzNew)
    yph_yzNew_sorted, zph_yzNew_sorted = yph_yzNew[sorted_indices_yzNew], zph_yzNew[sorted_indices_yzNew]
    xph_midNew, yph_midNew, zph_midNew = xphNew[indecesorbital], yphNew[indecesorbital], zphNew[indecesorbital]  

    snapOld = snapsOld[np.argmin(np.abs(tfbsOld - tfb))]
    dataph = np.loadtxt(f'{folderOld}/_photo{snapOld}.txt')
    xph, yph, zph, volph = dataph[0], dataph[1], dataph[2], dataph[3]
    yz_ph = np.abs(xph-Rt) < volph**(1/3)
    yph_yz, zph_yz = make_slices([yph, zph], yz_ph)
    falselong_ph_yz = np.arctan2(zph_yz, yph_yz) 
    sorted_indices_yz = np.argsort(falselong_ph_yz)  # Sorting by y-coordinate
    yph_yz_sorted, zph_yz_sorted = yph_yz[sorted_indices_yz], zph_yz[sorted_indices_yz]
    xph_mid, yph_mid, zph_mid = xph[indecesorbital], yph[indecesorbital], zph[indecesorbital]
    
    snapLNew = snapsLNew[np.argmin(np.abs(tfbsLNew - tfb))]
    dataphLNew = np.loadtxt(f'{folderLNew}/LowResNewAMR_photo{snapNew}.txt')
    xphLNew, yphLNew, zphLNew, volphLNew = dataphLNew[0], dataphLNew[1], dataphLNew[2], dataphLNew[3]
    yz_phLNew = np.abs(xphLNew-Rt) < volphLNew**(1/3)
    yph_yzLNew, zph_yzLNew = make_slices([yphLNew, zphLNew], yz_phLNew)
    falselong_ph_yzLNew = np.arctan2(zph_yzLNew, yph_yzLNew)
    sorted_indices_yzLNew = np.argsort(falselong_ph_yzLNew)
    yph_yzLNew_sorted = yph_yzLNew[sorted_indices_yzLNew]
    zph_yzLNew_sorted = zph_yzLNew[sorted_indices_yzLNew]
    xph_midLNew, yph_midLNew, zph_midLNew = xphLNew[indecesorbital], yphLNew[indecesorbital], zphLNew[indecesorbital]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    ax1.plot(xph_mid/apo, yph_mid/apo,  c = 'yellowgreen', markersize = 10, marker = 'H', label = 'Old')
    ax1.plot(xph_midNew/apo, yph_midNew/apo,  c = 'b', markersize = 10, marker = 'H', label = 'New AMR')
    ax1.plot(xphLNew[indecesorbital]/apo, yphLNew[indecesorbital]/apo,  c = 'C1', markersize = 10, marker = 'H', label = 'LowRes New AMR')
    # just to connect the first and last 
    ax1.plot([xph[first_idx]/apo, xph[last_idx]/apo], [yph[first_idx]/apo, yph[last_idx]/apo], c = 'yellowgreen', markersize = 10, marker = 'H')
    ax1.plot([xphNew[first_idx]/apo, xphNew[last_idx]/apo], [yphNew[first_idx]/apo, yphNew[last_idx]/apo], c = 'b', markersize = 10, marker = 'H')
    ax1.plot([xphLNew[first_idx]/apo, xphLNew[last_idx]/apo], [yphLNew[first_idx]/apo, yphLNew[last_idx]/apo], c = 'C1', markersize = 10, marker = 'H')
    # ax1.quiver(xph_mid/apo, yph_mid/apo, Vxph[indecesorbital]/40, Vyph[indecesorbital]/40, angles='xy', scale_units='xy', scale=0.7, color="k", width=0.003, headwidth = 6)
    # ax1.text(-2.6, -2.8, r'z = 0', fontsize = 25)
    # ax1.text(-2.6, 1.65, f't = {np.round(tfb[i],2)}' + r' t$_{\rm fb}$', color = 'k', fontsize = 26)
    ax1.set_xlabel(r'X [$R_{\rm a}$]', fontsize = 25)
    ax1.set_ylabel(r'Y [$R_{\rm a}$]', fontsize = 25)
    ax1.set_xlim(-4, 2)
    ax1.set_ylim(-3, 2)
    
    ax2.plot(yph_yz_sorted/apo, zph_yz_sorted/apo,  c = 'yellowgreen', markersize = 10, marker = 'H', label = 'Old')
    ax2.plot(yph_yzNew_sorted/apo, zph_yzNew_sorted/apo,  c = 'b', markersize = 10, marker = 'H', label = 'New AMR')
    ax2.plot(yph_yzLNew_sorted/apo, zph_yzLNew_sorted/apo,  c = 'C1', markersize = 10, marker = 'H', label = 'LowRes New AMR')
    # # if yph_yz_sorted is not empty, connect the last and first point
    if len(yph_yz_sorted) > 0:
        ax2.plot([yph_yz_sorted[-1]/apo, yph_yz_sorted[0]/apo], [zph_yz_sorted[-1]/apo, zph_yz_sorted[0]/apo], c = 'k', alpha = 0.5)
    if len(yph_yzNew_sorted) > 0:
        ax2.plot([yph_yzNew_sorted[-1]/apo, yph_yzNew_sorted[0]/apo], [zph_yzNew_sorted[-1]/apo, zph_yzNew_sorted[0]/apo], c = 'C1', alpha = 0.5)
    if len(yph_yzLNew_sorted) > 0:
        ax2.plot([yph_yzLNew_sorted[-1]/apo, yph_yzLNew_sorted[0]/apo], [zph_yzLNew_sorted[-1]/apo, zph_yzLNew_sorted[0]/apo], c = 'b', alpha = 0.5)
    ax2.set_xlabel(r'Y [$R_{\rm a}$]', fontsize = 25)
    ax2.set_ylabel(r'Z [$R_{\rm a}$]', fontsize = 25)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-3.5, 3.5)
    plt.suptitle(f'Photosphere at t = {np.round(tfb,2)} t$_{{fb}}$', fontsize = 30)

    for ax in [ax1, ax2]:
        ax.scatter(0,0, c= 'k', marker = 'x', s=80)
        ax.legend(fontsize = 20)

# %%
