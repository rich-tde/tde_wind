""" Total and regional angle-integrated mass fallback, temperature, luminosity and trapping radius."""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
from src import orbits as orb
import Utilities.prelude as prel
import healpy as hp
from scipy.stats import gmean
from Utilities.operators import sort_list, choose_observers    
from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall


# first_eq = 88 # observer eq. plane
# final_eq = 103+1 #observer eq. plane
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
compton = 'Compton'
which_obs = 'dark_bright_z'
checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']
checkslegend = ['Low', 'Middle', 'High']
colors_res = ['C1', 'yellowgreen', 'darkviolet']

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
tfallback = things['t_fb_days']
t_fb_days_cgs = tfallback * 24 * 3600 # in seconds
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 

#%% Pick observers
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_axis, label_axis, colors_axis, lines_axis = choose_observers(observers_xyz, which_obs)
observers_xyz = observers_xyz.T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]

Rph_all = []
Lph_all = []
approxL_ph_all = []
Temp_ph_all = []
Rtr_all = []
Mdot_all = []
Temp_tr_all = []
tfbs_all = []
figTr, axTr = plt.subplots(1, 1, figsize=(10, 6))
figL, (axL, axTph, axC2) = plt.subplots(1, 3, figsize=(27, 6))
figMdot, (axMdot, axTtr, axC) = plt.subplots(1, 3, figsize=(27, 6))
for c, check in enumerate(checks):
        print(check)
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
        snaps, tfbs, Lums = data[:, 0], data[:, 1], data[:, 2]
        tfbs, snaps, Lums = sort_list([tfbs, snaps, Lums], snaps, unique=True)
        snaps = snaps.astype(int)
        Lph_all.append(Lums)
        idx_maxLum = np.argmax(Lums)
        tfbs_all.append(tfbs)

        data_wind = np.loadtxt(f'{abspath}/data/{folder}/wind/Mdot_{check}.csv',  
                        delimiter = ',', 
                        skiprows=1,  
                        unpack=True)
        tfb_fall, mfall = data_wind[1], data_wind[2]

        r_tr_mean = np.zeros((len(indices_axis), len(snaps)))
        Mdot_mean = np.zeros((len(indices_axis), len(snaps)))
        den_tr_mean = np.zeros((len(indices_axis), len(snaps)))
        Vr_tr_mean = np.zeros((len(indices_axis), len(snaps)))  
        Temp_tr_mean = np.zeros((len(indices_axis), len(snaps)))    
        eta_axis = np.zeros((len(indices_axis), len(snaps)))
        r_ph_mean = np.zeros((len(indices_axis), len(snaps)))
        Lum_ph_mean = np.zeros((len(indices_axis), len(snaps)))
        Temp_ph_mean = np.zeros((len(indices_axis), len(snaps)))
        # (all) angle-integrated quantities
        r_ph_snap = np.zeros(len(snaps))
        Temp_ph_snap = np.zeros(len(snaps))
        approxL_ph_snap = np.zeros(len(snaps))
        r_tr_snap = np.zeros(len(snaps))
        Temp_tr_snap = np.zeros(len(snaps))
        Mdot_snap = np.zeros(len(snaps))
        for s, snap in enumerate(snaps): 
                tfb = tfbs[s]
                dataph = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
                xph, yph, zph, Temp_ph, RadDen_ph, Lum_ph = dataph[0], dataph[1], dataph[2], dataph[5], dataph[6], dataph[-2]
                r_ph = np.sqrt(xph**2 + yph**2 + zph**2)
                Temprad_ph = (RadDen_ph*prel.en_den_converter/prel.alpha_cgs)**(1/4)  
                r_ph_snap[s] = np.mean(r_ph)
                Temp_ph_snap[s] = np.mean(Temprad_ph) #np.mean(Temp_ph) #
                approxL_ph_snap[s] = np.mean(r_ph*prel.Rsol_cgs * Temprad_ph**2)**2
                dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz") 
                x_tr, y_tr, z_tr, den_tr, Vr_tr, Temp_tr, Rad_den_tr = \
                        dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['Vr_tr'], dataRtr['Temp_tr'], dataRtr['Rad_den_tr'] 
                r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
                Temprad_tr = (Rad_den_tr*prel.en_den_converter/prel.alpha_cgs)**(1/4)  
                r_tr_snap[s] = np.mean(r_tr)
                Temp_tr_snap[s] = np.mean(Temprad_tr) #np.mean(Temp_tr)
                Mdot_tr =  r_tr**2 * den_tr * np.abs(Vr_tr)
                Mdot_snap[s] = 4 * np.pi/192 * np.sum(Mdot_tr) #np.mean(Mdot_tr)
                for i, observer in enumerate(indices_axis):
                        Lum_ph_mean[i][s] = np.mean(Lum_ph[observer])   
                        Temp_ph_mean[i][s] = np.mean(Temprad_ph[observer])                    
                        # nonzero = r_tr[observer] != 0 
                        # if np.logical_and(check == 'HiResNewAMR', np.round(tfb,2)>1):
                        #         print(f'{np.round(tfb,2)}, {label_axis[i]}: percentage non zero Rtr/total = {int(len(nonzero[nonzero==True])/len(nonzero)*100)}%')
                        r_tr_mean[i][s] = np.mean(r_tr[observer])
                        r_ph_mean[i][s] = np.mean(r_ph[observer])
                        Vr_tr_mean[i][s] = np.mean(Vr_tr[observer])
                        den_tr_mean[i][s] = np.mean(den_tr[observer])
                        Temp_tr_mean[i][s] = np.mean(Temprad_tr[observer])
                        Mdot_mean[i][s] = np.mean(Mdot_tr[observer])

                        t_dyn = (r_tr_mean[i][s]/Vr_tr_mean[i][s])*prel.tsol_cgs/t_fb_days_cgs # you want it in t_fb
                        tfb_adjusted = tfb - t_dyn
                        find_time = np.argmin(np.abs(tfb_fall-tfb_adjusted))
                        eta_axis[i][s] = np.abs(Mdot_mean[i][s]/mfall[find_time])
        Temp_ph_all.append(Temp_ph_snap)
        Rtr_all.append(r_tr_snap)
        Temp_tr_all.append(Temp_tr_snap)
        Mdot_all.append(Mdot_snap)
        Rph_all.append(r_ph_snap)
        approxL_ph_all.append(approxL_ph_snap)

        for i, observer in enumerate(indices_axis):
                axTr.plot(tfbs, r_tr_mean[i]/Rt, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                # axTr.plot(tfbs, r_ph_mean[i]/Rt, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                axTtr.plot(tfbs, Temp_tr_mean[i]/Rp, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                # axTr.scatter(tfbs[idx_maxLum], r_tr_mean[i][idx_maxLum]/Rt, c = colors_res[c], s = 85, marker = 'X')
                axL.plot(tfbs, Lum_ph_mean[i]/Ledd_cgs, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                axTph.plot(tfbs, Temp_ph_mean[i]/Rp, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                # axL.scatter(tfbs[idx_maxLum], Lum_obs_time[i][idx_maxLum]/Ledd_cgs, c = colors_res[c], s = 85, marker = 'X')
                axMdot.plot(tfbs, Mdot_mean[i]/Medd_sol, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                # axMdot.scatter(tfbs[idx_maxLum], Mdot_mean[i][idx_maxLum]/Medd_sol, c = colors_res[c], s = 85, marker = 'X')
                axC.plot(tfbs, (r_tr_mean[i]*prel.Rsol_cgs*Temp_tr_mean[i]**2)**2/Ledd_cgs, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')
                axC2.plot(tfbs, (r_ph_mean[i]*prel.Rsol_cgs*Temp_ph_mean[i]**2)**2/Ledd_cgs, c = colors_res[c], ls = lines_axis[i], label = label_axis[i] if c == 0 else '')

axTr.axhline(amin/Rt, c = 'k', ls = '--', label = r'$a_{\rm mb}$')        
axTr.set_ylabel(r'$r_{\rm tr} [r_{\rm t}]$')
axL.set_ylabel(r'$L_{\rm ph} [L_{\rm Edd}]$')
axC.set_ylabel(r'$r_{\rm tr}^2 T_{\rm tr}^4 [L_{\rm Edd}]$')
axC2.set_ylabel(r'$r_{\rm ph}^2 T_{\rm ph}^4 [L_{\rm Edd}]$')
axMdot.set_ylabel(r'$\dot{M}_{\rm w} (R_{\rm tr}) [\dot{M}_{\rm Edd}]$')
axTtr.set_ylabel(r'$T_{\rm tr} [K]$')
axTph.set_ylabel(r'$T_{\rm ph} [K]$')
original_ticks = axTr.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [axTr, axL, axMdot, axTtr, axTph, axC, axC2]:
    ax.set_xlabel(r't [$t_{\rm fb}$]')
    ax.set_xticks(new_ticks)
    ax.tick_params(axis='both', which='major', width=1.2, length=9, color = 'k')
    ax.tick_params(axis='both', which='minor', width=1, length=7, color = 'k')
    ax.grid()
    ax.set_xlim(0, 1.8) 
    ax.set_yscale('log')
    ax.legend(fontsize = 16)
#     ax.set_title(r'Mean on observers (NON discarding the one with $r_{\rm tr}=0$)')

axTr.set_ylim(1, 100)
axTtr.set_ylim(9e2, 5e4)
axL.set_ylim(1e-1, 11)
axC.set_ylim(1, 1e5)
axC2.set_ylim(1, 1e5)
axTph.set_ylim(9e2, 5e4)
axMdot.set_ylim(1e3, 1e6)
figTr.savefig(f'{abspath}/Figs/next_meeting/Rtr_res.png')
figL.savefig(f'{abspath}/Figs/next_meeting/L_res.png')
figMdot.savefig(f'{abspath}/Figs/next_meeting/Mdot_res.png')


#  compare with other resolutions
# pvalue 
# statL = np.zeros(len(tfbL)) 
# pvalueL = np.zeros(len(tfbL))
# for i, snapi in enumerate(snapsL):
#         # LowRes data
#         photo = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/photo/LowResNewAMR_photo{snapi}.txt')
#         xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
#         rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
#         if rph_i.any() < R0:
#                 print('Less than R0:', rph_i[rph_i<R0])
#         # ksL = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
#         # statL[i], pvalueL[i] = ksL.statistic, ksL.pvalue


#%% TOTAL
figRTr_total, axRTr_total = plt.subplots(1, 1, figsize=(10, 6))
figL_total, (axL_total, axTph_total, axapproxL_total) = plt.subplots(1, 3, figsize=(30, 6))
figTtr_total, axTtr_total = plt.subplots(1, 1, figsize=(10, 6))
figMdot_total, (axMdot_total, axerr) = plt.subplots(2, 1, figsize = (9, 9), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)

tfb_ratioL, ratioL, rel_errL  = ratio_BigOverSmall(tfbs_all[0], Mdot_all[0], tfbs_all[1], Mdot_all[1])
tfb_ratioH, ratioH, rel_errH  = ratio_BigOverSmall(tfbs_all[2], Mdot_all[2], tfbs_all[1], Mdot_all[1])

for i, check in (enumerate(checks)):
        # if i != 2:
        #        continue
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        axRTr_total.plot(tfbs_all[i], Rtr_all[i]/Rp, c = colors_res[i], label = checkslegend[i])
        axL_total.plot(tfbs_all[i], Lph_all[i]/Ledd_cgs, c = colors_res[i], label = checkslegend[i])
        axapproxL_total.plot(tfbs_all[i], approxL_ph_all[i]/Ledd_cgs, c = colors_res[i], label = checkslegend[i])
        axTph_total.plot(tfbs_all[i], Temp_ph_all[i]/Rp, c = colors_res[i], label = checkslegend[i])
        axMdot_total.plot(tfbs_all[i], Mdot_all[i]/Medd_sol, c = colors_res[i], label = checkslegend[i])
        if check == 'HiResNewAMR':
                axMdot_total.plot(tfbs_all[i], Rtr_all[i]/Rg, c = colors_res[i], ls = '--', label = r'$r_{\rm tr}/r_{\rm g}$ HiRes')
        axTtr_total.plot(tfbs_all[i], Temp_tr_all[i]/Rp, c = colors_res[i], label = checkslegend[i]) 

axerr.plot(tfb_ratioL, ratioL, c = 'C1')
axerr.plot(tfb_ratioL, ratioL, c = 'yellowgreen', ls = (0, (5, 10)))
axerr.plot(tfb_ratioH, ratioH, c = 'darkviolet')
axerr.plot(tfb_ratioH, ratioH, c = 'yellowgreen', ls = (0, (5, 10)))

axRTr_total.set_ylabel(r'$r_{\rm tr} [r_{\rm t}]$')
axL_total.set_ylabel(r'$L_{\rm ph} [L_{\rm Edd}]$')
axapproxL_total.set_ylabel(r'$r_{\rm ph}^2 T_{\rm ph}^4 [L_{\rm Edd}]$')
axTph_total.set_ylabel(r'$T_{\rm ph} [K]$')
axMdot_total.set_ylabel(r'$\dot{M}_{\rm w}(r_{\rm tr}) [\dot{M}_{\rm Edd}]$')
axTtr_total.set_ylabel(r'$T_{\rm tr} [K]$')
axerr.set_ylabel(r'$\mathcal{R}$')
original_ticks = axL_total.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
for ax in [axRTr_total, axL_total, axMdot_total, axTtr_total, axTph_total, axapproxL_total, axerr]:
    ax.set_xlabel(r't [$t_{\rm fb}$]')
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width=1.2, length=9, color = 'k')
    ax.tick_params(axis='both', which='minor', width=1, length=7, color = 'k')
    ax.grid()
    ax.set_xlim(0, 2.5) 
    ax.legend(fontsize = 16)
    if ax != axerr:
        ax.set_yscale('log')

axL_total.set_ylim(1e-1, 5)
axapproxL_total.set_ylim(1e-1, 1e4)
axMdot_total.set_ylim(1e3, 1e6)
axRTr_total.set_ylim(1, 80) 
axTtr_total.set_ylim(9e2, 5e4)
axTph_total.set_ylim(9e2, 5e4)
axerr.set_ylim(.9, 4)
figRTr_total.savefig(f'{abspath}/Figs/next_meeting/Rtr_total.png')
figL_total.savefig(f'{abspath}/Figs/next_meeting/L_total.png')
figMdot_total.savefig(f'{abspath}/Figs/next_meeting/Mdot_total.png')




# %%
