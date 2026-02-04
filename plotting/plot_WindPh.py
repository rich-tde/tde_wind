""" Total and regional angle-integrated radial velocity, density, temperature, mass fallback, luminosity and trapping radius
for wind photospheric cells."""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src import orbits as orb
import Utilities.prelude as prel
import healpy as hp
from Utilities.operators import sort_list, choose_observers, to_spherical_components  

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
compton = 'Compton'
which_obs = 'left_right_in_out_z' #'left_right_z' #'arch', 'quadrants', 'ax is'
check = 'HiResNewAMR' 

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
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfbs, Lums = data[:, 0], data[:, 1], data[:, 2]
tfbs, snaps, Lums = sort_list([tfbs, snaps, Lums], snaps, unique=True)
snaps = snaps.astype(int)
idx_maxLum = np.argmax(Lums)
# dataDiss = data = np.load(f'{abspath}/data/{folder}/wind/{check}_RdissSec.npz', allow_pickle=True)
# diss_list = data['diss_list'].item()

def mean_nonzero(arr, axis=1):
    count = np.count_nonzero(arr, axis=axis)
    return np.divide(
        np.sum(arr, axis=axis),
        count,
        out=np.zeros_like(count, dtype=float),
        where=count != 0
    )

# Pick observers
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_axis, label_axis, colors_axis, lines_axis = choose_observers(observers_xyz, which_obs)
print([len(indices_axis[i]) for i in range(len(indices_axis))], flush=True)
observers_xyz = observers_xyz.T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]

Lum_ph_allSum = np.zeros(len(snaps))
ratio_Rtr = np.zeros((len(indices_axis), len(snaps)))
r_tr_sec = np.zeros((len(indices_axis), len(snaps)))
r_tr_perc16sec = np.zeros((len(indices_axis), len(snaps)))
r_tr_perc84sec = np.zeros((len(indices_axis), len(snaps)))
r_trnonzero_sec = np.zeros((len(indices_axis), len(snaps)))
r_ph_sec = np.zeros((len(indices_axis), len(snaps)))
r_ph_perc16sec = np.zeros((len(indices_axis), len(snaps)))
r_ph_perc84sec = np.zeros((len(indices_axis), len(snaps)))
r_phnonzero_sec = np.zeros((len(indices_axis), len(snaps)))
Mdot_sec = np.zeros((len(indices_axis), len(snaps)))
Vr_tr_sec = np.zeros((len(indices_axis), len(snaps)))  
Vr_ph_sec = np.zeros((len(indices_axis), len(snaps)))  
den_tr_sec = np.zeros((len(indices_axis), len(snaps)))
den_ph_sec = np.zeros((len(indices_axis), len(snaps)))
Temp_tr_sec = np.zeros((len(indices_axis), len(snaps))) 
Temp_ph_sec = np.zeros((len(indices_axis), len(snaps)))   
TempGas_ph_sec = np.zeros((len(indices_axis), len(snaps))) 
Lum_allph_secSum = np.zeros((len(indices_axis), len(snaps)))
Lum_allph_secmean = np.zeros((len(indices_axis), len(snaps)))
Lum_adv_tr_sec = np.zeros((len(indices_axis), len(snaps)))
# Lum_adv_ph_sec = np.zeros((len(indices_axis), len(snaps)))
L_diss_sec = np.zeros((len(indices_axis), len(snaps)))

# fig, ax = plt.subplots(1, 3, figsize=(18,6))
for s, snap in enumerate(snaps): 
        # you are considering all the photosphere, not only the unbound
        xph, yph, zph, vol_ph, den_ph, Temp_ph, RadDen_ph, vx_ph, vy_ph, vz_ph, Press_ph, IE_den_ph, _, _, Lum_ph, _ = \
                np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
        den_ph /=prel.den_converter # it was saved in cgs
        r_ph = np.sqrt(xph**2 + yph**2 + zph**2)
        vel_ph = np.sqrt(vx_ph**2 + vy_ph**2 + vz_ph**2)
        mass_ph = den_ph * vol_ph
        Lum_ph_allSum[s] = np.sum(Lum_ph) # CGS
        Vr_ph, _, _ = to_spherical_components(vx_ph, vy_ph, vz_ph, xph, yph, zph)
        bern_ph = orb.bern_coeff(r_ph, vel_ph, den_ph, mass_ph, Press_ph, IE_den_ph, RadDen_ph, params)

        Temprad_ph = (RadDen_ph*prel.en_den_converter/prel.alpha_cgs)**(1/4)  
        # Lum_adv_ph = 4 * np.pi * r_ph**2 * Vr_ph * RadDen_ph # advective luminosity

        dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz") # NB it is selected to be only done by wind cells
        x_tr, y_tr, z_tr, den_tr, Vr_tr, Temp_tr, Rad_den_tr, vol_tr = \
                dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['Vr_tr'], dataRtr['Temp_tr'], dataRtr['Rad_den_tr'], dataRtr['vol_tr']
        mass_tr = den_tr * vol_tr
        dim_tr = (vol_tr)**(1/3)
        r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        Temprad_tr = (Rad_den_tr*prel.en_den_converter/prel.alpha_cgs)**(1/4)  
        Mdot_tr =  np.pi * dim_tr**2 * den_tr * Vr_tr 
        Lum_adv_tr = 4 * np.pi * r_tr**2 * Vr_tr * Rad_den_tr # advective luminosity
        for i, observer in enumerate(indices_axis):
                lab = label_axis[i]
                # L_diss_sec[i][s] = dataDiss[s][f'Ldisstot_pos {lab}']
                
                exist_rtr = np.logical_and(r_tr[observer] > 1.5*Rt, y_tr[observer]*y[observer]>0) # second condition to have only the observers in the wanted region
                obs_tr = observer[exist_rtr]
                ratio_Rtr[i][s] = len(r_tr[obs_tr]) / len(r_tr[observer])
                # photo_wind = np.logical_and(bern_ph[observer]>0, Vr_ph[observer]>0)
                # obs_wind_Rtr = observer[np.logical_and(exist_rtr, photo_wind)]
                Vr_ph_sec[i][s] = np.sum(Vr_ph[observer] * mass_ph[observer]) / np.sum(mass_ph[observer])
                den_ph_sec[i][s] = np.sum(den_ph[observer] * mass_ph[observer]) / np.sum(mass_ph[observer])
                Temp_ph_sec[i][s] = np.sum(Temprad_ph[observer] * vol_ph[observer]) / np.sum(vol_ph[observer])   
                TempGas_ph_sec[i][s] = np.sum(Temp_ph[observer] * vol_ph[observer]) / np.sum(vol_ph[observer])           
                r_ph_sec[i][s] = np.median(r_ph[observer])  
                r_ph_perc16sec[i][s] = np.percentile(r_ph[observer], 16)
                r_ph_perc84sec[i][s] = np.percentile(r_ph[observer], 84)
                r_phnonzero_sec[i][s] = np.median(r_ph[obs_tr])
                Lum_allph_secSum[i][s] = np.sum(Lum_ph[observer]) # CGS
                Lum_allph_secmean[i][s] = np.mean(Lum_ph[observer]) # CGS
                
                Vr_tr_sec[i][s] = np.sum(Vr_tr[observer] * mass_tr[observer]) / np.sum(mass_tr[observer])
                den_tr_sec[i][s] = np.sum(den_tr[observer] * mass_tr[observer]) / np.sum(mass_tr[observer])
                Temp_tr_sec[i][s] = np.sum(Temprad_tr[observer] * vol_tr[observer]) / np.sum(vol_tr[observer])                 
                Lum_adv_tr_sec[i][s] = np.sum(Lum_adv_tr[obs_tr]) * prel.en_converter/prel.tsol_cgs

                r_tr_sec[i][s] = np.median(r_tr[observer])  
                r_tr_perc16sec[i][s] = np.percentile(r_tr[observer], 16)
                r_tr_perc84sec[i][s] = np.percentile(r_tr[observer], 84)
                r_trnonzero_sec[i][s] = np.median(r_tr[obs_tr]) 
                Temp_tr_sec[i][s] = np.sum(Temprad_tr[obs_tr] * vol_tr[obs_tr]) / np.sum(vol_tr[obs_tr])
                # Mdot_sec[i][s] = (np.mean(r_tr[obs_tr]))**2 /np.sum(dim_tr[obs_tr]**2) * np.sum(Mdot_tr[obs_tr])
                Mdot_sec[i][s] = np.pi * (np.median(r_tr[obs_tr]**2 * den_tr[obs_tr] * Vr_tr[obs_tr]))


# ax[0].set_ylabel(r'z [r$_{\rm t}$]')
# Plot
figTr, axTr = plt.subplots(1, 1, figsize=(10, 8))
figratios, (axTrnonzero, axNtr, axratio) = plt.subplots(1, 3, figsize=(27, 6))
fig, (axVph, axdph, axTph) = plt.subplots(1, 3, figsize=(26, 6))
figL, (axL, axLmean) = plt.subplots(2, 1, figsize=(9, 13))
figM, (axMdotobs, axLtrph) = plt.subplots(1, 2, figsize=(18, 6))

for i, observer in enumerate(indices_axis):
        if label_axis[i] == 'south pole':
               continue
        axTr.plot(tfbs, r_tr_sec[i]/Rt, c = colors_axis[i], ls = ':')
        # axTr.fill_between(tfbs, r_tr_perc16sec[i]/Rt, r_tr_perc84sec[i]/Rt, color=colors_axis[i], alpha=0.2)
        axTr.plot(tfbs, r_ph_sec[i]/Rt, c = colors_axis[i], label = label_axis[i])
        # axTr.fill_between(tfbs, r_ph_perc16sec[i]/Rt, r_ph_perc84sec[i]/Rt, color=colors_axis[i], alpha=0.3)
        
        axTrnonzero.plot(tfbs, r_trnonzero_sec[i]/Rt, c = colors_axis[i]) #, label = r'$r_{\rm tr}$' if i == 0 else '')
        # axTrnonzero.plot(tfbs, r_phnonzero_sec[i]/Rt, c = colors_axis[i], label = r'$r_{\rm ph}$' if i == 0 else '')
        
        axNtr.plot(tfbs, ratio_Rtr[i], c = colors_axis[i], label = label_axis[i])
        axratio.plot(tfbs, r_phnonzero_sec[i]/r_trnonzero_sec[i], c = colors_axis[i]) #, label = label_axis[i])
        
        axVph.plot(tfbs, Vr_ph_sec[i] * conversion_sol_kms, c = colors_axis[i], label = label_axis[i])
        # axVph.plot(tfbs, Vr_tr_sec[i] * conversion_sol_kms, c = colors_axis[i],  label = label_axis[i])
        axdph.plot(tfbs, den_ph_sec[i] * prel.den_converter, c = colors_axis[i])
        # axdph.plot(tfbs, den_tr_sec[i] * prel.den_converter, c = colors_axis[i])
        axTph.plot(tfbs, Temp_ph_sec[i]/Rp, c = colors_axis[i])
        # axTph.plot(tfbs, TempGas_ph_sec[i]/Rp, c = colors_axis[i], ls = '--')
        # axTph.plot(tfbs, Temp_tr_sec[i]/Rp, c = colors_axis[i])
        axL.plot(tfbs, Lum_allph_secSum[i]/Lum_ph_allSum, c = colors_axis[i], label = label_axis[i])#,   label =r'$L_{\rm FLD} (r_{\rm ph, all})$' if i ==0 else '')
        axLmean.plot(tfbs, Lum_allph_secmean[i], c = colors_axis[i])
        
        axMdotobs.plot(tfbs, Mdot_sec[i]/Medd_sol, c = colors_axis[i], label = label_axis[i])
        if i == 1:
               axMdotobs.plot(tfbs, r_trnonzero_sec[i]/Rg, c = colors_axis[i], ls = '--', label = r'$r_{\rm tr}/r_{\rm g}$' if i ==0 else '')
        axLtrph.plot(tfbs, Lum_allph_secSum[i]/Lum_ph_allSum, c = colors_axis[i], label =r'$L_{\rm FLD} (r_{\rm ph})$' if i == 0 else '')
        axLtrph.plot(tfbs, Lum_adv_tr_sec[i]/Lum_ph_allSum, c = colors_axis[i], ls = ':', label =r'$L_{\rm adv} (r_{\rm tr})$' if i == 0 else '')

axLmean.plot(tfbs, Lums, c = 'k', ls = '--', label = r'Total')       
axTr.set_ylabel(r'median $r_{\rm obs} [r_{\rm t}]$')
axTrnonzero.set_ylabel(r'median nonzero $r_{\rm tr} [r_{\rm t}]$')
axNtr.set_ylabel(r'Fraction of obs with adv region')
axratio.set_ylabel(r'$r_{\rm ph}/r_{\rm tr}$ in adv. region')
axVph.set_ylabel(r'v$_{\rm ph}$ [km/s]')
axdph.set_ylabel(r'$\rho_{\rm ph}$ [g/cm$^3]$')    
axTph.set_ylabel(r'$T_{\rm rad, ph} [K]$')
axL.set_ylabel(r'$\sum_{i \in \mathcal{I}} L_{{\rm ph}, i}/\sum_{\rm i=0}^{N_{\rm obs}} L_{{\rm ph}, i}$')
axMdotobs.set_ylabel(r'$\dot{M}_{\rm w} (r_{\rm tr}) [\dot{M}_{\rm Edd}]$')
axLtrph.set_ylabel(r'$\sum_{i \in \mathcal{I}} L_i/\sum_{\rm i=0}^{N_{\rm obs}} L_{{\rm ph}, i}$')
axLmean.set_ylabel(r'mean $L_{\rm ph}$')
original_ticks = axTr.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
new_labels = [f'{tick:.2f}' if tick in original_ticks else '' for tick in new_ticks]
for ax in [axTrnonzero, axTr, axratio, axNtr, axVph, axdph, axL, axLmean, axMdotobs, axLtrph, axTph]:
        if ax != axL:
                ax.set_xlabel(r't [$t_{\rm fb}$]')
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(new_labels)
        ax.tick_params(axis='both', which='major', width=1.2, length=9, color = 'k')
        ax.tick_params(axis='both', which='minor', width=1, length=7, color = 'k')
        ax.grid()
        ax.set_xlim(0.05, 2.25) 
        if ax != axratio:
                ax.set_yscale('log')
        ax.legend(fontsize = 20)
        #     ax.set_title(r'Mean on observers (NON discarding the one with $r_{\rm tr}=0$)')

axTr.set_ylim(1, 100)
axratio.set_ylim(1, 10)
axdph.set_ylim(1e-14, 1e-10)
axVph.set_ylim(1e3, 2e4)
axTph.set_ylim(2e2, 5e4)
axL.set_ylim(1e-2, 2)
axLmean.set_ylim(1e38, 5e43)
axMdotobs.set_ylim(7e2, 5e5)
axLtrph.set_ylim(1e-2, 2)
figTr.suptitle(r'Dotted line: $r_{\rm tr}$, solid line: $r_{\rm ph}$', fontsize=22)
figTr.tight_layout()
fig.tight_layout()


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



# %%
