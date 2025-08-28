""" Plot the photosphere."""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src import orbits as orb
import matplotlib.colors as colors
from matplotlib import cm
import Utilities.prelude as prel
import healpy as hp
from matplotlib import gridspec
from scipy.stats import gmean, ks_2samp
from Utilities.basic_units import radians
from Utilities.operators import find_ratio, sort_list
from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall
from src.wind_profile import choose_observers
matplotlib.rcParams['figure.dpi'] = 150

#%%
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
check ='NewAMR'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']

#%%
# tvisc = (R0**3/(prel.G*Mbh))**(1/2) * (1/0.1) * (1/0.3)**2
# tvisc_days = tvisc*prel.tsol_cgs/(3660*24)
# print(tvisc_days)

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_axis, label_axis, colors_axis = choose_observers(observers_xyz, 'focus_axis')
observers_xyz = observers_xyz.T
# HEALPIX
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
# r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
# theta = np.arctan2(y, x)          # Azimuthal angle in radians
# phi = np.arccos(z / r)            # Elevation angle in radians
# # Convert to latitude and longitude
# longitude_moll = theta              
# latitude_moll = np.pi / 2 - phi 

# indecesorbital = np.concatenate(np.where(latitude_moll==0))
# long_orb, lat_orb = longitude_moll[indecesorbital], latitude_moll[indecesorbital]

# print('Longitude from HELAPIX min and max: ' , np.min(theta), np.max(theta))
# print('Longitude for mollweide min and max: ', np.min(longitude_moll), np.max(longitude_moll))
# print('Latitude from HELAPIX min and max: ' , np.min(phi), np.max(phi))
# print('Latitude for mollweide min and max: ', np.min(latitude_moll), np.max(latitude_moll))
# print('So we shift of pi/2 the latitude')
# # Plot in 2D using a Mollweide projection
# fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
# img = ax.scatter(longitude_moll, latitude_moll, s=20, c=np.arange(len(longitude_moll)), cmap='viridis')
# # ax.scatter(longitude_moll[first_eq:final_eq], latitude_moll[first_eq:final_eq], s=10, c='r')
# ax.scatter(long_orb, lat_orb, s=10, c='r')
# plt.colorbar(img, ax=ax, label='Observer Number')
# ax.set_title("Observers on the Sphere (Mollweide Projection)")
# ax.grid(True)
# ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
# ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
# plt.tight_layout()
# plt.show()

#%% NB DATA ARE NOT SORTED
ph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}_phidx_fluxes.txt')
snaps, tfb, allindices_ph = ph_data[:, 0].astype(int), ph_data[:, 1], ph_data[:, 2:]
flux_test = np.zeros(len(snaps))
fluxes = []
for i, snap in enumerate(snaps):
        selected_lines = np.concatenate(np.where(snaps == snap))
        # eliminate the even rows (photosphere indices) of allindices_ph
        _, selected_fluxes = selected_lines[0], selected_lines[1]
        fluxes.append(allindices_ph[selected_fluxes])
        flux_test[i] = np.sum(allindices_ph[selected_fluxes])
fluxes = np.array(fluxes)
tfb, fluxes, snaps = sort_list([tfb, fluxes, snaps], snaps, unique=True)
mean_rph = np.zeros(len(tfb))
mean_rph_weig = np.zeros(len(tfb))
gmean_ph = np.zeros(len(tfb))
median_ph = np.zeros(len(tfb))
size_at_median = np.zeros(len(tfb))
mean_size = np.zeros(len(tfb))
percentile16 = np.zeros(len(tfb))
percentile84 = np.zeros(len(tfb))

for i, snapi in enumerate(snaps):
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        dim_i = vol_i**(1/3)
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_rph[i] = np.mean(rph_i)
        mean_rph_weig[i] = np.sum(rph_i*fluxes[i])/np.sum(fluxes[i])
        gmean_ph[i] = gmean(rph_i)
        median_ph[i] = np.median(rph_i)
        idx_median = np.argmin(np.abs(rph_i - median_ph[i]))
        size_at_median[i] = dim_i[idx_median]
        mean_size[i] = np.mean(dim_i)
        percentile16[i] = np.percentile(rph_i, 16)
        percentile84[i] = np.percentile(rph_i, 84)

rph_obs_time = np.zeros((len(tfb), len(indices_axis)))
den_obs_time = np.zeros((len(tfb), len(indices_axis)))
T_obs_time = np.zeros((len(tfb), len(indices_axis)))
Lum_obs_time = np.zeros((len(tfb), len(indices_axis)))
rtr_obs_time = np.zeros((len(tfb), len(indices_axis)))
# alpha_obs_time = np.zeros((len(tfb), len(indices_axis)))
flux_obs_time = np.zeros((len(tfb), len(indices_axis)))
for i, snapi in enumerate(snaps):
        xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph, Pressph, IE_denph, alphaph, _, Lumph, _ = \
                np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snapi}.txt')
        dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snapi}.npz")
        fluxph = fluxes[i]
        dim = volph**(1/3)
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        x_tr, y_tr, z_tr = \
            dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr']
        rtr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        for j, observer in enumerate(indices_axis):
                rph_obs_time[i, j] = np.mean(rph[observer])    
                rtr_obs_time[i, j] = np.mean(rtr[observer])   
                den_obs_time[i, j] = np.mean(denph[observer])
                T_obs_time[i, j] = np.mean(Tempph[observer])
                Lum_obs_time[i, j] = np.mean(Lumph[observer]/192) # you forgot the division in saving the data in fld
                # alpha_obs_time[i, j] = np.mean(alphaph[observer])
                flux_obs_time[i, j] = np.mean(fluxph[observer])
fig, ((ax1, ax2), (ax3, ax5)) = plt.subplots(2, 2, figsize=(20, 15))
figf, ax6 = plt.subplots(1, 1, figsize=(10, 7))
figTr, axTr = plt.subplots(1, 1, figsize=(10, 7))
for i, observer in enumerate(indices_axis):
        ax1.plot(tfb, rph_obs_time[:, i]/apo, label = label_axis[i], c = colors_axis[i])
        ax2.plot(tfb, den_obs_time[:, i]*prel.den_converter, label = label_axis[i], c = colors_axis[i])
        ax3.plot(tfb, T_obs_time[:, i], label = label_axis[i], c = colors_axis[i])
        # ax4.plot(tfb, alpha_obs_time[:, i], label = label_axis[i], c = colors_axis[i])
        ax5.plot(tfb, Lum_obs_time[:, i], label = label_axis[i], c = colors_axis[i])
        ax6.plot(tfb, flux_obs_time[:, i], label = label_axis[i], c = colors_axis[i])
        axTr.plot(tfb, rtr_obs_time[:, i]/apo, label = label_axis[i], c = colors_axis[i])
ax1.axhline(Rp/apo, color = 'gray', linestyle = '--', label = r'R$_{\rm p}$')
ax1.axhline(R0/apo, color = 'gray', linestyle = ':', label = r'R$_0$')
ax1.legend(fontsize = 16)
ax1.set_ylabel(r'$\langle R_{\rm ph}\rangle [R_{\rm a}]$')
ax1.set_ylim(1e-2, 10)
ax2.set_ylabel(r'$\langle \rho_{\rm ph}\rangle$ [g/cm$^3$]')
ax2.set_ylim(1e-14, 5e-9)
ax3.set_ylabel(r'$\langle T_{\rm ph}\rangle$ [K]')
ax3.set_ylim(5e3, 1e10)
ax5.set_ylabel(r'$\langle L_{\rm ph}\rangle$ [erg/s]')
ax5.axhline(y = 1.26e42, color = 'gray', linestyle = '--')
ax5.set_ylim(1e38, 1e41)
ax6.set_ylabel(r'$\langle$ Flux$_{\rm ph}\rangle$ [erg/s cm$^2$]')
ax6.legend(fontsize = 16)
axTr.set_ylabel(r'$\langle R_{\rm tr}\rangle [R_{\rm a}]$')
axTr.legend(fontsize = 16)
for ax in [ax1, ax2, ax3, ax5, ax6, axTr]:
        ax.set_xlabel(r't [$t_{\rm fb}$]')
        ax.tick_params(axis='both', which='major', width=1.2, length=9, color = 'k')
        ax.tick_params(axis='both', which='minor', width=1, length=7, color = 'k')
        ax.grid()
        ax.set_yscale('log')
fig.suptitle(f'{check}', fontsize = 30)
fig.tight_layout()
fig.savefig(f'{abspath}/Figs/{folder}/photo_profile.png')
#%%
plt.figure()
plt.plot(tfb, mean_rph/apo, c = 'k', label = 'mean')
plt.plot(tfb, mean_rph_weig/apo, c = 'firebrick', label = 'weighted by flux')
plt.plot(tfb, gmean_ph/apo, c = 'deepskyblue', label = 'geometric mean')
plt.plot(tfb, median_ph/apo, c = 'forestgreen', label = 'median')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel(r'$\langle R_{ph}\rangle [R_a]$')
plt.yscale('log')
plt.grid()
plt.legend(fontsize = 16)

#%% compare with other resolutions
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
