""" Plot the photosphere."""
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src import orbits as orb
import matplotlib.colors as colors
import Utilities.prelude as prel
import healpy as hp
from scipy.stats import gmean, ks_2samp, tstd
from Utilities.sections import make_slices
from Utilities.operators import make_tree, sort_list
from Utilities.time_extractor import days_since_distruption
matplotlib.rcParams['figure.dpi'] = 150


abspath = '/Users/paolamartire/shocks'
G = 1
first_eq = 88 # observer eq. plane
final_eq = 104 #observer eq. plane
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = '' # '' or 'HiRes' or 'LowRes'
snap = '267'
snaps_cdf = [164, 199, 267]
compton = 'Compton'
extr = 'rich'
if snap in ['164', '248']:
        also_xz = True
else:
      also_xz = False

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
path = f'{abspath}/TDE/{folder}/{snap}'
saving_path = f'Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

Rt = Rstar * (Mbh/mstar)**(1/3)
# Rs = 2*G*Mbh / c**2
R0 = 0.6 * Rt
Rp =  Rt / beta
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T

#%% HEALPIX
# x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
# r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
# theta = np.arctan2(y, x)          # Azimuthal angle in radians
# phi = np.arccos(z / r)            # Elevation angle in radians
# # Convert to latitude and longitude
# longitude = theta              
# latitude = np.pi / 2 - phi 
# # Plot in 2D using a Mollweide projection
# fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
# img = ax.scatter(longitude, latitude, s=20, c=np.arange(192))
# ax.scatter(longitude[first_eq:final_eq], latitude[first_eq:final_eq], s=10, c='r')
# plt.colorbar(img, ax=ax, label='Observer Number')
# ax.set_title("Observers on the Sphere (Mollweide Projection)")
# ax.grid(True)
# ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
# ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
# plt.title(f' Oribital plane indices: {int(first_eq)} - {int(final_eq)}')
# plt.tight_layout()
# plt.show()

# NB DATA ARE NOT SORTED
ph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}{extr}_phidx_fluxes.txt')
snaps, tfb, allindices_ph = ph_data[:, 0].astype(int), ph_data[:, 1], ph_data[:, 2:]
allindices_ph = sort_list(allindices_ph, snaps)
tfb = np.sort(tfb)
# eliminate the even rows (photosphere indices) of allindices_ph
fluxes = allindices_ph[1::2]
snaps = np.unique(np.sort(snaps))
tfb = np.unique(tfb)

mean_rph = np.zeros(len(tfb))
mean_rph_weig = np.zeros(len(tfb))
gmean_ph = np.zeros(len(tfb))
mean_size = np.zeros(len(tfb))
standard_dev = np.zeros(len(tfb))

for i, snapi in enumerate(snaps):
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        dim_i = vol_i**(1/3)
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_rph[i] = np.mean(rph_i)
        gmean_ph[i] = gmean(rph_i)
        mean_rph_weig[i] = np.sum(rph_i*fluxes[i])/np.sum(fluxes[i])
        mean_size[i] = np.mean(dim_i)
        standard_dev[i] = np.std(rph_i)
plt.figure()
plt.plot(tfb, mean_rph/apo, c = 'k', label = 'mean')
plt.plot(tfb, mean_rph_weig/apo, c = 'r', label = 'weighted by flux')
plt.plot(tfb, gmean_ph/apo, c = 'b', label = 'geometric mean')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel(r'$\langle R_{ph}\rangle [R_a]$')
plt.yscale('log')
plt.grid()
plt.title(f'Check: {check}')
plt.legend()
plt.savefig(f'{abspath}/Figs/{folder}/photo_mean.png')

try:
        print('exists')
        data = np.loadtxt(f'{abspath}/data/{folder}/photo_mean.txt')
except:
        with open(f'{abspath}/data/{folder}/photo_mean.txt', 'a') as f:
                f.write(f'# tfb, mean_rph,gmean,  weighted by flux\n')
                f.write(' '.join(map(str, tfb)) + '\n')
                f.write(' '.join(map(str, mean_rph)) + '\n')
                f.write(' '.join(map(str, gmean_ph)) + '\n')
                f.write(' '.join(map(str, mean_rph_weig)) + '\n')
                f.close()

#%% compare with other resolutions
# Load data
ph_dataL = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}LowRes/LowRes{extr}_phidx_fluxes.txt')
snapsL, tfbL, allindices_phL = ph_dataL[:, 0].astype(int), ph_dataL[:, 1], ph_dataL[:, 2:]
allindices_phL = sort_list(allindices_phL, snapsL)
tfbL = np.sort(tfbL)
# eliminate the even rows (photosphere indices) of allindices_phL
fluxesL = allindices_phL[1::2]
snapsL = np.unique(np.sort(snapsL))
tfbL = np.unique(tfbL)

ph_dataH = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}HiRes/HiRes{extr}_phidx_fluxes.txt')
snapsH, tfbH, allindices_phH = ph_dataH[:, 0].astype(int), ph_dataH[:, 1], ph_dataH[:, 2:]
allindices_phH = sort_list(allindices_phH, snapsH)
tfbH = np.sort(tfbH)
# eliminate the even rows (photosphere indices) of allindices_phH
fluxesH = allindices_phH[1::2]
snapsH = np.unique(np.sort(snapsH))
tfbH = np.unique(tfbH)

# compute mean
mean_rphL = np.zeros(len(tfbL))
mean_rphL_weig = np.zeros(len(tfbL))
gmean_phL = np.zeros(len(tfbL))
mean_sizeL = np.zeros(len(tfbL))
statL = np.zeros(len(tfbL))
pvalueL = np.zeros(len(tfbL))
diffL_weigh = []
diffL = []

for i, snapi in enumerate(snapsL):
        photoFid = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snapi}.txt')
        xph_iFid, yph_iFid, zph_iFid = photoFid[0], photoFid[1], photoFid[2] 
        rph_iFid = np.sqrt(xph_iFid**2 + yph_iFid**2 + zph_iFid**2)
        # LowRes data
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}LowRes/photo/LowRes_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        dim_i = vol_i**(1/3)
        ksL = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        statL[i], pvalueL[i] = ksL.statistic, ksL.pvalue
        # mean 
        mean_rphL[i] = np.mean(rph_i)
        gmean_phL[i] = gmean(rph_i)
        mean_rphL_weig[i] = np.sum(rph_i*fluxesL[i])/np.sum(fluxesL[i])
        mean_sizeL[i] = np.mean(dim_i)
        time = tfbL[i]
        idx = np.argmin(np.abs(tfb-time))
        diffL.append(2*np.abs(mean_rph[idx]-mean_rphL[i])/(mean_rph[idx]+mean_rphL[i]))
        diffL_weigh.append(np.abs(mean_rph_weig[idx]-mean_rphL_weig[i])/(mean_rph_weig[idx]+mean_rphL_weig[i]))

mean_rphH = np.zeros(len(tfbH))
mean_rphH_weig = np.zeros(len(tfbH))
gmean_phH = np.zeros(len(tfbH))
mean_sizeH = np.zeros(len(tfbH))
statH = np.zeros(len(tfbH))
pvalueH = np.zeros(len(tfbH))
diffH_weigh = []
diffH = []

for i, snapi in enumerate(snapsH):
        photoFid = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snapi}.txt')
        xph_iFid, yph_iFid, zph_iFid = photoFid[0], photoFid[1], photoFid[2] 
        rph_iFid = np.sqrt(xph_iFid**2 + yph_iFid**2 + zph_iFid**2)
        # HiRes data
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}HiRes/photo/HiRes_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        dim_i = vol_i**(1/3)
        ksH = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        statH[i], pvalueH[i] = ksH.statistic, ksH.pvalue
        # mean
        mean_rphH[i] = np.mean(rph_i)
        gmean_phH[i] = gmean(rph_i)
        mean_rphH_weig[i] = np.sum(rph_i*fluxesL[i])/np.sum(fluxesL[i])
        mean_sizeH[i] = np.mean(dim_i)
        time = tfbH[i]
        idx = np.argmin(np.abs(tfb-time))
        diffH.append(2*np.abs(mean_rph[idx]-mean_rphH[i])/(mean_rph[idx]+mean_rphH[i]))
        diffH_weigh.append(np.abs(mean_rph_weig[idx]-mean_rphH_weig[i])/(mean_rph_weig[idx]+mean_rphH_weig[i]))

#%% Compare the photosphere distribution
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        photoL = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}LowRes/photo/LowRes_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        photoH = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}HiRes/photo/HiRes_photo{chosen_snap}.txt')
        xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        rphL_cdf = np.sort(rphL)
        cumL = list(np.arange(len(rphL_cdf))/len(rphL_cdf))
        rphL_cdf = list(rphL_cdf)
        rph_cdf = np.sort(rph)
        cum = list(np.arange(len(rph_cdf))/len(rph_cdf))
        rph_cdf = list(rph_cdf)
        rphH_cdf = np.sort(rphH)
        cumH = list(np.arange(len(rphH_cdf))/len(rphH_cdf))
        rphH_cdf = list(rphH_cdf)

        rphL_cdf_weig = np.sort(rphL_weig)
        cum_weigL = list(np.arange(len(rphL_cdf_weig))/len(rphL_cdf_weig))
        rphL_cdf_weig = list(rphL_cdf_weig)
        rph_cdf_weig = np.sort(rph_weig)
        cum_weig = list(np.arange(len(rph_cdf_weig))/len(rph_cdf_weig))
        rph_cdf_weig = list(rph_cdf_weig)
        rphH_cdf_weig = np.sort(rphH_weig)
        cum_weigH = list(np.arange(len(rphH_cdf_weig))/len(rphH_cdf_weig))
        rphH_cdf_weig = list(rphH_cdf_weig)

        # CDF of rph 
        ax[i].set_title(r'$t/t_{fb}$='+f'{np.round(time,2)}', fontsize = 20)
        ax[i].plot(np.array(rphL_cdf)/apo, cumL, alpha = 0.5, color = 'C1', label = 'Low')
        ax[i].plot(np.array(rph_cdf)/apo, cum, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        ax[i].plot(np.array(rphH_cdf)/apo, cumH, alpha = 0.5, color = 'darkviolet', label = 'High')
        # ax1.set_ylim(0.5, 1.1)
        for i in range(3):
                ax[i].grid()
                ax[i].set_xlabel(r'$R_{ph} [R_a]$')
        ax[0].legend(fontsize = 18)
        ax[0].set_ylabel('CDF')

#%% p values
plt.figure(figsize=(10, 5))
plt.plot(tfbL, pvalueL, color = 'C1', label = 'Low')
plt.plot(tfbH, pvalueH, color = 'darkviolet', label = 'High')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel('p-value')
plt.yscale('log')
plt.ylim(1e-20, 1)
plt.grid()
plt.legend(fontsize = 18)

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, mean_rphL_weig/apo, c = 'C1', label = 'Low')
ax1.plot(tfb, mean_rph_weig/apo, c = 'yellowgreen', label = 'Fid')
ax1.plot(tfbH, mean_rphH_weig/apo, c = 'darkviolet', label = 'High')
ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18)

ax2.plot(tfbL, diffL_weigh, color = 'C1')
ax2.plot(tfbH, diffH_weigh, color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')
for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.grid()
plt.suptitle('Flux-weighted photosphere')
plt.savefig(f'{abspath}/Figs/multiple/Rph_diff.png')
print('median diffL_weigh:', np.median(np.nan_to_num(diffL_weigh)))
print('median diffH_weigh:', np.median(np.nan_to_num(diffH_weigh)))

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, mean_rphL/apo, c = 'C1', label = 'Low')
ax1.plot(tfbL, (mean_rphL + 0.5*mean_sizeL)/apo, c = 'C1', alpha = 0.5)
ax1.plot(tfbL, (mean_rphL - 0.5*mean_sizeL)/apo, c = 'C1', alpha = 0.5)
ax1.fill_between(tfbL, (mean_rphL - 0.5*mean_sizeL)/apo, (mean_rphL + 0.5*mean_sizeL)/apo, color = 'C1', alpha = 0.2)
ax1.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
# plot standard deviation as errror bar
ax1.errorbar(tfb, mean_rph/apo, yerr = standard_dev/apo, fmt='o', color = 'yellowgreen', alpha = 0.5)
ax1.plot(tfb, (mean_rph + 0.5*mean_size)/apo, c = 'yellowgreen', alpha = 0.5)
ax1.plot(tfb, (mean_rph - 0.5*mean_size)/apo, c = 'yellowgreen', alpha = 0.5)
ax1.fill_between(tfb, (mean_rph - 0.5*mean_size)/apo, (mean_rph + 0.5*mean_size)/apo, color = 'yellowgreen', alpha = 0.2)
ax1.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
ax1.plot(tfbH, (mean_rphH + 0.5*mean_sizeH)/apo, c = 'darkviolet', alpha = 0.5)
ax1.plot(tfbH, (mean_rphH - 0.5*mean_sizeH)/apo, c = 'darkviolet', alpha = 0.5)
ax1.fill_between(tfbH, (mean_rphH - 0.5*mean_sizeH)/apo, (mean_rphH + 0.5*mean_sizeH)/apo, color = 'darkviolet', alpha = 0.2)

ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18)

ax2.plot(tfbL, diffL, color = 'darkorange')
ax2.plot(tfbH, diffH, color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')
ax2.set_ylim(1e-2, 10)
for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlim(0, np.max(tfb))
ax2.set_xlabel(r't [$t_{fb}$]')
# plt.suptitle('Standard average photosphere')
print('median diffL:', np.median(diffL))
print('median diffH:', np.median(diffH))
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/Rph_diff.pdf')


# %%
Nph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/nouvrich{check}_phidx_fluxes.txt')
Nsnaps, Ntfb, Nallindices_ph = Nph_data[:, 0].astype(int), Nph_data[:, 1], Nph_data[:, 2:]
Nallindices_ph = sort_list(Nallindices_ph, Nsnaps)
Ntfb = np.sort(Ntfb)
# eliminate the even rows (photosphere indices) of Nallindices_ph
Nfluxes = Nallindices_ph[1::2]
Nsnaps = np.unique(np.sort(Nsnaps))
Ntfb = np.unique(Ntfb)

Nph_dataL = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/nouvrich{check}_phidx_fluxes.txt')
NsnapsL, NtfbL, Nallindices_phL = Nph_dataL[:, 0].astype(int), Nph_dataL[:, 1], Nph_dataL[:, 2:]
Nallindices_phL = sort_list(Nallindices_phL, NsnapsL)
NtfbL = np.sort(NtfbL)
# eliminate the even rows (photosphere indices) of Nallindices_phL
Nfluxes = Nallindices_phL[1::2]
NsnapsL = np.unique(np.sort(NsnapsL))
NtfbL = np.unique(NtfbL)
# %
# %%
plt.figure()
for i, snapi in enumerate(Nsnaps):
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        Nphoto = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/nouvrich{check}_photo{snapi}.txt')
        Nxph_i, Nyph_i, Nzph_i, Nvol_i = Nphoto[0], Nphoto[1], Nphoto[2], Nphoto[3]
        Nrph_i = np.sqrt(Nxph_i**2 + Nyph_i**2 + Nzph_i**2)
        plt.scatter(np.arange(192), rph_i/Nrph_i, s = 10, label = 't/tfb = '+f'{np.round(Ntfb[i],2)}')
        plt.ylabel(r'$R_{ph, old}/ R_{ph, new}$')
        plt.xlabel('Observer')
        plt.yscale('log')
plt.legend()
plt.grid()
# %%
plt.figure()
for i, snapi in enumerate(Nsnaps):
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}LowRes/photo/LowRes_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        Nphoto = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}LowRes/photo/nouvrichLowRes_photo{snapi}.txt')
        Nxph_i, Nyph_i, Nzph_i, Nvol_i = Nphoto[0], Nphoto[1], Nphoto[2], Nphoto[3]
        Nrph_i = np.sqrt(Nxph_i**2 + Nyph_i**2 + Nzph_i**2)
        plt.scatter(np.arange(192), rph_i/Nrph_i, s = 10, label = 't/tfb = '+f'{np.round(Ntfb[i],2)}')
        plt.ylabel(r'$R_{ph, old}/ R_{ph, new}$')
        plt.xlabel('Observer')
        plt.yscale('log')
plt.legend()
plt.title('Check: Low')
plt.grid()
# %%
