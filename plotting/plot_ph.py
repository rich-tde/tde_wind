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
from matplotlib import gridspec
from scipy.stats import gmean, ks_2samp, tstd
from Utilities.sections import make_slices
from Utilities.basic_units import radians
from Utilities.operators import find_ratio, sort_list
from Utilities.time_extractor import days_since_distruption
matplotlib.rcParams['figure.dpi'] = 150

#%%
abspath = '/Users/paolamartire/shocks'
G = 1
# first_eq = 88 # observer eq. plane
# final_eq = 103+1 #observer eq. plane
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
snap = '164'
snaps_cdf = [164, 199, 267]
compton = 'Compton'
extr = 'rich'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'{abspath}/TDE/{folder}/{snap}'
saving_path = f'Figs/{folder}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

Rt = Rstar * (Mbh/mstar)**(1/3)
# Rs = 2*G*Mbh / c**2
R0 = 0.6 * Rt
Rp =  Rt / beta
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

#%%
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
# HEALPIX
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
# Convert to latitude and longitude
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 

indecesorbital = np.concatenate(np.where(latitude_moll==0))
long_orb, lat_orb = longitude_moll[indecesorbital], latitude_moll[indecesorbital]

print('Longitude from HELAPIX min and max: ' , np.min(theta), np.max(theta))
print('Longitude for mollweide min and max: ', np.min(longitude_moll), np.max(longitude_moll))
print('Latitude from HELAPIX min and max: ' , np.min(phi), np.max(phi))
print('Latitude for mollweide min and max: ', np.min(latitude_moll), np.max(latitude_moll))
print('So we shift of pi/2 the latitude')
# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(longitude_moll, latitude_moll, s=20, c=np.arange(768))
# ax.scatter(longitude_moll[first_eq:final_eq], latitude_moll[first_eq:final_eq], s=10, c='r')
ax.scatter(long_orb, lat_orb, s=10, c='r')

plt.colorbar(img, ax=ax, label='Observer Number')
ax.set_title("Observers on the Sphere (Mollweide Projection)")
ax.grid(True)
ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
plt.tight_layout()
plt.show()

#%% NB DATA ARE NOT SORTED
ph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{extr}_phidx_fluxes.txt')
snaps, tfb, allindices_ph = ph_data[:, 0].astype(int), ph_data[:, 1], ph_data[:, 2:]
# eliminate the even rows (photosphere indices) of allindices_ph
# tfb = tfb[1::2]
# snaps = snaps[1::2]
fluxes = allindices_ph[1::2]
# fluxes = sort_list(fluxes, snaps)
# tfb = np.sort(tfb)
# snaps = np.sort(snaps)

snaps_ph = np.arange(80,348)
flux_test = np.zeros(len(snaps_ph))
for i, snap in enumerate(snaps_ph):
        selected_lines = np.concatenate(np.where(snaps == snap))
        print(selected_lines)
        selected_idx, selected_fluxes = selected_lines[0], selected_lines[1]
        flux_test[i] = np.sum(allindices_ph[selected_idx])
print(flux_test, len(tfb))
plt.scatter(np.unique(np.sort(tfb)), flux_test, s=2)
plt.yscale('log')
#%%

mean_rph = np.zeros(len(tfb))
mean_rph_weig = np.zeros(len(tfb))
gmean_ph = np.zeros(len(tfb))
mean_size = np.zeros(len(tfb))
standard_dev = np.zeros(len(tfb))
percentile5 = np.zeros(len(tfb))
percentile95 = np.zeros(len(tfb))

for i, snapi in enumerate(snaps):
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        dim_i = vol_i**(1/3)
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_rph[i] = np.mean(rph_i)
        gmean_ph[i] = gmean(rph_i)
        mean_rph_weig[i] = np.sum(rph_i*fluxes[i])/np.sum(fluxes[i])
        mean_size[i] = np.mean(dim_i)
        standard_dev[i] = np.std(rph_i)
        percentile5[i] = np.percentile(rph_i, 5)
        percentile95[i] = np.percentile(rph_i, 95)
plt.figure()
plt.plot(tfb, mean_rph/apo, c = 'k', label = 'mean')
plt.plot(tfb, mean_rph_weig/apo, c = 'r', label = 'weighted by flux')
plt.plot(tfb, gmean_ph/apo, c = 'b', label = 'geometric mean')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel(r'$\langle R_{ph}\rangle [R_a]$')
plt.yscale('log')
plt.grid()
plt.title(f'Fiducial res')
plt.legend()
plt.savefig(f'{abspath}/Figs/{folder}/photo_means.png')

try:
        print('exists')
        data = np.loadtxt(f'{abspath}/data/{folder}/photo_mean.txt', 'w')
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
ph_dataL = np.loadtxt(f'{abspath}/data/{folder}LowRes/LowRes{extr}_phidx_fluxes.txt')
snapsL, tfbL, allindices_phL = ph_dataL[:, 0].astype(int), ph_dataL[:, 1], ph_dataL[:, 2:]
allindices_phL = sort_list(allindices_phL, snapsL)
tfbL = np.sort(tfbL)
snapsL = np.sort(snapsL)
# eliminate the even rows (photosphere indices) of allindices_phL
tfbL = tfbL[1::2]
snapsL = snapsL[1::2]
fluxesL = allindices_phL[1::2]
# snapsL = np.unique(np.sort(snapsL))
# tfbL = np.unique(tfbL)

ph_dataH = np.loadtxt(f'{abspath}/data/{folder}HiRes/HiRes{extr}_phidx_fluxes.txt')
snapsH, tfbH, allindices_phH = ph_dataH[:, 0].astype(int), ph_dataH[:, 1], ph_dataH[:, 2:]
allindices_phH = sort_list(allindices_phH, snapsH)
tfbH = np.sort(tfbH)
snapsH = np.sort(snapsH)
# eliminate the even rows (photosphere indices) of allindices_phH
tfbH = tfbH[1::2]
snapsH = snapsH[1::2]
fluxesH = allindices_phH[1::2]
# snapsH = np.unique(np.sort(snapsH))
# tfbH = np.unique(tfbH)

# compute mean
mean_rphL = np.zeros(len(tfbL))
mean_rphL_weig = np.zeros(len(tfbL))
gmean_phL = np.zeros(len(tfbL))
mean_sizeL = np.zeros(len(tfbL))
percentile5L = np.zeros(len(tfbL))
percentile95L = np.zeros(len(tfbL))
statL = np.zeros(len(tfbL))
pvalueL = np.zeros(len(tfbL))
ratioL_weigh = np.zeros(len(tfbL))
ratioL = np.zeros(len(tfbL))

for i, snapi in enumerate(snapsL):
        # need the fiducial again for the KS stat
        photoFid = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{snapi}.txt')
        xph_iFid, yph_iFid, zph_iFid = photoFid[0], photoFid[1], photoFid[2] 
        rph_iFid = np.sqrt(xph_iFid**2 + yph_iFid**2 + zph_iFid**2)
        # LowRes data
        photo = np.loadtxt(f'{abspath}/data/{folder}LowRes/photo/LowRes_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        dim_i = vol_i**(1/3)
        ksL = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        statL[i], pvalueL[i] = ksL.statistic, ksL.pvalue
        # mean 
        mean_rphL[i] = np.mean(rph_i)
        gmean_phL[i] = gmean(rph_i)
        percentile5L[i] = np.percentile(rph_i, 5)
        percentile95L[i] = np.percentile(rph_i, 95)
        mean_rphL_weig[i] = np.sum(rph_i*fluxesL[i])/np.sum(fluxesL[i])
        mean_sizeL[i] = np.mean(dim_i)
        time = tfbL[i]
        idx = np.argmin(np.abs(tfb-time))
        ratioL[i] = find_ratio(mean_rph[idx], mean_rphL[i])
        ratioL_weigh[i] = find_ratio(mean_rph_weig[idx], mean_rphL_weig[i])

mean_rphH = np.zeros(len(tfbH))
mean_rphH_weig = np.zeros(len(tfbH))
gmean_phH = np.zeros(len(tfbH))
mean_sizeH = np.zeros(len(tfbH))
percentile5H = np.zeros(len(tfbH))
percentile95H = np.zeros(len(tfbH))
statH = np.zeros(len(tfbH))
pvalueH = np.zeros(len(tfbH))
ratioH_weigh = np.zeros(len(tfbH))
ratioH = np.zeros(len(tfbH))

for i, snapi in enumerate(snapsH):
        # need the fiducial again for the KS stat
        photoFid = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{snapi}.txt')
        xph_iFid, yph_iFid, zph_iFid = photoFid[0], photoFid[1], photoFid[2] 
        rph_iFid = np.sqrt(xph_iFid**2 + yph_iFid**2 + zph_iFid**2)
        # HiRes data
        photo = np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        dim_i = vol_i**(1/3)
        ksH = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        statH[i], pvalueH[i] = ksH.statistic, ksH.pvalue
        # mean
        mean_rphH[i] = np.mean(rph_i)
        gmean_phH[i] = gmean(rph_i)
        percentile5H[i] = np.percentile(rph_i, 5)
        percentile95H[i] = np.percentile(rph_i, 95)
        mean_rphH_weig[i] = np.sum(rph_i*fluxesH[i])/np.sum(fluxesH[i])
        mean_sizeH[i] = np.mean(dim_i)
        time = tfbH[i]
        idx = np.argmin(np.abs(tfb-time))
        ratioH[i] = find_ratio(mean_rph[idx], mean_rphH[i])
        ratioH_weigh[i] = find_ratio(mean_rph_weig[idx], mean_rphH_weig[i])
#%% Compare the photosphere distribution
fig, ax = plt.subplots(3, 3, figsize=(20, 12))
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        photoL = np.loadtxt(f'{abspath}/data/{folder}LowRes/photo/LowRes_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        photoH = np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{chosen_snap}.txt')
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

        # plot the distibution of rph
        # zoom in 
        ax[0][i].hist(rphL[rphL<2*apo]/apo, bins = 10, alpha = 0.5, color = 'C1', label = 'Low')
        ax[0][i].hist(rph[rph<2*apo]/apo, bins = 10, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        ax[0][i].hist(rphH[rphH<2*apo]/apo, bins = 10, alpha = 0.5, color = 'darkviolet', label = 'High')
        ax[0][0].set_ylabel(r'Counts for R$\leq 2R_{\rm a}$', fontsize = 20)
        ax[0][i].set_title(r'$t/t_{fb}$='+f'{np.round(time,2)}', fontsize = 20)
        ax[0][i].set_xlim(-0.1,2)

        ax[1][i].hist(rphL/apo, bins = 20, alpha = 0.5, color = 'C1', label = 'Low')
        ax[1][i].hist(rph/apo, bins = 20, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        ax[1][i].hist(rphH/apo, bins = 20, alpha = 0.5, color = 'darkviolet', label = 'High')
        ax[1][0].set_ylabel('Counts (all)')

        # CDF of rph 
        ax[2][i].plot(np.array(rphL_cdf)/apo, cumL, alpha = 0.5, color = 'C1', label = 'Low')
        ax[2][i].plot(np.array(rph_cdf)/apo, cum, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        ax[2][i].plot(np.array(rphH_cdf)/apo, cumH, alpha = 0.5, color = 'darkviolet', label = 'High')
        
        for i in range(3):
                ax[2][i].grid()
                ax[2][i].set_xlabel(r'$R_{ph} [R_a]$')
        ax[1][0].legend(fontsize = 18)
        ax[2][0].set_ylabel('CDF')
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/Rphdistribution.png')

#%% Rph and fluxes
fig = plt.figure(figsize=(25, 25))
gs = gridspec.GridSpec(4, 3, width_ratios=[1,1,1], height_ratios=[3, 3, 3, 0.2], hspace=0.2, wspace = 0.2)
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        photoL = np.loadtxt(f'{abspath}/data/{folder}LowRes/photo/LowRes_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        photoH = np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{chosen_snap}.txt')
        xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        axL = fig.add_subplot(gs[0, i])  # First row
        ax = fig.add_subplot(gs[1, i])  # Second row
        axH = fig.add_subplot(gs[2, i])  # Third row
        # Rph and fluxes
        axL.scatter(np.arange(len(rphL)), rphL/apo, c = fluxesL[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        ax.scatter(np.arange(len(rph)), rph/apo, c = fluxes[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        img = axH.scatter(np.arange(len(rphH)), rphH/apo, c = fluxesH[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        cbar = fig.add_subplot(gs[3, i])  # Colorbar subplot below the first two
        cb = fig.colorbar(img, cax=cbar, orientation='horizontal')
        cb.set_label('Flux')
        axL.set_title(r'$t/t_{\rm fb}$=' + f'{np.round(time,2)}', fontsize = 25)
        axH.set_xlabel(r'Observer index')

        axL.text(0, 4.1, 'Low', fontsize = 25)
        ax.text(0, 4.1, 'Fid', fontsize = 25)
        axH.text(0, 4.1, 'High', fontsize = 25)

        for ax in [axL, ax, axH]:
                ax.set_ylim(0,4.5)
                if i == 0:
                        ax.set_ylabel(r'$R_{\rm ph} [R_{\rm a}]$')
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/RphFluxes.png')

#%% Rph and fluxes
fig = plt.figure(figsize=(25, 25))
gs = gridspec.GridSpec(4, 3, width_ratios=[1,1,1], height_ratios=[3, 3, 3, 0.2], hspace=0.2, wspace = 0.2)
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        photoL = np.loadtxt(f'{abspath}/data/{folder}LowRes/photo/LowRes_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        photoH = np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{chosen_snap}.txt')
        xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        axL = fig.add_subplot(gs[0, i])  # First row
        ax = fig.add_subplot(gs[1, i])  # Second row
        axH = fig.add_subplot(gs[2, i])  # Third row
        # Rph and fluxes
        axL.scatter(latitude_moll*radians, rphL/apo, c = fluxesL[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        ax.scatter(latitude_moll*radians, rph/apo, c = fluxes[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        img = axH.scatter(latitude_moll*radians, rphH/apo, c = fluxesH[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        cbar = fig.add_subplot(gs[3, i])  # Colorbar subplot below the first two
        cb = fig.colorbar(img, cax=cbar, orientation='horizontal')
        cb.set_label('Flux')
        axL.set_title(r'$t/t_{\rm fb}$=' + f'{np.round(time,2)}', fontsize = 25)
        axH.set_xlabel(r'Latitude [rad]')
        ax.set_xlabel('')
        axL.set_xlabel('')
        # delte the "radians" from the xlabel

        axL.text(-1.4, 4.1, 'Low', fontsize = 22)
        ax.text(-1.4, 4.1, 'Fid', fontsize = 22)
        axH.text(-1.4, 4.1, 'High', fontsize = 22)

        for ax in [axL, ax, axH]:
                ax.set_ylim(0,4.5)
                if i == 0:
                        ax.set_ylabel(r'$R_{\rm ph} [R_{\rm a}]$')
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/RphFluxes_latitude.png')

#%% Same but as mellowide projection
fig, ax = plt.subplots(3, 3, figsize=(20, 12), subplot_kw={'projection': 'mollweide'})
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        photoL = np.loadtxt(f'{abspath}/data/{folder}LowRes/photo/LowRes_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        photoH = np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{chosen_snap}.txt')
        xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        for j in range(3):
                ax[i][j].grid(True)
                ax[i][j].set_xticks(np.radians(np.linspace(-180, 180, 9)))
                ax[i][j].set_xticklabels(['-180°', '', '-90°', '', '0°', '', '90°', '', '180°'], fontsize = 18)
                ax[i][j].set_yticks(np.radians(np.linspace(-90, 90, 5)))
                ax[i][j].set_yticklabels(['-90°', '-45°', '0°', '45°', ''])
        
        imgL = ax[0][i].scatter(longitude_moll, latitude_moll, c = rphL/apo, s = fluxesL[chosen_idx]/1e14, norm = colors.LogNorm(vmin = 0.1, vmax = 4))
        img = ax[1][i].scatter(longitude_moll, latitude_moll, c = rph/apo, s = fluxes[chosen_idx]/1e14, norm = colors.LogNorm(vmin = 0.1, vmax = 4))
        imgH = ax[2][i].scatter(longitude_moll, latitude_moll, c = rphH/apo, s = fluxesH[chosen_idx]/1e14, norm = colors.LogNorm(vmin = 0.1, vmax = 4))

        cb = plt.colorbar(imgH, orientation='horizontal')
        cb.set_label(r'$R_{\rm ph} [R_{\rm a}]$')
        ax[0][i].set_title(r'$t/t_{\rm fb}$=' + f'{np.round(time,2)}', fontsize = 25)
plt.savefig(f'{abspath}/Figs/multiple/photo/RphFluxes_moll_cR.png')
#%% Same but as mellowide projection with color = flux
fig, ax = plt.subplots(3, 3, figsize=(20, 12), subplot_kw={'projection': 'mollweide'})
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{extr}_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        photoL = np.loadtxt(f'{abspath}/data/{folder}LowRes/photo/LowRes_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        photoH = np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{chosen_snap}.txt')
        xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        for j in range(3):
                ax[i][j].grid(True)
                ax[i][j].set_xticks(np.radians(np.linspace(-180, 180, 9)))
                ax[i][j].set_xticklabels(['-180°', '', '-90°', '', '0°', '', '90°', '', '180°'], fontsize = 18)
                ax[i][j].set_yticks(np.radians(np.linspace(-90, 90, 5)))
                ax[i][j].set_yticklabels(['-90°', '-45°', '0°', '45°', ''])
        
        imgL = ax[0][i].scatter(longitude_moll, latitude_moll, s = rphL/50, c = fluxesL[chosen_idx], norm = colors.LogNorm(vmin = np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        img = ax[1][i].scatter(longitude_moll, latitude_moll, s = rph/50, c = fluxes[chosen_idx], norm = colors.LogNorm(vmin = np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        imgH = ax[2][i].scatter(longitude_moll, latitude_moll, s = rphH/50, c = fluxesH[chosen_idx], norm = colors.LogNorm(vmin = np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        cb = plt.colorbar(imgH, orientation='horizontal')
        cb.set_label(r'Flux')
        ax[0][i].set_title(r'$t/t_{\rm fb}$=' + f'{np.round(time,2)}', fontsize = 25)
plt.savefig(f'{abspath}/Figs/multiple/photo/RphFluxes_moll_cF.png')

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
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(15, 7), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, mean_rphL_weig/apo, c = 'C1', label = 'Low')
ax1.plot(tfb, mean_rph_weig/apo, c = 'yellowgreen', label = 'Fid')
ax1.plot(tfbH, mean_rphH_weig/apo, c = 'darkviolet', label = 'High')
ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18, loc = 'lower right')

ax2.plot(tfbL, ratioL_weigh, color = 'yellowgreen')
ax2.plot(tfbL, ratioL_weigh, '--', color = 'C1')
ax2.plot(tfbH, ratioH_weigh, color = 'yellowgreen')
ax2.plot(tfbH, ratioH_weigh, '--', color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')

ax3.plot(tfbL, mean_rphL/apo, c = 'C1', label = 'Low')
ax3.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
ax3.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
ax3.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')

ax4.plot(tfbL, ratioL, color = 'yellowgreen')
ax4.plot(tfbL, ratioL, '--', color = 'C1')
ax4.plot(tfbH, ratioH, color = 'yellowgreen')
ax4.plot(tfbH, ratioH, '--', color = 'darkviolet')
ax4.set_ylabel(r'$|\Delta_{\rm rel}|$')
for ax in [ax1, ax2, ax3, ax4]:
        if ax in [ax1, ax3]:
                ax.set_yscale('log')
        ax.grid()
        if ax in [ax2, ax4]:
                ax.set_xlabel(r't [$t_{fb}$]')
                ax.set_ylim(0.5, 3)
ax1.set_title('Flux-weighted photosphere')
ax3.set_title('Standard average photosphere')
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/WeightOrNotRph.png', bbox_inches='tight')
# print('median ratioL_weigh:', np.median(np.nan_to_num(ratioL_weigh)))
# print('median ratioH_weigh:', np.median(np.nan_to_num(ratioH_weigh)))

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, mean_rphL/apo, c = 'C1', label = 'Low')
ax1.plot(tfbL, (mean_rphL + 0.5*mean_sizeL)/apo, c = 'C1', alpha = 0.5)
ax1.plot(tfbL, (mean_rphL - 0.5*mean_sizeL)/apo, c = 'C1', alpha = 0.5)
ax1.fill_between(tfbL, (mean_rphL - 0.5*mean_sizeL)/apo, (mean_rphL + 0.5*mean_sizeL)/apo, color = 'C1', alpha = 0.2)
ax1.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
ax1.plot(tfb, (mean_rph + 0.5*mean_size)/apo, c = 'yellowgreen', alpha = 0.5)
ax1.plot(tfb, (mean_rph - 0.5*mean_size)/apo, c = 'yellowgreen', alpha = 0.5)
ax1.fill_between(tfb, (mean_rph - 0.5*mean_size)/apo, (mean_rph + 0.5*mean_size)/apo, color = 'yellowgreen', alpha = 0.2)
ax1.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
ax1.plot(tfbH, (mean_rphH + 0.5*mean_sizeH)/apo, c = 'darkviolet', alpha = 0.5)
ax1.plot(tfbH, (mean_rphH - 0.5*mean_sizeH)/apo, c = 'darkviolet', alpha = 0.5)
ax1.fill_between(tfbH, (mean_rphH - 0.5*mean_sizeH)/apo, (mean_rphH + 0.5*mean_sizeH)/apo, color = 'darkviolet', alpha = 0.2)
ax1.set_yscale('log')
ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18)

ax2.plot(tfbL, ratioL,  color = 'yellowgreen')
ax2.plot(tfbL, ratioL, '--', color = 'darkorange')
ax2.plot(tfbH, ratioH, color = 'yellowgreen')
ax2.plot(tfbH, ratioH, '--', color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')
ax2.set_ylim(0.9, 3)
for ax in [ax1, ax2]:
        ax.grid()
        ax.set_xlim(0, np.max(tfb))
ax2.set_xlabel(r't [$t_{fb}$]')
# plt.suptitle('Standard average photosphere')
print('median ratioL:', np.median(ratioL))
print('median ratioH:', np.median(ratioH))
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/Rph_diffR.png')

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, mean_rphL/apo, c = 'C1', label = 'Low')
ax1.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
ax1.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
# plot standard deviation as errror bar
ax1.errorbar(tfb, mean_rph/apo, yerr = standard_dev/apo, color = 'yellowgreen', alpha = 0.5)
ax1.set_yscale('log')
ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18)

ax2.plot(tfbL, ratioL,  color = 'yellowgreen')
ax2.plot(tfbL, ratioL, '--', color = 'darkorange')
ax2.plot(tfbH, ratioH, color = 'yellowgreen')
ax2.plot(tfbH, ratioH, '--', color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')
ax2.set_ylim(0.9, 3)
for ax in [ax1, ax2]:
        ax.grid()
        ax.set_xlim(0, np.max(tfb))
ax2.set_xlabel(r't [$t_{fb}$]')
# plt.suptitle('Standard average photosphere')
print('median ratioL:', np.median(ratioL))
print('median ratioH:', np.median(ratioH))
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/Rph_diffDev.png')
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(tfbL, mean_rphL/apo, c = 'C1', label = 'Low')
ax1.plot(tfbL, (mean_rphL + percentile95L)/apo, c = 'C1', alpha = 0.2, linestyle = '--')
ax1.plot(tfbL, (mean_rphL - percentile5L)/apo, c = 'C1', alpha = 0.2, linestyle = '--')
ax1.fill_between(tfbL, (mean_rphL - percentile5L)/apo, (mean_rphL + percentile95L)/apo, color = 'C1', alpha = 0.2)
ax1.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
# plot standard deviation as errror bar
ax1.plot(tfb, (mean_rph + percentile95)/apo, c = 'yellowgreen', alpha = 0.2, linestyle = '--')
ax1.plot(tfb, (mean_rph - percentile5)/apo, c = 'yellowgreen', alpha = 0.2, linestyle = '--')
ax1.fill_between(tfb, (mean_rph - percentile5)/apo, (mean_rph + percentile95)/apo, color = 'yellowgreen', alpha = 0.2)
ax1.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
ax1.plot(tfbH, (mean_rphH + percentile95H)/apo, c = 'darkviolet', alpha = 0.2, linestyle = '--')
ax1.plot(tfbH, (mean_rphH - percentile5H)/apo, c = 'darkviolet', alpha = 0.2, linestyle = '--')
ax1.fill_between(tfbH, (mean_rphH - percentile5H)/apo, (mean_rphH + percentile95H)/apo, color = 'darkviolet', alpha = 0.2)
ax1.set_yscale('log')
ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18)

ax2.plot(tfbL, ratioL,  color = 'yellowgreen')
ax2.plot(tfbL, ratioL, '--', color = 'darkorange')
ax2.plot(tfbH, ratioH, color = 'yellowgreen')
ax2.plot(tfbH, ratioH, '--', color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')
ax2.set_ylim(0.9, 3)
for ax in [ax1, ax2]:
        ax.grid()
        ax.set_xlim(0, np.max(tfb))
ax2.set_xlabel(r't [$t_{fb}$]')
# plt.suptitle('Standard average photosphere')
print('median ratioL:', np.median(ratioL))
print('median ratioH:', np.median(ratioH))
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/photo/Rph_diffPerc.png')
# %%
