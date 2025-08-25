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
snap = '238'
compton = 'Compton'

commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
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
img = ax.scatter(longitude_moll, latitude_moll, s=20, c=np.arange(len(longitude_moll)), cmap='viridis')
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
ph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{commonfold}NewAMR/NewAMR_phidx_fluxes.txt')
snaps, tfb, allindices_ph = ph_data[:, 0].astype(int), ph_data[:, 1], ph_data[:, 2:]
# snaps_ph = np.arange(np.min(snaps), np.max(snaps)+1)
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
        photo = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/photo/NewAMR_photo{snapi}.txt')
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
plt.title(f'Fiducial res')
plt.legend(fontsize = 16)

# try:
#         print('exists')
#         data = np.loadtxt(f'{abspath}/data/{commonfold}/photo_stat.txt')
# except FileNotFoundError:
#         print('save')
#         with open(f'{abspath}/data/{commonfold}/photo_stat.txt', 'a') as f:
#                 f.write(f'# tfb, median_ph, mean_rph, gmean, weighted by flux\n')
#                 f.write(' '.join(map(str, tfb)) + '\n')
#                 f.write(' '.join(map(str, median_ph)) + '\n')
#                 f.write(' '.join(map(str, mean_rph)) + '\n')
#                 f.write(' '.join(map(str, gmean_ph)) + '\n')
#                 f.write(' '.join(map(str, mean_rph_weig)) + '\n')
#                 f.close()

#%% compare with other resolutions
# Load data
ph_dataL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/LowResNewAMR_phidx_fluxes.txt')
snapsL, tfbL, allindices_phL = ph_dataL[:, 0].astype(int), ph_dataL[:, 1], ph_dataL[:, 2:]
fluxesL = []
for i, snap in enumerate(snapsL):
        selected_lines = np.concatenate(np.where(snapsL == snap))
        # eliminate the even rows (photosphere indices) of allindices_ph
        selected_idx, selected_fluxesL = selected_lines[0], selected_lines[1]
        fluxesL.append(allindices_phL[selected_fluxesL])
fluxesL = np.array(fluxesL)
tfbL, fluxesL, snapsL = sort_list([tfbL, fluxesL, snapsL], snapsL, unique=True)

ph_dataH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/HiResNewAMR_phidx_fluxes.txt')
snapsH, tfbH, allindices_phH = ph_dataH[:, 0].astype(int), ph_dataH[:, 1], ph_dataH[:, 2:]
fluxesH = []
for i, snap in enumerate(snapsH):
        selected_lines = np.concatenate(np.where(snapsH == snap))
        # eliminate the even rows (photosphere indices) of allindices_ph
        selected_idx, selected_fluxesH = selected_lines[0], selected_lines[1]
        fluxesH.append(allindices_phH[selected_fluxesH])
fluxesH = np.array(fluxesH)
tfbH, fluxesH, snapsH = sort_list([tfbH, fluxesH, snapsH], snapsH, unique=True)
plt.figure()
plt.scatter(tfbL, np.sum(fluxesL, axis=1), s=2, c = 'C1')
plt.scatter(tfb, np.sum(fluxes, axis=1), s=2, c = 'yellowgreen')
plt.scatter(tfbH, np.sum(fluxesH, axis=1), s=2, c = 'darkviolet')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel('Total flux [cgs]')
plt.yscale('log')

#%% compute mean
mean_rphL = np.zeros(len(tfbL))
mean_rphL_weig = np.zeros(len(tfbL))
gmean_phL = np.zeros(len(tfbL))
median_phL = np.zeros(len(tfbL))
percentile16L = np.zeros(len(tfbL))
percentile84L = np.zeros(len(tfbL))
statL = np.zeros(len(tfbL))
pvalueL = np.zeros(len(tfbL))

for i, snapi in enumerate(snapsL):
        # LowRes data
        photo = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/photo/LowResNewAMR_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        if rph_i.any() < R0:
                print('Less than R0:', rph_i[rph_i<R0])
        # ksL = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        # statL[i], pvalueL[i] = ksL.statistic, ksL.pvalue
        mean_rphL[i] = np.mean(rph_i)
        mean_rphL_weig[i] = np.sum(rph_i*fluxesL[i])/np.sum(fluxesL[i])
        gmean_phL[i] = gmean(rph_i)
        median_phL[i] = np.median(rph_i)
        percentile16L[i] = np.percentile(rph_i, 16)
        percentile84L[i] = np.percentile(rph_i, 84)

timeL_ratio, ratioL = ratio_BigOverSmall(tfb, mean_rph, tfbL, mean_rphL)
_, ratioLmedian = ratio_BigOverSmall(tfb, median_ph, tfbL, median_phL)
_, ratioL_weigh = ratio_BigOverSmall(tfb, mean_rph_weig, tfbL, mean_rphL_weig)
_, ratioL_gmean = ratio_BigOverSmall(tfb, gmean_ph, tfbL, gmean_phL)

mean_rphH = np.zeros(len(tfbH))
mean_rphH_weig = np.zeros(len(tfbH))
gmean_phH = np.zeros(len(tfbH))
median_phH = np.zeros(len(tfbH))
percentile16H = np.zeros(len(tfbH))
percentile84H = np.zeros(len(tfbH))

for i, snapi in enumerate(snapsH):
        # HiRes data
        photo = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/photo/HiResNewAMR_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        # mean
        mean_rphH[i] = np.mean(rph_i)
        gmean_phH[i] = gmean(rph_i)
        median_phH[i] = np.median(rph_i)
        idx_median = np.argmin(np.abs(rph_i - median_phH[i]))
        percentile16H[i] = np.percentile(rph_i, 16)
        percentile84H[i] = np.percentile(rph_i, 84)
        mean_rphH_weig[i] = np.sum(rph_i*fluxesH[i])/np.sum(fluxesH[i])

timeH_ratio, ratioH = ratio_BigOverSmall(tfb, mean_rph, tfbH, mean_rphH)
_, ratioHmedian = ratio_BigOverSmall(tfb, median_ph, tfbH, median_phH)
_, ratioH_weigh = ratio_BigOverSmall(tfb, mean_rph_weig, tfbH, mean_rphH_weig)
_, ratioH_gmean = ratio_BigOverSmall(tfb, gmean_ph, tfbH, gmean_phH)

#%% Compare the photosphere distribution
snaps_cdf = [snaps[np.argmin(np.abs(tfb-0.46))], snaps[np.argmin(np.abs(tfb-0.53))], snaps[np.argmin(np.abs(tfb-0.57))]]
fig, ax = plt.subplots(3, 3, figsize=(20, 12))
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]      
        photo = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/photo/NewAMR_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)

        idx_L = np.argmin(np.abs(tfbL - time))
        chosen_snapL = snapsL[idx_L]
        photoL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/photo/LowResNewAMR_photo{chosen_snapL}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        # time = tfb[np.argmin(np.abs(snapsH - chosen_snap))]
        # chosen_idx = np.argmin(np.abs(snapsH - chosen_snap))
        # photoH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/photo/HiResNewAMR_photo{chosen_snap}.txt')
        # xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        # rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        # rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        rphL_cdf = np.sort(rphL)
        cumL = list(np.arange(len(rphL_cdf))/len(rphL_cdf))
        rphL_cdf = list(rphL_cdf)
        rph_cdf = np.sort(rph)
        cum = list(np.arange(len(rph_cdf))/len(rph_cdf))
        rph_cdf = list(rph_cdf)
        # rphH_cdf = np.sort(rphH)
        # cumH = list(np.arange(len(rphH_cdf))/len(rphH_cdf))
        # rphH_cdf = list(rphH_cdf)

        # rphL_cdf_weig = np.sort(np.log10(rphL_weig))
        # cum_weigL = list(np.arange(len(rphL_cdf_weig))/len(rphL_cdf_weig))
        # rphL_cdf_weig = list(rphL_cdf_weig)
        # rph_cdf_weig = np.sort(np.log10(rph_weig))
        # cum_weig = list(np.arange(len(rph_cdf_weig))/len(rph_cdf_weig))
        # rph_cdf_weig = list(rph_cdf_weig)
        # # rphH_cdf_weig = np.sort(rphH_weig)
        # # cum_weigH = list(np.arange(len(rphH_cdf_weig))/len(rphH_cdf_weig))
        # # rphH_cdf_weig = list(rphH_cdf_weig)

        # plot the distibution of rph with logspaced bin
        binsLzoom = np.linspace(np.log10(np.min(rphL[rphL<2*Rt]/Rt)), 2, 20)
        binszoom = np.linspace(np.log10(np.min(rph[rph<2*Rt]/Rt)), 2, 20)
        # binsHzoom = np.linspace(np.log10(np.min(rphH[rphH<2*Rt]/Rt)), 2, 20)
        # zoom in 
        ax[0][i].hist(rphL[rphL<2*Rt]/Rt, bins = 60, alpha = 0.5, color = 'C1', label = 'Low')
        ax[0][i].hist(rph[rph<2*Rt]/Rt, bins = 60, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        # ax[0][i].hist(rphH[rphH<2*Rt]/Rt, bins = binsHzoom, alpha = 0.5, color = 'darkviolet', label = 'High')
        ax[0][0].set_ylabel(r'Counts for R$\leq 2R_{\rm a}$', fontsize = 20)
        ax[0][i].set_title(r'$t/t_{\rm fb}$='+f'{np.round(time,2)}', fontsize = 20)
        ax[0][i].set_xlim(-0.1,2)

        binsL = np.logspace(np.log10(np.min(rphL/Rt)), np.log10(np.max(rphL/Rt)), 50)
        bins = np.logspace(np.log10(np.min(rph/Rt)), np.log10(np.max(rph/Rt)), 50)
        # binsH = np.logspace(np.log10(np.min(rphH/Rt)), np.log10(np.max(rphH/Rt)), 50)
        ax[1][i].hist(rphL/Rt, bins = 60, alpha = 0.5, color = 'C1', label = 'Low')
        ax[1][i].hist(rph/Rt, bins = 60, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        # ax[1][i].hist(rphH/Rt, bins = binsH, alpha = 0.5, color = 'darkviolet', label = 'High')
        ax[1][0].set_ylabel('Counts (all)')

        # CDF of rph 
        print('CDF:', len(cumL))
        ax[2][i].plot(np.array(rphL_cdf)/Rt, cumL, alpha = 0.5, color = 'C1', label = 'Low')
        ax[2][i].plot(np.array(rph_cdf)/Rt, cum, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
        # ax[2][i].plot(np.array(rphH_cdf)/Rt, cumH, alpha = 0.5, color = 'darkviolet', label = 'High')
        
        for i in range(3):
                ax[2][i].grid()
                ax[2][i].set_xlabel(r'$R_{ph} [R_t]$')
        ax[0][0].legend(fontsize = 18)
        ax[2][0].set_ylabel('CDF')
plt.tight_layout()
# plt.savefig(f'{abspath}/Figs/multiple/photo/Rphdistribution.png')

#%% Rph and fluxes
fig = plt.figure(figsize=(25, 25))
gs = gridspec.GridSpec(4, 3, width_ratios=[1,1,1], height_ratios=[3, 3, 3, 0.2], hspace=0.2, wspace = 0.2)
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/photo/NewAMR_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        time = tfb[np.argmin(np.abs(snapsL - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snapsL - chosen_snap))
        photoL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/photo/LowResNewAMR_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        time = tfb[np.argmin(np.abs(snapsH - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snapsH - chosen_snap))
        # photoH = np.loadtxt(f'{abspath}/data/{commonfold}HiRes/photo/HiRes_photo{chosen_snap}.txt')
        # xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        # rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        # rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        axL = fig.add_subplot(gs[0, i])  # First row
        ax = fig.add_subplot(gs[1, i])  # Second row
        axH = fig.add_subplot(gs[2, i])  # Third row
        # Rph and fluxes
        axL.scatter(np.arange(len(rphL)), rphL/apo, c = fluxesL[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        ax.scatter(np.arange(len(rph)), rph/apo, c = fluxes[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        # img = axH.scatter(np.arange(len(rphH)), rphH/apo, c = fluxesH[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
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

#%% Rph and fluxes
fig = plt.figure(figsize=(25, 25))
gs = gridspec.GridSpec(4, 3, width_ratios=[1,1,1], height_ratios=[3, 3, 3, 0.2], hspace=0.2, wspace = 0.2)
for i, chosen_snap in enumerate(snaps_cdf):
        time = tfb[np.argmin(np.abs(snaps - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snaps - chosen_snap))
        photo = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/photo/NewAMR_photo{chosen_snap}.txt')
        xph, yph, zph = photo[0], photo[1], photo[2] 
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        rph_weig = rph*fluxes[chosen_idx]/np.sum(fluxes[chosen_idx])
        time = tfb[np.argmin(np.abs(snapsL - chosen_snap))]
        chosen_idx = np.argmin(np.abs(snapsL - chosen_snap))
        photoL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/photo/LowResNewAMR_photo{chosen_snap}.txt')
        xphL, yphL, zphL = photoL[0], photoL[1], photoL[2]
        rphL = np.sqrt(xphL**2 + yphL**2 + zphL**2)
        # rphL_weig = rphL * fluxesL[chosen_idx]/np.sum(fluxesL[chosen_idx])
        # time = tfb[np.argmin(np.abs(snapsH - chosen_snap))]
        # chosen_idx = np.argmin(np.abs(snapsH - chosen_snap))
        # photoH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/photo/HiResNewAMR_photo{chosen_snap}.txt')
        # xphH, yphH, zphH = photoH[0], photoH[1], photoH[2]
        # rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        # rphH_weig = rphH * fluxesH[chosen_idx]/np.sum(fluxesH[chosen_idx])

        axL = fig.add_subplot(gs[0, i])  # First row
        ax = fig.add_subplot(gs[1, i])  # Second row
        axH = fig.add_subplot(gs[2, i])  # Third row
        # Rph and fluxes
        axL.scatter(latitude_moll*radians, rphL/apo, c = fluxesL[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        img = ax.scatter(latitude_moll*radians, rph/apo, c = fluxes[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
        # img = axH.scatter(latitude_moll*radians, rphH/apo, c = fluxesH[chosen_idx], s = 50, norm = colors.LogNorm(vmin = 1e12, vmax = 1e15))#np.min(fluxes[chosen_idx]), vmax = np.max(fluxes[chosen_idx])))
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
# plt.savefig(f'{abspath}/Figs/multiple/photo/RphFluxes_latitude.png')


#%% p values
plt.figure(figsize=(10, 5))
plt.plot(tfbL, pvalueL, color = 'C1', label = 'Low')
# plt.plot(tfbH, pvalueH, color = 'darkviolet', label = 'High')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel('p-value')
plt.yscale('log')
plt.ylim(1e-20, 1)
plt.grid()
plt.legend(fontsize = 18)

#%%
fig, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax8)) = plt.subplots(2, 4, figsize=(25, 7), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)

ax1.plot(tfbL, mean_rphL/apo, c = 'C1', label = 'Low')
ax1.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
ax1.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
ax1.set_ylabel(r'mean $R_{\rm ph} [R_a]$')

ax2.plot(timeL_ratio, ratioL, color = 'yellowgreen')
ax2.plot(timeL_ratio, ratioL, '--', color = 'C1')
ax2.plot(timeH_ratio, ratioH, color = 'yellowgreen')
ax2.plot(timeH_ratio, ratioH, '--', color = 'darkviolet')
ax2.set_ylabel(r'$\mathcal{R} R_{\rm ph}$')

ax3.plot(tfbL, gmean_phL/apo, c = 'C1', label = 'Low')
ax3.plot(tfb, gmean_ph/apo, c = 'yellowgreen', label = 'Fid')
ax3.plot(tfbH, gmean_phH/apo, c = 'darkviolet', label = 'High')
ax3.set_ylabel(r'gmean $R_{\rm ph} [R_a]$')

ax4.plot(timeL_ratio, ratioL_gmean, color = 'yellowgreen')
ax4.plot(timeL_ratio, ratioL_gmean, '--', color = 'C1')
ax4.plot(timeH_ratio, ratioH_gmean, color = 'yellowgreen')
ax4.plot(timeH_ratio, ratioH_gmean, '--', color = 'darkviolet') 

ax5.plot(tfbL, median_phL/apo, c = 'C1', label = 'Low')
ax5.plot(tfb, median_ph/apo, c = 'yellowgreen', label = 'Fid')
ax5.plot(tfbH, median_phH/apo, c = 'darkviolet', label = 'High')
ax5.set_ylabel(r'median $R_{\rm ph} [R_a]$')

ax6.plot(timeL_ratio, ratioLmedian, color = 'yellowgreen')
ax6.plot(timeL_ratio, ratioLmedian, '--', color = 'C1')
ax6.plot(timeH_ratio, ratioHmedian, color = 'yellowgreen')
ax6.plot(timeH_ratio, ratioHmedian, '--', color = 'darkviolet')

ax7.plot(tfbL, mean_rphL_weig/apo, c = 'C1', label = 'Low')
ax7.plot(tfb, mean_rph_weig/apo, c = 'yellowgreen', label = 'Fid')
ax7.plot(tfbH, mean_rphH_weig/apo, c = 'darkviolet', label = 'High')
ax7.set_ylabel(r'weighted mean $R_{ph} [R_a]$')

ax8.plot(timeL_ratio, ratioL_weigh, color = 'yellowgreen')
ax8.plot(timeL_ratio, ratioL_weigh, '--', color = 'C1')
ax8.plot(timeH_ratio, ratioH_weigh, color = 'yellowgreen')
ax8.plot(timeH_ratio, ratioH_weigh, '--', color = 'darkviolet')
# ax5.axvline(0.46)
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        if ax in [ax1, ax3, ax5, ax7]:
                ax.set_yscale('log')
                ax.axhline(R0/apo, c='k', ls=':', label=r'$R_0$')
                ax.axhline(Rt/apo, c='k', ls='--', label=r'$R_{\rm t}$')
                ax.set_ylim(2e-3, 4.5)
        ax.grid()
        if ax in [ax2, ax4, ax6, ax8]:
                ax.set_xlabel(r't [$t_{fb}$]')
                ax.set_ylim(0.5, 3)
ax1.legend(fontsize = 16, loc = 'lower right')
plt.tight_layout()

# %%
