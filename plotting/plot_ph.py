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
from scipy.stats import gmean, ks_2samp
from Utilities.sections import make_slices
from Utilities.operators import make_tree, sort_list
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
snap = '199'
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
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
# Convert to latitude and longitude
longitude = theta              
latitude = np.pi / 2 - phi 
# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(longitude, latitude, s=20, c=np.arange(192))
ax.scatter(longitude[first_eq:final_eq], latitude[first_eq:final_eq], s=10, c='r')
plt.colorbar(img, ax=ax, label='Observer Number')
ax.set_title("Observers on the Sphere (Mollweide Projection)")
ax.grid(True)
ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
plt.title(f' Oribital plane indices: {int(first_eq)} - {int(final_eq)}')
plt.tight_layout()
plt.show()

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

for i, snapi in enumerate(snaps):
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}{extr}_photo{snapi}.txt')
        xph_i, yph_i, zph_i = photo[0], photo[1], photo[2] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_rph[i] = np.mean(rph_i)
        gmean_ph[i] = gmean(rph_i)
        mean_rph_weig[i] = np.sum(rph_i*fluxes[i])/np.sum(fluxes[i])
plt.figure()
plt.plot(tfb, mean_rph/apo, c = 'k')
plt.plot(tfb, mean_rph_weig/apo, c = 'r')
plt.plot(tfb, gmean_ph/apo, c = 'b')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel(r'$\langle R_{ph}\rangle [R_a]$')
plt.yscale('log')
plt.grid()
plt.title(f'Check: {check}')
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
        xph_i, yph_i, zph_i = photo[0], photo[1], photo[2] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        ksL = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        statL[i], pvalueL[i] = ksL.statistic, ksL.pvalue
        # mean 
        mean_rphL[i] = np.mean(rph_i)
        gmean_phL[i] = gmean(rph_i)
        mean_rphL_weig[i] = np.sum(rph_i*fluxesL[i])/np.sum(fluxesL[i])
        time = tfbL[i]
        idx = np.argmin(np.abs(tfb-time))
        diffL.append(np.abs(1-mean_rphL[i]/mean_rph[idx]))
        diffL_weigh.append(np.abs(1-mean_rphL_weig[i]/mean_rph_weig[idx]))


mean_rphH = np.zeros(len(tfbH))
mean_rphH_weig = np.zeros(len(tfbH))
gmean_phH = np.zeros(len(tfbH))
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
        xph_i, yph_i, zph_i = photo[0], photo[1], photo[2] 
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        ksH = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
        statH[i], pvalueH[i] = ksH.statistic, ksH.pvalue
        # mean
        mean_rphH[i] = np.mean(rph_i)
        gmean_phH[i] = gmean(rph_i)
        mean_rphH_weig[i] = np.sum(rph_i*fluxesL[i])/np.sum(fluxesL[i])
        time = tfbH[i]
        idx = np.argmin(np.abs(tfb-time))
        diffH.append(np.abs(1-mean_rphH[i]/mean_rph[idx]))
        diffH_weigh.append(np.abs(1-mean_rphH_weig[i]/mean_rph_weig[idx]))

# Compare the photosphere distribution
chosen_snap = 164
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

#%%
# CDF of rph 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
ax1.plot(rphL_cdf_weig, cum_weigL, alpha = 0.5, color = 'C1', label = 'Low')
ax1.plot(rph_cdf_weig, cum_weig, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
ax1.plot(rphH_cdf_weig, cum_weigH, alpha = 0.5, color = 'darkviolet', label = 'High')
ax1.text(5, 0.6, r'Flux-weighted $R_{ph}$', fontsize = 18)
ax2.plot(rphL_cdf, cumL, alpha = 0.5, color = 'C1', label = 'Low')
ax2.plot(rph_cdf, cum, alpha = 0.5, color = 'yellowgreen', label = 'Fid')
ax2.plot(rphH_cdf, cumH, alpha = 0.5, color = 'darkviolet', label = 'High')
ax1.set_ylim(0.5, 1.1)
for ax in [ax1, ax2]:
        ax.set_ylabel('CDF')
        ax.legend()
        ax.grid()
ax2.set_xlabel(r'$R_{ph}$')
plt.suptitle(f'snap {chosen_snap}', fontsize = 18)

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
ax1.plot(tfb, mean_rph/apo, c = 'yellowgreen', label = 'Fid')
ax1.plot(tfbH, mean_rphH/apo, c = 'darkviolet', label = 'High')
ax1.set_ylabel(r'$\langle R_{ph} \rangle [R_a]$')
ax1.legend(fontsize = 18)

ax2.plot(tfbL, diffL, color = 'darkorange')
ax2.plot(tfbH, diffH, color = 'darkviolet')
ax2.set_xlabel(r't [$t_{fb}$]')
ax2.set_ylabel(r'$|\Delta_{\rm rel}|$')
for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.grid()
ax2.set_xlabel(r't [$t_{fb}$]')
# plt.suptitle('Standard average photosphere')
print('median diffL:', np.median(diffL))
print('median diffH:', np.median(diffH))
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/multiple/Rph_diff.pdf')


# %%