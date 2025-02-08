abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(f'{abspath}')
import numpy as np
import matplotlib.pyplot as plt
# import colorcet
import matplotlib.gridspec as gridspec
import healpy as hp
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import Utilities.prelude as prel
import src.orbits as orb

#%%
# PARAMETERS
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 
compton = 'Compton'


Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
DeltaE = orb.energy_mb(Rstar, mstar, Mbh, G=1) # energy of the mb debris 
DeltaE_cgs = DeltaE * prel.en_converter/prel.Msol_cgs
a = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
ph_data = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}/photo_mean.txt')
tfbRph, Rph = ph_data[0], ph_data[3]
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Enden_norm_single = Ledd / (4 * np.pi * prel.c_cgs * (Rph*prel.Rsol_cgs)**2) # [erg/cm^3] 
first_eq = 88 # observer eq. plane
final_eq = 104 #observer eq. plane

#%%
## DECISIONS
##
save = True
res0 = 'LowRes'
res1 = '' #'', 'HiRes', 'DoubleRad', 'LowRes'
res2 = 'HiRes' 

#
## DATA
#

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'{abspath}/data/{folder}'
#Res0 data
datares0 = np.load(f'{path}{res0}/colormapE_Alice/coloredEall_{res0}_thresh.npy') #shape (3, len(tfb), len(radii))
Radres0_267 = np.load(f'{path}{res0}/colormapE_Alice/coloredERadRph_{res0}267.npy')
tfb_datares0 = np.loadtxt(f'{path}{res0}/colormapE_Alice/coloredEall_{res0}_days.txt')
snap_res0 = tfb_datares0[0]
tfb_res0 = tfb_datares0[1]
radiires0 = np.load(f'{path}{res0}/colormapE_Alice/coloredEall_{res0}_radii.npy')
col_ieres0, col_orb_enres0, col_Radres0 = datares0[0], datares0[1], datares0[2]
# convert to cgs
col_ieres0 *= prel.en_converter
col_orb_enres0 *= prel.en_converter
col_Radres0 *= prel.en_converter
abs_col_orb_enres0 = np.abs(col_orb_enres0)
Etot0 = col_ieres0 + col_orb_enres0 + col_Radres0

#%% Res1 data
datares1 = np.load(f'{path}{res1}/colormapE_Alice/coloredEall_{res1}_thresh.npy') #shape (3, len(tfb), len(radii))
Radres1_267 = np.load(f'{path}{res1}/colormapE_Alice/coloredERadRph_{res1}267.npy')
tfb_datares1 = np.loadtxt(f'{path}{res1}/colormapE_Alice/coloredEall_{res1}_days.txt')
snap_res1 = tfb_datares1[0]
tfb_res1 = tfb_datares1[1]
radiires1 = np.load(f'{path}{res1}/colormapE_Alice/coloredEall_{res1}_radii.npy')
col_ieres1, col_orb_enres1, col_Radres1 = datares1[0], datares1[1], datares1[2]
# convert to cgs
col_ieres1 *= prel.en_converter
col_orb_enres1 *= prel.en_converter
abs_col_orb_enres1 = np.abs(col_orb_enres1)
col_Radres1 *= prel.en_converter
Etot1 = col_ieres1 + col_orb_enres1 + col_Radres1

#%% Res2 data
datares2 = np.load(f'{path}{res2}/colormapE_Alice/coloredEall_{res2}_thresh.npy')
Radres2_267 = np.load(f'{path}{res2}/colormapE_Alice/coloredERadRph_{res2}267.npy')
tfb_datares2 = np.loadtxt(f'{path}{res2}/colormapE_Alice/coloredEall_{res2}_days.txt')
snap_res2 = tfb_datares2[0]
tfb_res2 = tfb_datares2[1]
radiires2 = np.load(f'{path}{res2}/colormapE_Alice/coloredEall_{res2}_radii.npy')
col_ieres2, col_orb_enres2, col_Radres2 = datares2[0], datares2[1], datares2[2]
# convert to cgs
col_ieres2 *= prel.en_converter
col_orb_enres2 *= prel.en_converter
col_Radres2 *= prel.en_converter
abs_col_orb_enres2 = np.abs(col_orb_enres2)
Etot2 = col_ieres2 + col_orb_enres2 + col_Radres2

# relative difference L and fid
rel_orb0 = np.zeros(len(tfb_res0))
rel_ie0 = np.zeros(len(tfb_res0))
rel_rad0 = np.zeros(len(tfb_res0))
rel_Etot0 = np.zeros(len(tfb_res0))
for i in range(len(tfb_res0)):
    # find the comparable time in res1
    time = tfb_res0[i]
    idx = np.argmin(np.abs(tfb_res1 - time))
    # compute the relative difference 
    rel_orb0[i] = 2 * np.abs((col_orb_enres1[idx] - col_orb_enres0[i]) / (col_orb_enres1[idx] + col_orb_enres0[i]))
    rel_ie0[i] = 2 * np.abs((col_ieres1[idx] - col_ieres0[i]) / (col_ieres1[idx] + col_ieres0[i]))
    rel_rad0[i] = 2 * np.abs((col_Radres1[idx] - col_Radres0[i]) / (col_Radres1[idx] + col_Radres0[i]))
    rel_Etot0[i] = 2 * np.abs((Etot1[idx] - Etot0[i]) / (Etot1[idx] + Etot0[i]))

# relative difference H and fid
rel_orb2 = np.zeros(len(tfb_res2))
rel_ie2 = np.zeros(len(tfb_res2))
rel_rad2 = np.zeros(len(tfb_res2))
rel_Etot2 = np.zeros(len(tfb_res2))
for i in range(len(tfb_res2)):
    # find the comparable time in res1
    time = tfb_res2[i]
    idx = np.argmin(np.abs(tfb_res1 - time))
    # compute the relative difference 
    rel_orb2[i] = 2*np.abs((abs_col_orb_enres1[idx] - abs_col_orb_enres2[i]) / (abs_col_orb_enres1[idx] + abs_col_orb_enres2[i]))
    rel_ie2[i] = 2*np.abs((col_ieres1[idx] - col_ieres2[i]) / (col_ieres1[idx] + col_ieres2[i]))
    rel_rad2[i] = 2*np.abs((col_Radres1[idx] - col_Radres2[i]) / (col_Radres1[idx] + col_Radres2[i]))
    rel_Etot2[i] = 2*np.abs((Etot1[idx] - Etot2[i]) / (Etot1[idx] + Etot2[i]))

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
ax1.plot(tfb_res0, abs_col_orb_enres0, label = r'Low', c = 'darkorange')
ax1.plot(tfb_res1, abs_col_orb_enres1, label = r'Fid', c = 'yellowgreen')
ax1.plot(tfb_res2, abs_col_orb_enres2, label = r'High', c = 'darkviolet')
ax1.set_ylabel(r'$|OE|$', fontsize = 18)
ax1.legend(fontsize = 15)

ax2.plot(tfb_res0, col_ieres0, label = r'Low', c = 'darkorange')
ax2.plot(tfb_res1, col_ieres1, label = r'Fid', c = 'yellowgreen')
ax2.plot(tfb_res2, col_ieres2, label = r'High', c = 'darkviolet')
ax2.set_ylabel(r'Internal energy', fontsize = 18)

ax3.plot(tfb_res0, col_Radres0, label = r'Low', c = 'darkorange')
ax3.plot(tfb_res1, col_Radres1, label = r'Fid', c = 'yellowgreen')
ax3.plot(tfb_res2, col_Radres2, label = r'High', c = 'darkviolet')
ax3.set_ylabel(r'Radiation energy', fontsize = 18)

ax4.plot(tfb_res0, np.abs(Etot0), label = r'Low', c = 'darkorange')
ax4.plot(tfb_res1, np.abs(Etot1), label = r'Fid', c = 'yellowgreen')
ax4.plot(tfb_res2, np.abs(Etot2), label = r'High', c = 'darkviolet')
ax4.set_xlabel(r'$t [t_{\rm fb}]$', fontsize = 22)
ax4.set_ylabel(r'$|$Total energy$|$', fontsize = 18)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_yscale('log')
    ax.grid()
#%% difference in the relative difference
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
ax1.plot(tfb_res0, rel_orb0, '--', label = r'Low and Fid', c = 'darkorange')
ax1.plot(tfb_res2, rel_orb2, label = r'Fid and High', c = 'darkviolet')
ax1.set_ylabel(r'$\Delta_{\rm rel, oe}$', fontsize = 22)

ax2.plot(tfb_res0, rel_ie0, '--', label = r'Low and Fid', c = 'darkorange')
ax2.plot(tfb_res2, rel_ie2, label = r'Fid and High', c = 'darkviolet')
ax2.legend(fontsize = 15)
ax2.set_ylabel(r'$\Delta_{\rm rel, ie}$', fontsize = 22)

ax3.plot(tfb_res0, rel_rad0, '--', label = r'Low and Fid', c = 'darkorange')
ax3.plot(tfb_res2, rel_rad2, label = r'Fid and High', c = 'darkviolet')
ax3.set_ylabel(r'$\Delta_{\rm rel, rad}$', fontsize = 22)

ax4.plot(tfb_res0, rel_Etot0, '--', label = r'Low and Fid', c = 'darkorange')
ax4.plot(tfb_res2, rel_Etot2, label = r'Fid and High', c = 'darkviolet')
ax4.set_xlabel(r'$t [t_{\rm fb}]$', fontsize = 22)
ax4.set_ylabel(r'$\Delta_{\rm rel, tot}$', fontsize = 22)

# Get the existing ticks on the x-axis
original_ticks = ax1.get_yticks()
# Calculate midpoints between each pair of ticks
# midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
# Combine the original ticks and midpoints
# new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid()
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1)
ax3.set_ylim(1e-2, 3)
plt.tick_params(axis = 'both', which = 'both', direction='in', labelsize = 20)
plt.tight_layout()
# if save:
#     plt.savefig(f'{abspath}/Figs/multiple/orbE_IE_relative_diff.pdf')


#%%
photo0 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res0}/photo/{res0}_photo267.txt')
# xph0, yph0, zph0, Radres0_267 = photo0[0], photo0[1], photo0[2], photo0[6]
xph0, yph0, zph0  = photo0[0], photo0[1], photo0[2]
rph0 = np.sqrt(xph0**2 + yph0**2 + zph0**2)
photo1 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res1}/photo/rich_photo267.txt')
# xph1, yph1, zph1, Radres1_267 = photo1[0], photo1[1], photo1[2], photo1[6]
xph1, yph1, zph1 = photo1[0], photo1[1], photo1[2]
rph1 = np.sqrt(xph1**2 + yph1**2 + zph1**2)
photo2 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res2}/photo/{res2}_photo267.txt')
# xph2, yph2, zph2, Radres2_267 = photo2[0], photo2[1], photo2[2], photo2[6]
xph2, yph2, zph2 = photo2[0], photo2[1], photo2[2]
rph2 = np.sqrt(xph2**2 + yph2**2 + zph2**2)
rphmax, rphmin = np.max([rph0, rph1, rph2]), np.min([rph0, rph1, rph2])
rphall = np.concatenate([rph0, rph1, rph2])

diffRad267_0 = 2 * np.abs(Radres0_267 - Radres1_267)/(Radres0_267 + Radres1_267)
diffRad267_2 = 2 * np.abs(Radres1_267 - Radres2_267)/(Radres1_267 + Radres2_267)
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
ax1.plot(radiires0/apo, Radres0_267*prel.en_converter, label = 'Low', c = 'darkorange')
ax1.plot(radiires1/apo, Radres1_267*prel.en_converter, label = 'Fid', c = 'yellowgreen')
ax1.plot(radiires2/apo, Radres2_267*prel.en_converter, label = 'High', c = 'darkviolet')
ax1.set_ylabel(r'Total radiation energy [erg]', fontsize = 22)

ax2.plot(radiires0/apo, diffRad267_0, label = 'Low-Fid', c = 'darkorange')
ax2.plot(radiires1/apo, diffRad267_2, label = 'Fid-High', c = 'darkviolet')
ax2.set_xlabel(r'$R [R_{\rm a}]$', fontsize = 22)
ax2.set_ylabel(r'$\Delta_{\rm rel, rad}$', fontsize = 22)
for ax in [ax1, ax2]:
    ax.grid()
    ax.loglog()
    ax.axvline(np.max(rph0)/apo, c = 'darkorange', linestyle = 'dashed')
    ax.axvline(np.min(rph0)/apo, c = 'darkorange', linestyle = 'dashed')
    ax.axvline(np.max(rph1)/apo, c = 'yellowgreen', linestyle = 'dashed')
    ax.axvline(np.min(rph1)/apo, c = 'yellowgreen', linestyle = 'dashed')
    ax.axvline(np.max(rph2)/apo, c = 'darkviolet', linestyle = 'dashed')
    ax.axvline(np.min(rph2)/apo, c = 'darkviolet', linestyle = 'dashed')
    ax.axvspan(rphmin/apo, rphmax/apo, color = 'k', alpha = 0.2)
    ax.legend(fontsize = 15)
    # for r in rphall:
    #     ax.axvline(r/apo, c = 'dodgerblue')

# %% Just the radiation energy outside photosphere
# En_tillph0 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res0}/convEn{res0}_267.txt')
# En_tillph0 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res0}/convEnNOcut{res0}_267.txt')
En_tillph0 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res0}/convEn{res0}nouvrich_267.txt')
Rad_tillph0, IE_tillph0, OE_tillph0, Rad_denphot0 = En_tillph0[0], En_tillph0[1], En_tillph0[2], En_tillph0[3]
Rad_tillph0 *= prel.en_converter
IE_tillph0 *= prel.en_converter
OE_tillph0 *= prel.en_converter
Rad_denphot0 *= prel.en_den_converter
# En_tillph1 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res1}/convEn{res1}_267.txt')
# En_tillph1 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res1}/convEnNOcut{res1}_267.txt')
En_tillph1 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res1}/convEn{res1}nouvrich_267.txt')
Rad_tillph1, IE_tillph1, OE_tillph1, Rad_denphot1 = En_tillph1[0], En_tillph1[1], En_tillph1[2], En_tillph1[3]
Rad_tillph1 *= prel.en_converter
IE_tillph1 *= prel.en_converter
OE_tillph1 *= prel.en_converter
Rad_denphot1 *= prel.en_den_converter
En_tillph2 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}{res2}/convEn{res2}_267.txt')
Rad_tillph2, IE_tillph2, OE_tillph2, Rad_denphot2 = En_tillph2[0], En_tillph2[1], En_tillph2[2], En_tillph2[3]
Rad_tillph2 *= prel.en_converter
IE_tillph2 *= prel.en_converter
OE_tillph2 *= prel.en_converter
Rad_denphot2 *= prel.en_den_converter
rel_diff0 = 2 * np.abs((Rad_tillph1 - Rad_tillph0) / (Rad_tillph1 + Rad_tillph0))
rel_diff2 = 2 * np.abs((Rad_tillph1 - Rad_tillph2) / (Rad_tillph1 + Rad_tillph2))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
ax1.plot(np.abs(OE_tillph0), label = r'Low', c = 'darkorange')
ax1.plot(np.abs(OE_tillph1), label = r'Fid', c = 'yellowgreen')
ax1.plot(np.abs(OE_tillph2), label = r'High', c = 'darkviolet')
ax1.set_ylabel(r'$|OE|$', fontsize = 18)
ax1.legend(fontsize = 15)

ax2.plot(IE_tillph0, label = r'Low', c = 'darkorange')
ax2.plot(IE_tillph1, label = r'Fid', c = 'yellowgreen')
ax2.plot(IE_tillph2, label = r'High', c = 'darkviolet')
ax2.set_ylabel(r'Internal energy', fontsize = 18)

ax3.plot(Rad_tillph0, label = r'Low', c = 'darkorange')
ax3.plot(Rad_tillph1, label = r'Fid', c = 'yellowgreen')
ax3.plot(Rad_tillph2, label = r'High', c = 'darkviolet')
ax3.set_ylabel(r'Radiation energy', fontsize = 18)
ax3.set_xlabel(r'observers', fontsize = 22)

for ax in [ax1, ax2, ax3]:
    ax.set_yscale('log')
    ax.axvline(first_eq, c = 'dodgerblue')  
    ax.axvline(final_eq, c = 'dodgerblue')
    ax.axvspan(first_eq, final_eq, color = 'dodgerblue', alpha = 0.2)
    ax.grid()

# %% Just radiation
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 10))
ax3.plot(Rad_tillph0, label = r'Low', c = 'darkorange')
ax3.plot(Rad_tillph1, label = r'Fid', c = 'yellowgreen')
ax3.plot(Rad_tillph2, label = r'High', c = 'darkviolet')
ax3.set_ylabel(r'Radiation energy', fontsize = 18)

ax4.plot(Rad_denphot0, label = r'Low', c = 'darkorange')
ax4.plot(Rad_denphot1, label = r'Fid', c = 'yellowgreen')
ax4.plot(Rad_denphot2, label = r'High', c = 'darkviolet')
ax4.set_ylabel(r'Rad en density', fontsize = 18)
ax4.set_xlabel(r'observers', fontsize = 22)

for ax in [ax3, ax4]:
    ax.set_yscale('log')
    ax.axvline(first_eq, c = 'dodgerblue')  
    ax.axvline(final_eq, c = 'dodgerblue')
    ax.axvspan(first_eq, final_eq, color = 'dodgerblue', alpha = 0.2)
    ax.grid()
# %% Check the observers on the sphere
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
# Convert to latitude and longitude
longitude = theta              
latitude = np.pi / 2 - phi 
#
# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(longitude, latitude, s=20, c = np.arange(192), cmap='viridis')
plt.colorbar(img, ax=ax, label=r'Observers number')
ax.grid(True)

# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(1,2, figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax[0].scatter(longitude, latitude, s=20, c = rel_diff0, cmap='viridis', norm=colors.LogNorm(5e-1, 1.8))
plt.colorbar(img, ax=ax[0], label=r'$\Delta_{\rm rel}$ Rad energy Low-Fid', orientation='horizontal')

img = ax[1].scatter(longitude, latitude, s=20, c = rel_diff2, cmap='viridis', norm=colors.LogNorm(1e-1, 1.8))
plt.colorbar(img, ax=ax[1], label=r'$\Delta_{\rm rel}$ Rad energy Fid-High', orientation='horizontal')
for i in range(2):
    ax[i].grid(True)
    ax[i].set_xticks(np.radians(np.linspace(-180, 180, 9)))
    ax[i].set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'], fontsize=10)
    # smaller ylabels
    ax[i].set_yticks(np.radians(np.linspace(-90, 90, 9)))
    ax[i].set_yticklabels(['-90°', '-60', '-45°', '-30°', '0°', '30°', '45°', '60°', '90°'], fontsize=10)
plt.tight_layout()
plt.show()
# %%
