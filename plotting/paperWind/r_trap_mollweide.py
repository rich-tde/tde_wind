
import sys

sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list
import src.orbits as orb
from src.Wind.Rtrapp_tdiff import load_and_smooth_rtrap

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps_chosen = [76, 109, 131, 151]
times = [1, 1.5, 2, 2.2]
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
Rt = things['Rt']
which_obs = '3d_arch'

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
indices_obs, label_obs, colors_obs, _ = choose_observers(observers_xyz, which_obs)
markers_obs = ['o', 'p', 's', 'X', 'X']
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfbs, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfbs = sort_list([snaps, Lum, tfbs], tfbs, unique=True) 
snaps = snaps.astype(int)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs

for s, snap in enumerate(snaps):
    if snap < 76 :
        continue
    pathtrap = f"{abspath}/data/{folder}/trap"
    dataRtr = load_and_smooth_rtrap(pathtrap, check, snap)
    kappa_tr, d_tr, Vr_tr = \
        dataRtr['kappa_tr'], dataRtr['den_tr'], dataRtr['Vr_tr']
    kappa_tr = kappa_tr * prel.Rsol_cgs**2 / prel.Msol_cgs

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection='mollweide')
    ax2 = fig.add_subplot(1, 3, 2, projection='mollweide')
    ax3 = fig.add_subplot(1, 3, 3, projection='mollweide')
    for i, idx in enumerate(indices_obs):
        img1 = ax1.scatter(
            longitude_moll[idx],
            latitude_moll[idx],
            c=kappa_tr[idx],
            marker=markers_obs[i],
            s=100,
            cmap='rainbow',
            norm=colors.LogNorm(vmin=3e-1, vmax=40),
            label=label_obs[i])

        img2 = ax2.scatter(
            longitude_moll[idx],
            latitude_moll[idx],
            c=d_tr[idx]*prel.den_converter,
            marker=markers_obs[i],
            s=100,
            cmap='rainbow',
            norm=colors.LogNorm(vmin=2e-13, vmax=1e-9),
            label=label_obs[i])
        
        img3 = ax3.scatter(
            longitude_moll[idx],
            latitude_moll[idx],
            c=Vr_tr[idx]*conversion_sol_kms,
            marker=markers_obs[i],
            s=100,
            cmap='rainbow',
            norm=colors.LogNorm(vmin=1e3, vmax=1e4),
            label=label_obs[i])
        
    cbar1 = fig.colorbar(img1, orientation='horizontal', ax=ax1)
    cbar1.set_label(r'$\kappa$ [cm$^2$/g]')
    cbar1.ax.tick_params(which='major', length=8, width=1.2)
    cbar1.ax.tick_params(which='minor', length=4, width=1)

    cbar2 = fig.colorbar(img2, orientation='horizontal', ax=ax2)
    cbar2.set_label(r'$\rho$ [g/cm$^3$]')
    cbar2.ax.tick_params(which='major', length=8, width=1.2)
    cbar2.ax.tick_params(which='minor', length=4, width=1)

    cbar3 = fig.colorbar(img3, orientation='horizontal', ax=ax3)
    cbar3.set_label(r'$V_r$ [cm/s]')
    cbar3.ax.tick_params(which='major', length=8, width=1.2)
    cbar3.ax.tick_params(which='minor', length=4, width=1)

    for ax in (ax1, ax2, ax3):
        if which_obs == 'funnel' or which_obs == '3d_arch':
            ax.axhline(np.pi/3, color='k', ls='--')
            ax.axhline(-np.pi/3, color='k', ls='--')
            if which_obs == '3d_arch':
                # draw a line at pi6, but ubtile np.pi in x
                ax.plot([-np.pi, -np.pi/2], [np.pi/6, np.pi/6], color='k', ls='--')
                ax.plot([np.pi/2, np.pi], [np.pi/6, np.pi/6], color='k', ls='--')
                ax.plot([-np.pi, -np.pi/2], [-np.pi/6, -np.pi/6], color='k', ls='--')
                ax.plot([np.pi/2, np.pi], [-np.pi/6, -np.pi/6], color='k', ls='--')
                ax.plot([-np.pi/2, -np.pi/2], [-np.pi/3, np.pi/3], color='k', ls='--')
                ax.plot([np.pi/2, np.pi/2], [-np.pi/3, np.pi/3], color='k', ls='--')
        elif which_obs == 'left_right_z':
            ax.plot([-np.pi/2, -np.pi/2], [-np.pi/6, np.pi/6], color='k', ls='--')
            ax.plot([np.pi/2, np.pi/2], [-np.pi/6, np.pi/6], color='k', ls='--')
            ax.plot([-np.pi/2, -np.pi/2], [-np.pi/6, np.pi/6], color='k', ls='--')
            ax.plot([np.pi/2, np.pi/2], [-np.pi/6, np.pi/6], color='k', ls='--')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(True)
        ax.set_xticks(np.radians(np.arange(-180, 181, 90)))
        ytick_labels = ['180°', '135°', '90°', '45°', '0°']
        ax.set_yticks(np.radians(np.arange(-90, 91, 45)))
        ax.set_yticklabels(ytick_labels)
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Shared decorations
    fig.suptitle(f'time = {tfbs[s]:.1f} ' + r't$_{\rm fb}$', fontsize=20)
    fig.tight_layout()
    plt.savefig(f'{abspath}/Figs/{folder}/Wind/Rtr{which_obs}/Rtr_{snap}.png', dpi=300)
    plt.close()