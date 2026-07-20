
import sys

sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list
import src.orbits as orb
from src.Wind.Rtrapp_tdiff import load_and_smooth_rtrap
mpl.rcParams['savefig.transparent'] = False  # Only figure patch
mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
mpl.rcParams['axes.facecolor'] = 'white'      # Axes patch

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
which_obs = 'split_stream'  # 'funnel', '3d_arch', 'split_stream', 'left_right_z'
moll_spl = 'x_posneg'  # 'x_posneg' or '' for standard mollweide

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
indices_obs, label_obs, colors_obs, _ = choose_observers(observers_xyz, which_obs)
markers_obs = ['o', 'p', 's', 'H', 'X', 'X']
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
latitude_moll = np.arcsin(z / r)  
longitude_moll = np.arctan2(y, x)          # +x at 0, +y at pi/2, -x at pi, -y at -pi/2 
if moll_spl == 'x_posneg':
    longitude_moll += np.pi/2  # Shift to have +x at pi/2, +y at pi, -x at -pi/2, -y at 0
    longitude_moll[longitude_moll > np.pi] -= 2*np.pi  # Wrap to [-pi, pi]
        
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfbs, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfbs = sort_list([snaps, Lum, tfbs], tfbs, unique=True) 
snaps = snaps.astype(int)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs

for s, snap in enumerate(snaps):
    if snap != 151:
        continue
    pathtrap = f"{abspath}/data/{folder}/trap"
    dataRtr = load_and_smooth_rtrap(pathtrap, check, snap)
    kappa_tr, d_tr, Vr_tr = \
        dataRtr['kappa_tr'], dataRtr['den_tr'], dataRtr['Vr_tr']
    kappa_tr = kappa_tr * prel.Rsol_cgs**2 / prel.Msol_cgs

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection='mollweide')# if moll_spl == '' else None)
    ax2 = fig.add_subplot(1, 3, 2, projection='mollweide')# if moll_spl == '' else None)
    ax3 = fig.add_subplot(1, 3, 3, projection='mollweide')# if moll_spl == '' else None)
    for i, idx in enumerate(indices_obs):
        # if i != 4:
        #     continue
        img1 = ax1.scatter(
            longitude_moll[idx],
            latitude_moll[idx],
            c=kappa_tr[idx],
            marker=markers_obs[i],
            s=80,
            cmap='rainbow',
            norm=colors.LogNorm(vmin=3e-1, vmax=40),
            label=label_obs[i])

        img2 = ax2.scatter(
            longitude_moll[idx],
            latitude_moll[idx],
            c=d_tr[idx]*prel.den_converter,
            marker=markers_obs[i],
            s=80,
            cmap='rainbow',
            norm=colors.LogNorm(vmin=2e-13, vmax=1e-9),
            label=label_obs[i])
        
        img3 = ax3.scatter(
            longitude_moll[idx],
            latitude_moll[idx],
            c=Vr_tr[idx]*conversion_sol_kms,
            marker=markers_obs[i],
            s=80,
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
        for spine in ax.spines.values():
            spine.set_visible(False)
        if moll_spl == 'x_posneg':
            xtick_labels = ['90°', '180°', '-90°', '0°', '90°']
            ax.set_xticks(np.radians(np.arange(-180, 181, 90)))
            ax.set_xticklabels(xtick_labels)
            # ytick_labels = ['-90°', '-45°', '0°', '45°', '90°']
            # ax.set_yticks(np.radians(np.arange(-90, 91, 45)))
            # ax.set_yticklabels(ytick_labels)
        else:
            ax.set_xticks(np.radians(np.arange(-180, 181, 90)))
        ytick_labels = ['180°', '135°', '90°', '45°', '0°']
        ax.set_yticks(np.radians(np.arange(-90, 91, 45)))
        ax.set_yticklabels(ytick_labels)
        if which_obs in ['funnel', '3d_arch', 'split_stream']:
            if which_obs == 'split_stream':
                lim_plot_lines = np.pi/18
            if which_obs == '3d_arch':
                lim_plot_lines = np.pi/6
            # horizontal lines to separate the funnel region with dashed lines
            ax.axhline(np.pi/3, color='k', ls='--')
            ax.axhline(-np.pi/3, color='k', ls='--')
            if moll_spl == 'x_posneg':
                # vertical line to split stream-pericentre
                axvline = ax.axvline(0, color='k', ls='--')
                # horizontal lines for stream
                if which_obs == 'split_stream':
                    ax.plot([-np.pi, 0], [lim_plot_lines, lim_plot_lines], color='k', ls='--') #10 deg
                    ax.plot([-np.pi, 0], [-lim_plot_lines, -lim_plot_lines], color='k', ls='--')
                elif which_obs == '3d_arch':
                    ax.axhline(lim_plot_lines, color='k', ls='--')
                    ax.axhline(-lim_plot_lines, color='k', ls='--')
            if moll_spl == '':
                # horizontal lines for stream
                if which_obs == 'split_stream':
                    ax.plot([-np.pi, -np.pi/2], [lim_plot_lines, lim_plot_lines], color='k', ls='--') 
                    ax.plot([-np.pi, -np.pi/2], [-lim_plot_lines, -lim_plot_lines], color='k', ls='--')
                    ax.plot([np.pi/2, np.pi], [lim_plot_lines, lim_plot_lines], color='k', ls='--')
                    ax.plot([np.pi/2, np.pi], [-lim_plot_lines, -lim_plot_lines], color='k', ls='--')
                elif which_obs == '3d_arch': 
                    ax.axhline(lim_plot_lines, color='k', ls='--')
                    ax.axhline(-lim_plot_lines, color='k', ls='--')
                # vertical lines for pericentre
                ax.plot([-np.pi/2, -np.pi/2], [-np.pi/3, np.pi/3], color='k', ls='--')
                ax.plot([np.pi/2, np.pi/2], [-np.pi/3, np.pi/3], color='k', ls='--')
        elif which_obs == 'left_right_z':
            # vertical lines to split stream-pericentre
            ax.plot([-np.pi/2, -np.pi/2], [-np.pi/6, np.pi/6], color='k', ls='--')
            ax.plot([np.pi/2, np.pi/2], [-np.pi/6, np.pi/6], color='k', ls='--')
            # horizontal lines toseparate the poles region with dashed lines
            ax.axhline(-np.pi/6, color='k', ls='--')
            ax.axhline(np.pi/6, color='k', ls='--')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True)

    # Shared decorations
    fig.suptitle(f'time = {tfbs[s]:.1f} ' + r't$_{\rm fb}$', fontsize=20)
    fig.tight_layout()
    # plt.savefig(f'{abspath}/Figs/{folder}/Wind/Rtr{which_obs}/Rtr_{snap}.png', dpi=300)
    # plt.close()