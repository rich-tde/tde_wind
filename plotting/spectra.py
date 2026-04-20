"""Post-processing of spectra:
1. Load the spectra computed in fld_curve.py, 
2. Weight the contribution by cos\theta of each ray to another
3. Plot them.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list
from scipy.interpolate import griddata

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snaps = [76, 109, 151] #[49, 64, 142] #0.4tfb, 0.78, 2.08 fb are snaps 49, 64, 142
x_axis = 'Temp'  # 'Freq' or 'Temp'
# Visible: 4.8e14-7.5e14 Hz  // UV: 7.5e14-3e15 // Xray: 3e15-3e19 Hz (tera:1e12, peta: 1e14, exa: 1e18)
low_freq_optical = 1.6767 * prel.ev_toHz #4.8e14
high_freq_optical = 3.358 * prel.ev_toHz #7.5e14
high_freq_UV = 7.748 * prel.ev_toHz #3e15
low_freq_Xray = 300 * prel.ev_toHz 
high_freq_Xray = 2e4 * prel.ev_toHz #3e19
L_min = 1e39
L_max = 1e42
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

def plot_spectra(folder, check, snaps, x_axis, choice = 'left_right_z'):
    # Load
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
    idx_opt = np.where(np.logical_and(freqs > low_freq_optical, freqs < high_freq_UV))[0][0]
    # idx_UV = np.where(np.logical_and(freqs > high_freq_optical, freqs < high_freq_UV))[0][0]
    idx_Xray = np.where(np.logical_and(freqs > high_freq_UV, freqs < high_freq_Xray))[0][0]

    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps_fld, tfb, Lum_fld = data[:, 0], data[:, 1], data[:, 2]
    snaps_fld, Lum_fld, tfb = sort_list([snaps_fld, Lum_fld, tfb], tfb, unique=True) 
    snaps_fld = snaps_fld.astype(int)

    # observers
    observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz)
    longitude_moll = np.arctan2(observers_xyz[1], observers_xyz[0])
    theta_obs = np.arccos(observers_xyz[2])
    latitude_moll = np.pi/2 - theta_obs
    cross_dot = np.matmul(observers_xyz.T,  observers_xyz)
    cross_dot[cross_dot<0] = 0
    cross_dot /= 192
    indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = choice)

    # For colomesh
    lon_1d = longitude_moll
    lat_1d = latitude_moll
    lon_grid = np.linspace(lon_1d.min(), lon_1d.max(), 360)
    lat_grid = np.linspace(lat_1d.min(), lat_1d.max(), 180)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    fig_moll = plt.figure(figsize=(22,len(snaps)*6))
    gs = gridspec.GridSpec(len(snaps), 2, hspace=0.2, wspace = 0.0)

    for s, snap in enumerate(snaps):
        time = tfb[snaps_fld == snap][0]
        L_photo = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')
        
        ax_op = fig_moll.add_subplot(gs[s, 0], projection='mollweide')
        # ax_uv = fig_moll.add_subplot(gs[s, 1], projection='mollweide')
        ax_x = fig_moll.add_subplot(gs[s, 1], projection='mollweide')

        if s == 0:
            ax_op.set_title("Optical + UV", fontsize=24, y = 1.15)
            ax_x.set_title("X-ray", fontsize=24, y = 1.15)
        # else:
        #     # invisible title to keep the same padding
        #     ax_op.set_title(" ", fontsize=1, y = 1.1)
        #     ax_x.set_title(" ", fontsize=1, y = 1.1)

        for ax in [ax_op,ax_x]:
            ax.set_xticks(np.radians(np.arange(-180, 181, 90))) 
            ax.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
            ax.set_yticks(np.radians(np.arange(-90, 91, 45))) 
            ax.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])

        Lum_op, Lum_UV, Lum_Xray = np.zeros(len(L_photo)), np.zeros(len(L_photo)), np.zeros(len(L_photo))
        for i in range(len(L_photo)):
            Lum_freq = freqs * L_photo[i]
            Lum_op[i] = np.sum(Lum_freq[idx_opt])
            # Lum_UV[i] = np.sum(Lum_freq[idx_UV])
            Lum_Xray[i] = np.sum(Lum_freq[idx_Xray])

        data_grid_op = griddata(
        points=(lon_1d, lat_1d),
        values=Lum_op,
        xi=(lon_mesh, lat_mesh),
        method='linear')
        ax_op.pcolormesh(lon_mesh, lat_mesh, data_grid_op, cmap = 'rainbow', norm=colors.LogNorm(vmin=L_min, vmax=L_max))

        # data_grid_uv = griddata(
        # points=(lon_1d, lat_1d),
        # values=Lum_UV,
        # xi=(lon_mesh, lat_mesh),
        # method='linear')
        # ax_uv.pcolormesh(lon_mesh, lat_mesh, data_grid_uv, cmap = 'rainbow', norm=colors.LogNorm(vmin=L_min, vmax=L_max))
    
        data_grid_x = griddata(
        points=(lon_1d, lat_1d),
        values=Lum_Xray,
        xi=(lon_mesh, lat_mesh),
        method='linear')
        img = ax_x.pcolormesh(lon_mesh, lat_mesh, data_grid_x, cmap = 'rainbow', norm=colors.LogNorm(vmin=L_min, vmax=L_max))
        
        L_photo = np.matmul(cross_dot, L_photo)
        fig_sp, ax = plt.subplots(1, 1, figsize=(8,6))
        if x_axis == 'Temp':
            x_value = freqs * prel.Hz_toK
            ax.set_xlabel('Temperature [K]')
            ax.set_xlim(1e3, 5e7)
        else:
            x_value = freqs
            ax.set_xlabel('Frequency [Hz]')
            ax.set_xlim(1e14, 1e19)
        for i_idx, idx in enumerate(indices_sorted):
            if len(idx) == 1:
                Lum = np.concatenate(L_photo[idx])
            else:
                Lum = np.median(L_photo[idx], axis = 0)
            ax.plot(x_value, freqs * Lum, label = f'{label_obs[i_idx]}', c = colors_obs[i_idx], ls = lines_obs[i_idx])
                        
        ax.tick_params(axis='both', which='major', length=8, width=1.2)
        ax.tick_params(axis='both', which='minor', length=5, width=1)
        ax.loglog()
        ax.set_ylim(L_min, L_max)
        ax.set_ylabel(r'$\nu L_{\nu}$ [erg s$^{-1}$]')
        ax.legend(fontsize=16)
        ax.set_title(f'SED at t = {np.round(time, 2)}' + r't$_{\rm fb}$', fontsize=20)
        
        plt.tight_layout()
    
    cbar = fig_moll.colorbar(img, ax=[ax_op, ax_x],  label =r'$\nu L_\nu$ [erg s$^{-1}$]', orientation='horizontal', aspect=45, pad=0.08)
    cbar.ax.tick_params(which='major',length = 6)
    cbar.ax.tick_params(which='minor',length = 4)
    fig_moll.subplots_adjust(top=0.90, bottom=0.2, left=0.06, right=0.96)
    # fig_moll.tight_layout()

plot_spectra(folder, check, snaps, x_axis)

# indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = 'left_right_in_out_z')

# fig, ax = plt.subplots(1, 1, figsize=(8,6))
# F_mean = []
# for i, idx_list in enumerate(indices_sorted):
#     F_mean.append(np.mean(L_photo[idx_list], axis=0))

# for idx, lab in enumerate(label_obs):
#     if x_axis == 'Freq':
#         ax.plot(freqs, freqs * F_mean[idx], c = colors_obs[idx], label = lab)
#         ax.set_xlabel('Frequency [Hz]')
#         ax.set_xlim(1e14, 1e19)
#     elif x_axis == 'Temp':  
#         ax.plot(Temp, freqs * F_mean[idx], c = colors_obs[idx], label = lab)
#         ax.set_xlabel('Temperature [K]')
#         ax.set_xlim(1e3, 1e8)
# ax.tick_params(axis='both', which='major', length=8, width=1.2)
# ax.tick_params(axis='both', which='minor', length=5, width=1)
# ax.loglog()
# ax.set_ylim(1e38, 1e43)
# ax.set_ylabel(r'$\nu F_{\nu}$ [erg s$^{-1}$ cm$^{-2}$]')
# ax.legend()
# ax.set_title(f't = {time:.2f}'+ r' t$_{\rm fb}$', fontsize=20)
# plt.tight_layout()


