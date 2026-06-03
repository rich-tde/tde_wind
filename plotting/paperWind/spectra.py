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
import src.orbits as orb

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
snaps = [76, 109, 151] #[49, 64, 142] #0.4tfb, 0.78, 2.08 fb are snaps 49, 64, 142
x_axis = 'Temp'  # 'Freq' or 'Temp'
choice = 'left_right_z' #'chunky_axes' #left_right_z' 
snaps_MG = np.concatenate([np.arange(107, 386), np.arange(387, 430)]) # I don't have snap386

# Visible: 4.8e14-7.5e14 Hz  // UV: 7.5e14-3e15 // Xray: 3e15-3e19 Hz (tera:1e12, peta: 1e14, exa: 1e18)
low_freq_optical = 1.6767 * prel.ev_toHz #4.8e14
high_freq_optical = 3.358 * prel.ev_toHz #7.5e14
high_freq_UV = 7.7488 * prel.ev_toHz #3e15
low_freq_Xray = 300 * prel.ev_toHz 
high_freq_Xray = 2e4 * prel.ev_toHz #3e19
L_min = 1e37
L_max = 1.1e42
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

def plot_spectra(folder, check, snaps, x_axis, choice):
    # Load
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectraNEW/freqs.txt')
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
    longitude_moll = np.arctan2(observers_xyz[1], observers_xyz[0]) # from -pi to pi, 0 at x axis, positive towards y axis
    theta_obs = np.arccos(observers_xyz[2]) # from 0 (+z axis) to pi (-z axis)
    latitude_moll = np.pi/2 - theta_obs  # from np.pi/2 (z axis) to -np.pi/2 (-z axis)
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
    
    fig_mollop = plt.figure(figsize=(len(snaps)*11, 7))
    gs_op = gridspec.GridSpec(2, len(snaps), wspace = 0.1, hspace = 0, height_ratios=[1, 0.08])
    fig_mollx = plt.figure(figsize=(len(snaps)*11, 7))
    gs_x = gridspec.GridSpec(2, len(snaps), wspace = 0.1, hspace = 0, height_ratios=[1, 0.08])

    fig_sp, ax = plt.subplots(1, len(snaps), figsize=(24,7))
    for s, snap in enumerate(snaps):
        time = tfb[snaps_fld == snap][0]
        L_photo = np.loadtxt(f'{pre_saving}/spectraNEW/{check}_spectra{snap}.txt')
        
        ax_op = fig_mollop.add_subplot(gs_op[0, s], projection='mollweide')
        # ax_uv = fig_moll.add_subplot(gs_uv[0, s], projection='mollweide')
        ax_x = fig_mollx.add_subplot(gs_x[0, s], projection='mollweide')

        ax_op.set_title(f'{np.round(time, 2)}' + r' t$_{\rm fb}$', fontsize=24, y = 1.15) 
        ax_x.set_title(f'{np.round(time, 2)}' + r' t$_{\rm fb}$', fontsize=24, y = 1.15)
        # else: 
        #     # invisible title to keep the same padding
        #     ax_op.set_title(" ", fontsize=1, y = 1.1)
        #     ax_x.set_title(" ", fontsize=1, y = 1.1)
  
        for ax_moll in [ax_op, ax_x]:
            ax_moll.set_xticks(np.radians(np.arange(-180, 181, 90))) 
            ax_moll.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
            ax_moll.set_yticks(np.radians(np.arange(-90, 91, 45))) 
            ax_moll.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])

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
        ax_op.pcolormesh(lon_mesh, lat_mesh, data_grid_op, cmap = 'rainbow', norm=colors.LogNorm(vmin=L_min, vmax=10*L_max))

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
        img = ax_x.pcolormesh(lon_mesh, lat_mesh, data_grid_x, cmap = 'rainbow', norm=colors.LogNorm(vmin=L_min, vmax=10*L_max))
        
        L_photo = np.matmul(cross_dot, L_photo)
        if x_axis == 'Temp':
            x_value = freqs * prel.Hz_toK
            ax[s].set_xlabel('Temperature (K)', fontsize = 30)
            ax[s].axvline(low_freq_optical * prel.Hz_toK, c = 'bisque', linewidth = .2)
            ax[s].axvspan(low_freq_optical * prel.Hz_toK, high_freq_optical * prel.Hz_toK, color='bisque', alpha=0.3)
            ax[s].axvline(high_freq_optical * prel.Hz_toK, c = 'lightsteelblue', linewidth = .2)
            ax[s].axvspan(high_freq_optical * prel.Hz_toK, high_freq_UV * prel.Hz_toK, color='orchid', alpha=0.2)
            ax[s].axvline(high_freq_UV * prel.Hz_toK, c = 'lightcoral', linewidth = .2)
            ax[s].axvline(high_freq_Xray * prel.Hz_toK, c = 'lightcoral', linewidth = .2)
            ax[s].axvspan(high_freq_UV * prel.Hz_toK, high_freq_Xray * prel.Hz_toK, color='lightsteelblue', alpha=0.2)
            ax[s].set_xlim(1e3, 1e7)
            if s == 0:
                ax[s].text(0.6 * high_freq_optical * prel.Hz_toK, L_max/10, 'Optical', rotation=90, fontsize=20)
                ax[s].text(0.6 * high_freq_UV * prel.Hz_toK, L_max/10, 'UV', rotation=90, fontsize=20)
                ax[s].text(1.5 * high_freq_UV * prel.Hz_toK, L_max/10, 'X-ray', rotation=90, fontsize=20)
        else:
            x_value = freqs
            ax[s].axvline(low_freq_optical, c = 'k')
            ax[s].axvspan(low_freq_optical, high_freq_optical, color='bisque', alpha=0.2)
            ax[s].axvline(high_freq_UV, c = 'k')
            # ax[s].axvline(low_freq_Xray, c = 'k')
            ax[s].axvline(high_freq_Xray, c = 'k')
            ax[s].set_xlabel('Frequency (Hz)', fontsize = 30)
            ax[s].set_xlim(1e14, 1e19)
        for i_idx, idx in enumerate(indices_sorted):
            if i_idx == 3:
                continue
            if len(idx) == 1:
                Lum = np.concatenate(L_photo[idx])
            else:
                Lum = np.median(L_photo[idx], axis = 0)
            ax[s].plot(x_value, freqs * Lum, label = f'{label_obs[i_idx]}', c = colors_obs[i_idx], linewidth = 3) #ls = lines_obs[i_idx]
                        
        ax[s].tick_params(axis='both', which='major', length=8, width=1.2)
        ax[s].tick_params(axis='both', which='minor', length=5, width=1)
        ax[s].loglog()
        ax[s].set_ylim(L_min, L_max)
        ax[s].set_title(f't = {np.round(time, 1)}' + r't$_{\rm fb}$', fontsize = 30)
        
    ax[0].set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)
    ax[0].legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/2.paperWind/spectra_{choice}.pdf', dpi=300)
    
    cbar_ax = fig_mollop.add_subplot(gs_op[1, 0:3])  # Colorbar subplot below the first two
    cb = fig_mollop.colorbar(img, cax=cbar_ax, orientation='horizontal', pad=0.07)
    cb.set_label(r'$\nu L_\nu$ [erg s$^{-1}$]')
    cb.ax.tick_params(which='major',length = 10)
    cb.ax.tick_params(which='minor',length = 6) 
    fig_mollop.suptitle("Optical + UV", fontsize=24) 
    cbar_ax = fig_mollx.add_subplot(gs_op[1, 0:3])  # Colorbar subplot below the first two
    cb = fig_mollx.colorbar(img, cax=cbar_ax, orientation='horizontal', pad=0.07)
    cb.set_label(r'$\nu L_\nu$ [erg s$^{-1}$]')
    cb.ax.tick_params(which='major',length = 10)
    cb.ax.tick_params(which='minor',length = 6) 
    fig_mollx.suptitle("X-ray", fontsize=24)
    # fig_moll.subplots_adjust(top=0.90, bottom=0.2, left=0.06, right=0.96)
    fig_mollop.tight_layout()

def plot_light_curves(folder, check, choice, group = 'bands'):
    # Load
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectraNEW/freqs.txt')
    idx_opt = np.where(np.logical_and(freqs > low_freq_optical, freqs < high_freq_optical))[0][0]
    idx_UV = np.where(np.logical_and(freqs > high_freq_optical, freqs < high_freq_UV))[0][0]
    idx_Xray = np.where(np.logical_and(freqs > high_freq_UV, freqs < high_freq_Xray))[0][0]

    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps_fld, tfb, Lum_fld = data[:, 0], data[:, 1], data[:, 2]
    snaps_fld, Lum_fld, tfb = sort_list([snaps_fld, Lum_fld, tfb], tfb, unique=True) 
    snaps_fld = snaps_fld.astype(int)

    # observers
    observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz)
    indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = choice)

    Lum_op_sum = []
    Lum_UV_sum = []
    Lum_Xray_sum = []
    time_spec = []
    for s, snap in enumerate(snaps_fld):
        try:
            L_photo = np.loadtxt(f'{pre_saving}/spectraNEW/{check}_spectra{snap}.txt')
            time_spec.append(tfb[snaps_fld == snap][0])
        except:
            continue
        Lum_op, Lum_UV, Lum_Xray = np.zeros(len(L_photo)), np.zeros(len(L_photo)), np.zeros(len(L_photo))
        for i in range(len(L_photo)):
            Lum_freq = freqs * L_photo[i]
            Lum_op[i] = np.sum(Lum_freq[idx_opt])
            Lum_UV[i] = np.sum(Lum_freq[idx_UV])
            Lum_Xray[i] = np.sum(Lum_freq[idx_Xray])

        Lum_op_sum.append(np.mean(Lum_op[indices_sorted], axis = 1))
        Lum_UV_sum.append(np.mean(Lum_UV[indices_sorted], axis = 1))
        Lum_Xray_sum.append(np.mean(Lum_Xray[indices_sorted], axis = 1))
    Lum_op_sum = np.transpose(np.array(Lum_op_sum))
    Lum_UV_sum = np.transpose(np.array(Lum_UV_sum))
    Lum_Xray_sum = np.transpose(np.array(Lum_Xray_sum))

    Lum_op_MG = []
    Lum_UV_MG = []
    Lum_Xray_MG = []
    time_MG = []
    # observers MG
    nside_mg = 8
    observers_xyz_MG = hp.pix2vec(nside_mg, np.arange(hp.nside2npix(nside_mg))) #shape: (3, 768)
    observers_xyz_MG = np.array(observers_xyz_MG)
    indices_sorted_MG, _, _, _ = choose_observers(observers_xyz_MG, choice = choice)

    for s, snap in enumerate(snaps_MG):
        L_photoMG = np.loadtxt(f'{pre_saving}/MG/snap_{snap}/L_snap_{snap}.txt')
        t_MG = np.loadtxt(f'{pre_saving}/MG/snap_{snap}/Time_snap_{snap}.txt') #it's in CGS
        time_MG.append(t_MG)
        Lum_op_MG_sum = np.sum(L_photoMG[:, 1:3], axis = 1)
        Lum_UV_MG_sum = np.sum(L_photoMG[:, 3:5], axis = 1)
        Lum_Xray_MG_sum = np.sum(L_photoMG[:, 8:], axis = 1)
        # Lum_op_MG_sum = L_photoMG[:, 1]
        # Lum_UV_MG_sum = L_photoMG[:, 3]
        # Lum_Xray_MG_sum = L_photoMG[:, 8]

        Lum_op_MG.append([np.mean(Lum_op_MG_sum[idx]) for idx in indices_sorted_MG])
        Lum_UV_MG.append([np.mean(Lum_UV_MG_sum[idx]) for idx in indices_sorted_MG])
        Lum_Xray_MG.append([np.mean(Lum_Xray_MG_sum[idx]) for idx in indices_sorted_MG])
    time_MG = np.array(time_MG) * prel.tsol_cgs / (3600*24*t_fb_days) # convert to tfb
    # print(time_MG)
    Lum_op_MG = np.transpose(np.array(Lum_op_MG))
    Lum_UV_MG = np.transpose(np.array(Lum_UV_MG))
    Lum_Xray_MG = np.transpose(np.array(Lum_Xray_MG))

    if group == 'sections': # each panel show a spherical sector
        len_plot = len(label_obs) 
        if np.array(label_obs).all() not in ['south pole', r'-$\hat{z}$']:
            print('Do not consider the south pole')
            len_plot -= 1
        fig_L, ax_L = plt.subplots(1, len_plot, figsize=(8*len_plot, 7)) 
        axes = [ax_L[k] for k in range(len_plot)]
        for k, obs in enumerate(np.arange(len_plot)):
            ax_L[k].plot(time_spec, Lum_op_sum[k], label = 'Optical', c = colors_obs[k], linewidth = 3)
            ax_L[k].plot(time_spec, Lum_UV_sum[k], label = f'UV', c = colors_obs[k], linewidth = 3, ls = '--')
            # ax_L[k].plot(time_spec, Lum_Xray_sum[k], label = f'Xray', c = colors_obs[k], linewidth = 3, ls = ':')
            ax_L[k].plot(time_MG, Lum_Xray_MG[k], label = f'Xray', c = colors_obs[k], linewidth = 3, ls = ':')
            ax_L[k].text(0.1, L_max/5, f'{label_obs[k]}', fontsize = 26)
        original_ticks = ax_L[0].get_xticks()
        ax_L[0].legend(fontsize = 25)
        ax_L[0].set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)

    if group == 'bands': # each panel show a band
        fig_L, (ax_op, ax_UV, ax_Xray) = plt.subplots(1, 3, figsize=(24, 7))
        axes = [ax_op, ax_UV, ax_Xray]
        for k, obs in enumerate(label_obs):
            if k == 3:
                continue
            ax_op.plot(time_spec, Lum_op_sum[k], label = f'{obs}', c = colors_obs[k], linewidth = 3)
            ax_op.plot(time_MG, Lum_op_MG[k], c = colors_obs[k], ls = '--', linewidth = 2)
            ax_UV.plot(time_spec, Lum_UV_sum[k], label = f'This work' if k == 2 else None, c = colors_obs[k], linewidth = 3)
            ax_UV.plot(time_MG, Lum_UV_MG[k], label = f'Giron+26' if k == 2 else None, c = colors_obs[k], ls = '--', linewidth = 2)
            ax_Xray.plot(time_spec, Lum_Xray_sum[k], c = colors_obs[k], linewidth = 3)
            ax_Xray.plot(time_MG, Lum_Xray_MG[k], c = colors_obs[k], ls = '--', linewidth = 2)

        ax_op.text(0.1, L_max/5, 'Optical', fontsize = 26)
        ax_UV.text(0.1, L_max/5, 'UV', fontsize = 26)
        ax_Xray.text(0.1, L_max/5, 'X-ray', fontsize = 26)
        original_ticks = ax_op.get_xticks()
        ax_op.legend(fontsize = 22)
        ax_UV.legend(fontsize = 22)
        ax_op.set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)

    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]   
    days_ticks = new_ticks*t_fb_days
    days_labels = [str(np.round(days_ticks[k],2)) if new_ticks[k] in original_ticks else "" for k in range(len(days_ticks))]
    for ax in axes:
        # ax.plot(tfb, Lum_fld, c = 'k', ls = '--', linewidth = 2)
        ax.set_yscale('log')
        ax.set_xticks(new_ticks)
        ax.set_xlabel(r't / t$_{\rm fb}$', fontsize = 30)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='major', width = 1, length = 7, color = 'k')
        ax.tick_params(axis='y', which='minor', width = 1, length = 4, color = 'k')
        ax.grid()
        ax.set_ylim(L_min, L_max)
        ax.set_xlim(-.05, np.max(tfb))

        ax2 = ax.twiny()
        # ax2.axvline(4.5)
        ax2.set_xticks(days_ticks)
        ax2.set_xlim(-0.05*t_fb_days, np.max(tfb)*t_fb_days)
        ax2.set_xticklabels(days_labels)
        ax2.set_xlabel(r't (days)', fontsize = 30)
    
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/2.paperWind/LCs_{choice}_{group}.pdf', dpi=300)

plot_spectra(folder, check, snaps, x_axis, choice)
plot_light_curves(folder, check, choice, group = 'sections')
plot_light_curves(folder, check, choice, group = 'bands')



