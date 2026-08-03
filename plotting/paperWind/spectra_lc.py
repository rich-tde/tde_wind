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
import os
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from lmfit import Model
from matplotlib import lines as mlines
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
snaps_spectra = [76, 109, 151]
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
Rt = things['Rt']
x_axis = 'Temp'  # 'Freq' or 'Temp'
choice = 'split_stream' #

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
# Visible: 4.8e14-7.5e14 Hz  // UV: 7.5e14-3e15 // Xray: 3e15-3e19 Hz (tera:1e12, peta: 1e14, exa: 1e18)
low_freq_optical = 1.6767 * prel.ev_toHz 
high_freq_optical = 3.358 * prel.ev_toHz #7.5e14
high_freq_UV = 7.7488 * prel.ev_toHz #3e15
low_freq_Xray = 300 * prel.ev_toHz 
high_freq_Xray = 2e4 * prel.ev_toHz #3e19
L_min, L_max = 1e38, 1.5e42
T_min, T_max = 1e3, 1e7
tmin, tmax = -0.05, 2.24
nu_min, nu_max = 1e14, 1e19
# Observations expectations 
L_Einstein = 3.5e43 #erg/s
L_eRos = 6e40
L_Rubin = 4e39
L_ZTF = 1.3e41
L_ULTRA = 5e40
# expectations = {'EP/WXT': L_Einstein, 'eROSITA': L_eRos, 'LSST': L_Rubin, 'ZTF': L_ZTF, 'ULTRASAT': L_ULTRA}

def lumfit(n, R, T):
    const = 2*prel.h_cgs/prel.c_cgs**2 
    planck = const * n**3 / (np.exp(prel.h_cgs*n/(prel.Kb_cgs*T))-1)
    Lum = 4 * (np.pi * R)**2 * planck  # L = 4piR^2 * pi * B since pi*B = flux
    return Lum

#%%
def lumtest(n, T):
    const = 2*prel.h_cgs/prel.c_cgs**2 
    planck = const * n**3 / (np.exp(prel.h_cgs*n/(prel.Kb_cgs*T))-1)
    return planck

x = prel.freqs
print(f'Min: {np.min(x):.2e}, Max: {np.max(x):.2e}')
plt.plot(x*prel.Hz_toK, x*lumtest(x, 8e3))
plt.loglog()
plt.ylim(1e5, 1e12)
plt.xlim(1e3, 1e7)
#%%
def plot_spectra(folder, check, snaps, x_axis, choice, in_moll = False):
    # Load
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt') 
    # idx_opt = np.where(np.logical_and(freqs > low_freq_optical, freqs < high_freq_UV))[0]
    # idx_UV = np.where(np.logical_and(freqs > high_freq_optical, freqs < high_freq_UV))[0][0]
    #band in angstrom = 1e7 cm and are wavelenght so the minimum gives the maximum freq
    idx_ztf_g = np.where(np.logical_and(freqs > prel.c_cgs/(prel.ztf_g_band[1]*1e-7), freqs <  prel.c_cgs/(prel.ztf_g_band[0]*1e-7)))[0]
    idx_ztf_r = np.where(np.logical_and(freqs > prel.c_cgs/(prel.ztf_r_band[1]*1e-7), freqs <  prel.c_cgs/(prel.ztf_r_band[0]*1e-7)))[0]
    idx_ztf_i = np.where(np.logical_and(freqs > prel.c_cgs/(prel.ztf_i_band[1]*1e-7), freqs <  prel.c_cgs/(prel.ztf_i_band[0]*1e-7)))[0]
    idx_swift_u_band = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_u_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_u_band[0]*1e-7)))[0]
    idx_swift_b_band = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_b_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_b_band[0]*1e-7)))[0]
    idx_swift_v_band = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_v_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_v_band[0]*1e-7)))[0]
    idx_opt = np.concatenate([idx_ztf_g, idx_ztf_r, idx_ztf_i, idx_swift_u_band, idx_swift_b_band, idx_swift_v_band])

    idx_swift_uvw1 = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_uvw1_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_uvw1_band[0]*1e-7)))[0]
    idx_swift_uvm2 = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_uvm2_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_uvm2_band[0]*1e-7)))[0]
    idx_swift_uvw2 = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_uvw2_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_uvw2_band[0]*1e-7)))[0]
    idx_UV = np.concatenate([idx_swift_uvw1, idx_swift_uvm2, idx_swift_uvw2])
    idx_Xray = np.where(np.logical_and(freqs > high_freq_UV, freqs < high_freq_Xray))[0]
    idx_fit = np.concatenate([idx_UV, idx_opt])
    # idx_fit = np.where(np.logical_and(freqs > 0, freqs < 1e40))[0]

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
    indices_sorted, label_obs, colors_obs, _, _ = choose_observers(observers_xyz, choice = choice)

    if in_moll:
        lon_1d = longitude_moll
        lat_1d = latitude_moll
        lon_grid = np.linspace(lon_1d.min(), lon_1d.max(), 360)
        lat_grid = np.linspace(lat_1d.min(), lat_1d.max(), 180)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        fig_mollop = plt.figure(figsize=(len(snaps)*11, 7))
        gs_op = gridspec.GridSpec(2, len(snaps), wspace = 0.1, hspace = 0, height_ratios=[1, 0.08])
        fig_mollx = plt.figure(figsize=(len(snaps)*11, 7))
        gs_x = gridspec.GridSpec(2, len(snaps), wspace = 0.1, hspace = 0, height_ratios=[1, 0.08])

    fig_sp, ax = plt.subplots(1, len(snaps), figsize=(24,8))
    fig_fit, ax_fit = plt.subplots(2, len(snaps), figsize=(18,8))
    handles_color, labels_color = [], []
    handles_local, labels_local = [], []
    for s, snap in enumerate(snaps):
        time = tfb[snaps_fld == snap][0]
        L_col = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')

        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        Lum_ph = photo['Lum']
        for i in range(len(L_col)):
            norm = Lum_ph[i] / np.trapezoid(L_col[i,:], freqs)
            L_col[i,:] *= norm
        
        if in_moll:
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

            Lum_op, Lum_UV, Lum_Xray = np.zeros(len(L_col)), np.zeros(len(L_col)), np.zeros(len(L_col))
            for i in range(len(L_col)):
                Lum_op[i] = np.trapezoid(L_col[i,idx_opt], freqs[idx_opt]) 
                Lum_UV[i] = np.trapezoid(L_col[i,idx_UV], freqs[idx_UV])
                Lum_Xray[i] = np.trapezoid(L_col[i,idx_Xray], freqs[idx_Xray])

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
            img = ax_x.pcolormesh(lon_mesh, lat_mesh, data_grid_x, cmap = 'rainbow', norm=colors.LogNorm(vmin=L_min, vmax=4*L_max))

            # for i_idx, idx in enumerate(indices_sorted):
            #     Lum_op_i = np.mean(Lum_op[idx])
            #     Lum_UV_i = np.mean(Lum_UV[idx])
            #     Lum_Xray_i = np.mean(Lum_Xray[idx])  
            #     print(f'At t = {np.round(time, 1)}' + r't$_{\rm fb}$' + f'Observer {label_obs[i_idx]}: ratio Xray/opt: {Lum_Xray_i/Lum_op_i:.2e}, ratio opt/UV: {Lum_op_i/Lum_UV_i:.2e}')   

        L_col = np.matmul(cross_dot, L_col)
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
            ax[s].set_xlim(T_min, T_max)
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
            ax[s].set_xlim(nu_min, nu_max)
        
        Rfit = np.zeros(len(indices_sorted))
        Tfit = np.zeros(len(indices_sorted))
        for i_idx, idx in enumerate(indices_sorted):
            # if i_idx > 8:
            #     continue
            if label_obs[i_idx] == 'South pole':
                continue
            if len(idx) == 1:
                Lum = np.concatenate(L_col[idx])
            else:
                Lum = np.mean(L_col[idx], axis = 0)
            if np.any(np.isnan(Lum)):
                print('skip ', i_idx)
                continue
            fit = pmodel.fit(Lum[idx_fit], n = freqs[idx_fit],  params=params)
            # fit = pmodel.fit(Lum, n = freqs,  params=params)
            Rfit[i_idx] = fit.params['R'].value
            Tfit[i_idx] = fit.params['T'].value
            # print(fit.fit_report())
            if not in_moll:
                print(f'At t = {np.round(time, 1)}' + r't$_{\rm fb}$' + f'Observer {label_obs[i_idx]}: Tfit = {Tfit[i_idx]:.2e} K, Rfit = {Rfit[i_idx]/prel.Rsol_cgs:.2e} rsol')
            line = ax[s].plot(x_value, freqs * Lum, label = f'{label_obs[i_idx]}' if s == 0 else None, c = colors_obs[i_idx])[0]
            if s == 0: 
                handles_color.append(line)
                labels_color.append(label_obs[i_idx])
        ax[s].set_title(f't = {np.round(time, 1)}' + r't$_{\rm fb}$', fontsize = 30, y = 1.15)

        for i_idx, idx in enumerate(indices_sorted): 
            if label_obs[i_idx] == 'South pole':
                continue
            BBfit = np.array([lumfit(freq, Rfit[i_idx], Tfit[i_idx]) for freq in freqs])
            lineB = ax[s].plot(x_value, freqs * BBfit, c = colors_obs[i_idx], ls = '-.', label = f'T={Tfit[i_idx]*1e-4:.1f}' + r' $\times 10^4$ K' if s < 2 else f'T={Tfit[i_idx]*1e-3:.1f}' + r' $\times 10^3$ K')[0]
            
            if s == 0: 
                handles_local.append(lineB)
                labels_local.append(f'T={Tfit[i_idx]*1e-4:.1f}' + r' $\times 10^4$ K' )

        Rfit_hist = np.zeros(len(L_col))
        Tfit_hist = np.zeros(len(L_col))
        for i in range(len(L_col)):
            fit_hist = pmodel.fit(L_col[i], n = freqs,  params=params)
            Rfit_hist[i] = fit_hist.params['R'].value
            Tfit_hist[i] = fit_hist.params['T'].value
        ax_fit[0][s].hist(Rfit_hist/(prel.Rsol_cgs), bins=20)
        ax_fit[1][s].hist(Tfit_hist*1e-4, bins=20)
        
    if x_axis == 'Temp':
        T_ticks = np.logspace(np.log10(T_min), np.log10(T_max), num=5)
        lambda_ticks = prel.c_cgs * 1e7 / (T_ticks/ prel.Hz_toK)
        lambda_labels = [f"{val:.2f}" for val in lambda_ticks]
        lambda_min = prel.c_cgs * 1e7 / (T_max / prel.Hz_toK)
        lambda_max = prel.c_cgs * 1e7 / (T_min / prel.Hz_toK)
    else: 
        nu_ticks = ax[0].get_xticks()
        nu_ticks = nu_ticks[nu_ticks > 0]
        lambda_ticks = prel.c_cgs * 1e7 / nu_ticks
        lambda_labels = [f"{val:.2f}" for val in lambda_ticks]
        lambda_min = prel.c_cgs * 1e7 / nu_max
        lambda_max = prel.c_cgs * 1e7 / nu_min

    for s in range(len(snaps)):
        ax[s].set_xticks(T_ticks)
        ax[s].set_ylim(L_min, L_max)
        ax[s].tick_params(axis='both', which='major', length=8, width=1.2)
        ax[s].tick_params(axis='both', which='minor', length=5, width=1)
        ax2 = ax[s].twiny()
        ax2.set_xticks(lambda_ticks) 
        ax2.set_xlim(lambda_max, lambda_min)
        ax2.set_xticklabels(lambda_labels)
        ax2.set_xlabel(r'$\lambda$ (nm)')
        ax2.set_xscale('log')
        ax[s].loglog()
        ax2.tick_params(axis='both', which='major', length=8, width=1.2)
        ax2.tick_params(axis='both', which='minor', length=5, width=1)
        ax_fit[0][s].set_xlabel(r'r$_{\rm BB} (r_\odot)$', fontsize = 20)
        ax_fit[1][s].set_xlabel(r'T ($10^4$ K)', fontsize = 20)
        ax_fit[1][s].set_xlim(0, 6)
        ax[s].legend(fontsize=16, loc = 'upper right')
    
    # legend1 = ax[0].legend(handles=handles_color,
    #                     labels=labels_color,
    #                     fontsize=17, loc='upper center',
    #                     bbox_to_anchor=(1.7, -.2),  # near bottom, centered
    #                     ncol=len(labels_color))

    legend_local = ax[0].legend(
                    handles=handles_local,      # your existing local handles
                    labels=labels_local,
                    loc='upper right',          # or wherever you want inside ax[0]
                    fontsize=14
                    )

    legend_colors = fig_sp.legend(
            handles=handles_color,
            labels=labels_color,
            loc='upper center',
            bbox_to_anchor=(0.525, 0.02),  # centered, near bottom of figure
            ncol=len(labels_color),
            fontsize=22) 


    # Legend 2: line-style explanation (solid vs dashed)
    # proxy_lines = []
    # proxy_lines = []
    # for l, line in enumerate(line_styles_parts):
    #     proxy_lines.append(
    #         mlines.Line2D([0], [0], color='k', ls=line, linewidth=2,
    #                     label=labels_parts[l])
    #     )

    ax_fit[0][0].set_ylabel(r'Counts', fontsize = 20)
    ax_fit[1][0].set_ylabel(r'Counts', fontsize = 20)
    ax[0].set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)
    fig_sp.tight_layout()
    fig_fit.tight_layout()
    fig_sp.savefig(f'{abspath}/Figs/2.paperWind/spectra_{choice}.pdf', dpi=300, bbox_inches='tight')
    fig_fit.savefig(f'{abspath}/Figs/{folder}/spectraFIT.png', dpi=300)
    
    if in_moll:
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
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
    idx_opt = np.where(np.logical_and(freqs > low_freq_optical, freqs < high_freq_optical))[0]
    idx_UV = np.where(np.logical_and(freqs > high_freq_optical, freqs < high_freq_UV))[0]
    idx_Xray = np.where(np.logical_and(freqs > high_freq_UV, freqs < high_freq_Xray))[0]

    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps_fld, tfb, Lum_fld = data[:, 0], data[:, 1], data[:, 2]
    snaps_fld, Lum_fld, tfb = sort_list([snaps_fld, Lum_fld, tfb], tfb, unique=True) 
    snaps_fld = snaps_fld.astype(int)
    tfb = np.array(tfb, dtype=float)
    idx_maxL = np.argmax(Lum_fld)

    # observers
    observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz)
    indices_sorted, label_obs, colors_obs, _, _ = choose_observers(observers_xyz, choice = choice)

    Lum_op_mean = []
    Lum_UV_mean = [] 
    Lum_Xray_mean = []
    time_col = []
    line_styles_parts = ['-', ':',]
    labels_parts = [r'This work', r'Giron+26']
    for s, snap in enumerate(snaps_fld):
        L_col = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        Lum_ph = photo['Lum'] 
        for i in range(len(L_col)):
            norm = Lum_ph[i] / np.trapezoid(L_col[i,:], freqs)
            L_col[i,:] *= norm
        Lum_op, Lum_UV, Lum_Xray = np.zeros(len(L_col)), np.zeros(len(L_col)), np.zeros(len(L_col))
        for i in range(len(L_col)):
            # Lum_freq = freqs * L_col[i] 
            # Lum_op[i] = np.sum(Lum_freq[idx_opt])
            Lum_op[i] = np.trapezoid(L_col[i,idx_opt], freqs[idx_opt]) 
            Lum_UV[i] = np.trapezoid(L_col[i,idx_UV], freqs[idx_UV])
            Lum_Xray[i] = np.trapezoid(L_col[i,idx_Xray], freqs[idx_Xray])
        
        Lum_op_i, Lum_UV_i, Lum_Xray_i = np.zeros(len(indices_sorted)), np.zeros(len(indices_sorted)), np.zeros(len(indices_sorted))
        for i_idx, idx in enumerate(indices_sorted):
            Lum_op_i[i_idx] = np.mean(Lum_op[idx])
            Lum_UV_i[i_idx] = np.mean(Lum_UV[idx])
            Lum_Xray_i[i_idx] = np.mean(Lum_Xray[idx])
        Lum_op_mean.append(Lum_op_i)
        Lum_UV_mean.append(Lum_UV_i)
        Lum_Xray_mean.append(Lum_Xray_i)
        time_col.append(tfb[s])
    Lum_op_mean = np.transpose(np.array(Lum_op_mean))
    Lum_UV_mean = np.transpose(np.array(Lum_UV_mean))
    Lum_Xray_mean = np.transpose(np.array(Lum_Xray_mean))

    Lum_op_MG = []
    Lum_UV_MG = []
    Lum_Xray_MG = []
    time_MG = []
    # observers MG
    nside_mg = 8
    observers_xyz_MG = hp.pix2vec(nside_mg, np.arange(hp.nside2npix(nside_mg))) #shape: (3, 768)
    observers_xyz_MG = np.array(observers_xyz_MG)
    indices_sorted_MG, _, _, _, _ = choose_observers(observers_xyz_MG, choice = choice)
    snaps_times_MG = np.loadtxt(f'{pre_saving}/MG/{check}_timesMG.csv', delimiter=',', dtype=float)
    snaps_MG = snaps_times_MG[:, 0].astype(int)
    time_MG = snaps_times_MG[:, 1]
    idx_MG_spectra = [np.argmin(np.abs(time_MG - 1.00)),
                        np.argmin(np.abs(time_MG - 1.54)), 
                        np.argmin(np.abs(time_MG - 2.23))]
    idx_fld_spectra = [np.argmin(np.abs(tfb - 1.00)),
                        np.argmin(np.abs(tfb - 1.54)), 
                        np.argmin(np.abs(tfb - 2.23))]
    idx_MG_spectra = np.array(idx_MG_spectra, dtype=int)
    idx_fld_spectra = np.array(idx_fld_spectra, dtype=int)

    for s, snap in enumerate(snaps_MG):
        L_colMG = np.loadtxt(f'{pre_saving}/MG/snap_{snap}/L_snap_{snap}.txt')
        t_MG = np.argmin(np.abs(snaps_MG - snap))
        Lum_op_MG_sum = np.sum(L_colMG[:, 1:3], axis = 1)
        Lum_UV_MG_sum = np.sum(L_colMG[:, 3:5], axis = 1)
        Lum_Xray_MG_sum = np.sum(L_colMG[:, 8:], axis = 1)
        # Lum_op_MG_mean = L_colMG[:, 1]
        # Lum_UV_MG_mean = L_colMG[:, 3]
        # Lum_Xray_MG_mean = L_colMG[:, 8]

        Lum_op_MG.append([np.mean(Lum_op_MG_sum[idx]) for idx in indices_sorted_MG])
        Lum_UV_MG.append([np.mean(Lum_UV_MG_sum[idx]) for idx in indices_sorted_MG])
        Lum_Xray_MG.append([np.mean(Lum_Xray_MG_sum[idx]) for idx in indices_sorted_MG])

    Lum_op_MG = np.transpose(np.array(Lum_op_MG))
    Lum_UV_MG = np.transpose(np.array(Lum_UV_MG))
    Lum_Xray_MG = np.transpose(np.array(Lum_Xray_MG))
    for i in np.arange(3):
        idx_t_MG = idx_MG_spectra[i]
        idx_t_fld = idx_fld_spectra[i]
        print(f'For t = {np.round(tfb[idx_t_fld], 2)} t_fb, MG time is {np.round(time_MG[idx_t_MG], 2)} t_fb')
        for k, obs in enumerate(label_obs):
            # if k == 3:
            #     continue
            if group == 'bandsMG':
                print(obs, '|| ratio Xray/opt: ', Lum_Xray_MG[k][idx_t_MG]/Lum_op_MG[k][idx_t_MG], ' ratio opt/UV: ', Lum_op_MG[k][idx_t_MG]/Lum_UV_MG[k][idx_t_MG])
            else:
                print(obs, '|| ratio Xray/opt: ', Lum_Xray_MG[k][idx_t_MG]/Lum_op_mean[k][idx_t_fld], ' ratio opt/UV: ', Lum_op_mean[k][idx_t_fld]/Lum_UV_mean[k][idx_t_fld])

    if group == 'sections': # each panel show a spherical sector
        len_plot = len(label_obs) if len(label_obs) > 1 else 2 # because then u substract 1
        if np.array(label_obs).all() not in ['South pole', r'-$\hat{z}$']:
            print('Do not consider the south pole')
            len_plot -= 1
        fig_L, ax_L = plt.subplots(1, len_plot, figsize=(8*len_plot, 7)) 
        axes = [ax_L[k] for k in range(len_plot)] if len_plot > 1 else [ax_L]
        for k in np.arange(len(axes)):
            Lum_op = Lum_op_mean[k]
            Lum_UV = Lum_UV_mean[k]
            Lum_XrayMG = Lum_Xray_MG[k]
            axes[k].plot(tfb, Lum_op, label = 'Optical', c = colors_obs[k])
            axes[k].scatter(tfb[np.argmax(Lum_op)], np.max(Lum_op), c = colors_obs[k], s = 200, marker = '*', edgecolors='k', zorder = 5)
            axes[k].plot(tfb, Lum_UV, label = f'UV', c = colors_obs[k], ls = '--')
            axes[k].scatter(tfb[np.argmax(Lum_UV)], np.max(Lum_UV), c = colors_obs[k], s = 200, marker = '*', edgecolors='k', zorder = 5)
            axes[k].plot(time_MG, Lum_XrayMG, label = f'Xray', c = colors_obs[k], ls = ':')
            axes[k].scatter(time_MG[np.argmax(Lum_XrayMG)], np.max(Lum_XrayMG), c = colors_obs[k], s = 200, marker = '*', edgecolors='k', zorder = 5)
            axes[k].text(0.1, L_max/5, f'{label_obs[k]}', fontsize = 26)
        original_ticks = axes[0].get_xticks()
        axes[0].legend(fontsize = 25)
        axes[0].set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)

    if group == 'bands' or group == 'bandsMG': # each panel show a band
        if group == 'bands' :
            fig_L, (ax_op, ax_UV) = plt.subplots(1, 2, figsize=(16, 7))
            fig_x, ax_Xray = plt.subplots(1, 1, figsize=(9, 7))
        else:
            fig_L, (ax_op, ax_UV, ax_Xray) = plt.subplots(1, 3, figsize=(24, 7))
        axes = [ax_op, ax_UV, ax_Xray] 
        handles_color, labels_color = [], []
        for k, obs in enumerate(label_obs):
            if label_obs[k] == 'South pole':
                continue
            Lum_op = Lum_op_mean[k]
            Lum_UV = Lum_UV_mean[k] 
            Lum_XrayMG = Lum_Xray_MG[k]
            line = ax_op.plot(tfb, Lum_op, label = f'{obs}', c = colors_obs[k])[0]
            handles_color.append(line)
            labels_color.append(f'{obs}')
            ax_UV.plot(tfb, Lum_UV, c = colors_obs[k])
            ax_Xray.plot(time_MG, Lum_XrayMG, c = colors_obs[k], label = f'{obs}', ls = line_styles_parts[1] if group == 'bandsMG' else line_styles_parts[0])
            if group == 'bands':
                ax_op.scatter(tfb[np.argmax(Lum_op)], np.max(Lum_op), c = colors_obs[k], s = 200, marker = '*', edgecolors='k', zorder = 5)
                ax_UV.scatter(tfb[np.argmax(Lum_UV)], np.max(Lum_UV), c = colors_obs[k], s = 200, marker = '*', edgecolors='k', zorder = 5)
                ax_Xray.scatter(time_MG[np.argmax(Lum_XrayMG)], np.max(Lum_XrayMG), c = colors_obs[k], s = 200, marker = '*', edgecolors='k', zorder = 5)

            if group == 'bandsMG':
                ax_Xray.plot(tfb, Lum_Xray_mean[k], c = colors_obs[k], ls = line_styles_parts[0])
                ax_op.plot(time_MG, Lum_op_MG[k], c = colors_obs[k], ls = line_styles_parts[1])
                ax_UV.plot(time_MG, Lum_UV_MG[k], c = colors_obs[k], ls = line_styles_parts[1])

        ax_op.text(0.05, L_max/3, 'Optical', fontsize = 26)
        ax_UV.text(0.05, L_max/3, 'UV', fontsize = 26)
        ax_Xray.text(0.05, L_max/3, 'X-ray', fontsize = 26)
        original_ticks = ax_op.get_xticks()
        ax_op.legend(fontsize = 16)
        
        if group == 'bandsMG':
            fig_L.legend(
                handles=handles_color,
                labels=labels_color,
                # loc='upper left',
                bbox_to_anchor=(0.95, 0.02),  # centered, near bottom of figure
                ncol=len(labels_color),
                fontsize=22) 
            
            proxy_lines = []
            proxy_lines = []
            for l, line in enumerate(line_styles_parts):
                proxy_lines.append(mlines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                                label=labels_parts[l]))

            ax_op.legend(handles=proxy_lines, fontsize=22, 
                                loc='lower right')

        ax_op.set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)
        ax_Xray.set_ylabel(r'$\nu L_{\nu}$ (erg/s)', fontsize = 30)

    if group == 'bands':
        ax_op.axhline(L_ZTF, c = 'gray', ls = '-.', linewidth = 1)
        ax_op.text(0, 0.6*L_ZTF, 'g-ZTF', fontsize = 16, color = 'gray')
        ax_op.axhline(L_Rubin, c = 'gray', ls = '-.', linewidth = 1)
        ax_op.text(0, 0.6*L_Rubin, ' g-Rubin', fontsize = 16, color = 'gray')
        ax_UV.axhline(L_ULTRA, c = 'gray', ls = '-.', linewidth = 1)
        ax_UV.text(0, 0.6*L_ULTRA, 'ULTRASAT', fontsize = 16, color = 'gray')
        ax_Xray.axhline(L_eRos, c = 'gray', ls = '-.', linewidth = 1)
        ax_Xray.text(1.75, 0.6*L_eRos, 'eROSITA', fontsize = 16, color = 'gray')
        ax_Xray.axhline(L_Einstein, c = 'gray', ls = '-.', linewidth = 1)
        # ax_Xray.text(1.75, 0.6*L_Einstein, 'EP/WXT', fontsize = 16, color = 'gray')
    
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]   
    days_ticks = new_ticks*t_fb_days
    days_labels = [str(np.round(days_ticks[k],2)) if new_ticks[k] in original_ticks else "" for k in range(len(days_ticks))]
    for ax in axes:
        if group == 'bands':
            if ax != ax_Xray:
                ax.plot(tfb, Lum_fld, c = 'k', ls = '-.')
                ax.scatter(tfb[idx_maxL], Lum_fld[idx_maxL], c = 'k', s = 200, marker = '*')
        ax.set_yscale('log')
        ax.set_xticks(new_ticks)
        ax.set_xlabel(r't / t$_{\rm fb}$', fontsize = 30)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='major', width = 1, length = 7, color = 'k')
        ax.tick_params(axis='y', which='minor', width = 1, length = 4, color = 'k')
        ax.grid()
        ax.set_ylim(L_min, L_max)
        ax.set_xlim(tmin, tmax) 

        ax2 = ax.twiny()
        # ax2.axvline(4.5)
        ax2.set_xticks(days_ticks)
        ax2.set_xlim(tmin*t_fb_days, tmax*t_fb_days)
        ax2.set_xticklabels(days_labels)
        ax2.set_xlabel(r't (days)', fontsize = 30)
    
    fig_L.tight_layout()
    fig_L.savefig(f'{abspath}/Figs/2.paperWind/LCs_{choice}_{group}.pdf', dpi=300, bbox_inches='tight')
    if group == 'bands':
        fig_x.tight_layout()
        fig_x.savefig(f'{abspath}/Figs/2.paperWind/LCsXray_{choice}_{group}.pdf', dpi=300)

def TRfit_in_time(folder, check, choice):
    # Load
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt') 
    #band in angstrom = 1e7 cm and are wavelenght so the minimum gives the maximum freq
    idx_ztf_g = np.where(np.logical_and(freqs > prel.c_cgs/(prel.ztf_g_band[1]*1e-7), freqs <  prel.c_cgs/(prel.ztf_g_band[0]*1e-7)))[0]
    idx_ztf_r = np.where(np.logical_and(freqs > prel.c_cgs/(prel.ztf_r_band[1]*1e-7), freqs <  prel.c_cgs/(prel.ztf_r_band[0]*1e-7)))[0]
    idx_ztf_i = np.where(np.logical_and(freqs > prel.c_cgs/(prel.ztf_i_band[1]*1e-7), freqs <  prel.c_cgs/(prel.ztf_i_band[0]*1e-7)))[0]
    idx_swift_u_band = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_u_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_u_band[0]*1e-7)))[0]
    idx_swift_b_band = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_b_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_b_band[0]*1e-7)))[0]
    idx_swift_v_band = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_v_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_v_band[0]*1e-7)))[0]
    idx_opt = np.concatenate([idx_ztf_g, idx_ztf_r, idx_ztf_i, idx_swift_u_band, idx_swift_b_band, idx_swift_v_band])

    idx_swift_uvw1 = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_uvw1_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_uvw1_band[0]*1e-7)))[0]
    idx_swift_uvm2 = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_uvm2_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_uvm2_band[0]*1e-7)))[0]
    idx_swift_uvw2 = np.where(np.logical_and(freqs > prel.c_cgs/(prel.swift_uvw2_band[1]*1e-7), freqs <  prel.c_cgs/(prel.swift_uvw2_band[0]*1e-7)))[0]
    idx_UV = np.concatenate([idx_swift_uvw1, idx_swift_uvm2, idx_swift_uvw2])
    idx_fit = np.concatenate([idx_UV, idx_opt])
    # idx_fit = np.where(np.logical_and(freqs > 0, freqs <  1e40))[0]

    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps_fld, tfb, Lum_fld = data[:, 0], data[:, 1], data[:, 2]
    snaps_fld, Lum_fld, tfb = sort_list([snaps_fld, Lum_fld, tfb], tfb, unique=True) 
    snaps_fld = snaps_fld.astype(int)
    tfb = np.array(tfb, dtype=float)

   # observers
    observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz)
    longitude_moll = np.arctan2(observers_xyz[1], observers_xyz[0]) # from -pi to pi, 0 at x axis, positive towards y axis
    theta_obs = np.arccos(observers_xyz[2]) # from 0 (+z axis) to pi (-z axis)
    latitude_moll = np.pi/2 - theta_obs  # from np.pi/2 (z axis) to -np.pi/2 (-z axis)
    cross_dot = np.matmul(observers_xyz.T,  observers_xyz)
    cross_dot[cross_dot<0] = 0
    cross_dot /= 192
    indices_sorted, label_obs, colors_obs, _, _ = choose_observers(observers_xyz, choice = choice)

    fig_sp, (axT, axR, axL) = plt.subplots(1, 3, figsize=(28,8))
    Lbolom_sec, Rfit_sec, Tfit_sec, Lumfit_sec = [], [], [], []
    for s, snap in enumerate(snaps_fld):
        L_col = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')

        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        Lum_ph = photo['Lum']
        for i in range(len(L_col)):
            norm = Lum_ph[i] / np.trapezoid(L_col[i,:], freqs)
            L_col[i,:] *= norm
        
        L_col = np.matmul(cross_dot, L_col)
        Lbolom, Rfit, Tfit, Lumfit = np.zeros(len(indices_sorted)), np.zeros(len(indices_sorted)), np.zeros(len(indices_sorted)), np.zeros(len(indices_sorted))
        for i_idx, idx in enumerate(indices_sorted):
            # if i_idx == 3:
            #     continue
            if len(idx) == 1:
                Lum = np.concatenate(L_col[idx])
            else:
                Lum = np.mean(L_col[idx], axis = 0)
            fit = pmodel.fit(Lum[idx_fit], n = freqs[idx_fit],  params=params)
            # fit = pmodel.fit(Lum, n = freqs,  params=params)
            Rfit[i_idx] = fit.params['R'].value
            Tfit[i_idx] = fit.params['T'].value
            BBfit = np.array([lumfit(freq, Rfit[i_idx], Tfit[i_idx]) for freq in freqs])
            Lumfit[i_idx] =  sci.trapezoid(BBfit, freqs)
            Lbolom[i_idx] = np.mean(Lum_ph[idx])
        Lbolom_sec.append(Lbolom)
        Rfit_sec.append(Rfit)
        Tfit_sec.append(Tfit)
        Lumfit_sec.append(Lumfit)
    Lbolom_sec = np.array(Lbolom_sec)
    Rfit_sec = np.array(Rfit_sec)
    Tfit_sec = np.array(Tfit_sec)
    Lumfit_sec = np.array(Lumfit_sec)
    header_cols = ['t_fb'] \
            + [f'Tfit_sec {obs}' for obs in label_obs] \
            + [f'Rfit_sec {obs}' for obs in label_obs]

    header_str = ','.join(header_cols)  # or delimiter if not comma

    np.savetxt(
        f'{abspath}/data/{folder}/wind/Tfit_intime_{choice}.txt',
        np.column_stack((tfb, Tfit_sec, Rfit_sec)),
        delimiter=',',
        header=header_str)
    
    for i_idx, idx in enumerate(indices_sorted):
        if label_obs[i_idx] == 'South pole':
            continue
        Rfit = Rfit_sec[:, i_idx]
        Tfit = Tfit_sec[:, i_idx]
        Lumfit = Lumfit_sec[:, i_idx]
        Lbolom = Lbolom_sec[:, i_idx]
        axT.plot(tfb, Tfit, c = colors_obs[i_idx], label = f'{label_obs[i_idx]}')
        axR.plot(tfb, Rfit, c = colors_obs[i_idx], label = f'{label_obs[i_idx]}')
        axL.plot(tfb, Lumfit/Lbolom, c = colors_obs[i_idx], label = f'{label_obs[i_idx]}')
    axR.set_ylabel(r'r$_{\rm BB}$ (cm)', fontsize = 30)
    axT.set_ylabel(r'T$_{\rm BB}$ (K)', fontsize = 30)
    axL.set_ylabel(r'L$_{\rm BB}/$L$_{\rm bol}$', fontsize = 30)
    axT.set_ylim(4e3, 8e4)
    axT.legend(fontsize=18)
    axR.set_ylim(1e11, 1e14)
    axL.set_ylim(1e-2, 1)
    for ax in [axT, axR, axL]:
        ax.set_xlabel(r't / t$_{\rm fb}$', fontsize = 30)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', width = 1.2, length = 9, color = 'k')
        ax.tick_params(axis='y', which='minor', width = 1, length = 5, color = 'k')
        ax.grid()

    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/{folder}/Wind/Tfit_intime_{choice}.png', dpi=300)
    
pmodel = Model(lumfit)
params = pmodel.make_params(R=1e13, T=1e4)
params['R'].min = 0.0    # R ≥ 0
params['T'].min = 0.0    # T ≥ 0  
plot_spectra(folder, check, snaps_spectra, x_axis, choice)
# TRfit_in_time(folder, check, choice)
# plot_light_curves(folder, check, choice, group = 'bands')
# plot_light_curves(folder, check, choice, group = 'sections')
# plot_light_curves(folder, check, choice, group = 'bandsMG')
