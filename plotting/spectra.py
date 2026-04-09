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

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 55
x_axis = 'Temp'  # 'Freq' or 'Temp'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

def plot_spectra(folder, check, snap, x_axis, choice = 'single_axis'):
    # Load
    pre_saving = f'{abspath}/data/{folder}'
    freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
    L_photo = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
    snaps, Lum, tfb = sort_list([snaps, Lum, tfb], tfb, unique=True) 
    snaps = snaps.astype(int)
    time = tfb[snaps == snap][0]
    # Plot
    fig = plt.figure(figsize=(24,10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1, .05], hspace=0.4, wspace = 0.2)
    ax1 = fig.add_subplot(gs[0, 0], projection='mollweide')
    ax1.grid(True)
    ax1.set_xticks(np.radians(np.arange(-180, 181, 90))) 
    ax1.set_xticklabels(['-180°', '-90°', '0°','90°', '180°'])
    ax1.set_yticks(np.radians(np.arange(-90, 91, 45))) 
    ax1.set_yticklabels(['-90°', '-45°', '0°', '45°','90°'])
    ax = fig.add_subplot(gs[0, 1]) 

    plt.suptitle(f't = {time:.2f}'+ r' t$_{\rm fb}$', fontsize=24)
    if x_axis == 'Temp':
        x_value = freqs * prel.Hz_toK
        ax.set_xlabel('Temperature [K]')
        ax.set_xlim(1e3, 5e7)
    else:
        x_value = freqs
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(1e14, 1e19)
    N_obs = L_photo.shape[0] 
    observers_xyz = hp.pix2vec(prel.NSIDE, range(N_obs)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz)
    longitude_moll_h = np.arctan2(observers_xyz[1], observers_xyz[0])
    theta_obs = np.arccos(observers_xyz[2])
    latitude_moll_h = np.pi/2 - theta_obs

    cross_dot = np.matmul(observers_xyz.T,  observers_xyz)
    cross_dot[cross_dot<0] = 0
    cross_dot /= 192
    Xray_obs2 = []
    Xray_obs5 = []
    idx_2e6 = np.where(freqs * prel.Hz_toK > 2e6)[0][0]
    idx_5e6 = np.where(freqs * prel.Hz_toK > 5e6)[0][0]
    for i in range(len(L_photo)):
        check = L_photo[i, idx_2e6:] * freqs[idx_2e6:] 
        if np.any(check > 1e39):
            Xray_obs2.append(i)
        check = L_photo[i, idx_5e6:] * freqs[idx_5e6:] 
        if np.any(check > 1e39):
            Xray_obs5.append(i)
    Xray_obs2 = np.array(Xray_obs2)
    Xray_obs5 = np.array(Xray_obs5)
    
    ax1.scatter(longitude_moll_h, latitude_moll_h, s = 100, facecolors='None', edgecolors='k') #color by intensity
    if len(Xray_obs2)>0:
        ax1.scatter(longitude_moll_h[Xray_obs2], latitude_moll_h[Xray_obs2], c = 'red', s = 100, edgecolors='k', label = r'$\nu F_{\nu} > 10^{39}$ erg s$^{-1}$ at $T > 2 \times 10^6$ K') 
        if len(Xray_obs5)>0:
            ax1.scatter(longitude_moll_h[Xray_obs5], latitude_moll_h[Xray_obs5], c = 'dodgerblue', s = 100, edgecolors='k', label = r'$\nu F_{\nu} > 10^{39}$ erg s$^{-1}$ at $T > 5 \times 10^6$ K') 
    # cbar_ax = fig.add_subplot(gs[1, 0]) 
    # cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label =r'I')
    # cbar.ax.tick_params(which='major',length = 5)
    # cbar.ax.tick_params(which='minor',length = 3)
    # move the legend outside the plot
    ax1.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.4), fontsize=20)
    ax1.set_title('Observers with X-ray emission', fontsize=20, pad=50)
    
    L_photo = np.matmul(cross_dot, L_photo)
    indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = choice)
    for i_idx, idx in enumerate(indices_sorted):
        Lum = np.concatenate(L_photo[idx])
        ax.plot(x_value, freqs * Lum, label = f'Obs {idx} ({label_obs[i_idx]})', c = colors_obs[i_idx], ls = lines_obs[i_idx])
                        
    ax.tick_params(axis='both', which='major', length=8, width=1.2)
    ax.tick_params(axis='both', which='minor', length=5, width=1)
    ax.loglog()
    ax.set_ylim(1e38, 4e41)
    ax.set_ylabel(r'$\nu F_{\nu}$ [erg s$^{-1}$]')
    ax.legend(fontsize=16)
    ax.set_title('(Weighted) spectra', fontsize=20)
    
    plt.tight_layout()
    plt.show()
    

plot_spectra(folder, check, snap, x_axis)

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


