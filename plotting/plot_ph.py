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
from scipy.stats import gmean
from Utilities.basic_units import radians
from Utilities.operators import find_ratio, sort_list, choose_observers
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
compton = 'Compton'
check ='NewAMR'
which_obs = 'dark_bright_z'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
Ledd = 1.26e38 * Mbh

#%%
# tvisc = (R0**3/(prel.G*Mbh))**(1/2) * (1/0.1) * (1/0.3)**2
# tvisc_days = tvisc*prel.tsol_cgs/(3660*24)
# print(tvisc_days)

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_axis, label_axis, colors_axis, lines_axis = choose_observers(observers_xyz, which_obs)
observers_xyz = observers_xyz.T
# HEALPIX
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
# r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
# theta = np.arctan2(y, x)          # Azimuthal angle in radians
# phi = np.arccos(z / r)            # Elevation angle in radians
# # Convert to latitude and longitude
# longitude_moll = theta              
# latitude_moll = np.pi / 2 - phi 

# indecesorbital = np.concatenate(np.where(latitude_moll==0))
# long_orb, lat_orb = longitude_moll[indecesorbital], latitude_moll[indecesorbital]

# print('Longitude from HELAPIX min and max: ' , np.min(theta), np.max(theta))
# print('Longitude for mollweide min and max: ', np.min(longitude_moll), np.max(longitude_moll))
# print('Latitude from HELAPIX min and max: ' , np.min(phi), np.max(phi))
# print('Latitude for mollweide min and max: ', np.min(latitude_moll), np.max(latitude_moll))
# print('So we shift of pi/2 the latitude')
# # Plot in 2D using a Mollweide projection
# fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
# img = ax.scatter(longitude_moll, latitude_moll, s=20, c=np.arange(len(longitude_moll)), cmap='viridis')
# # ax.scatter(longitude_moll[first_eq:final_eq], latitude_moll[first_eq:final_eq], s=10, c='r')
# ax.scatter(long_orb, lat_orb, s=10, c='r')
# plt.colorbar(img, ax=ax, label='Observer Number')
# ax.set_title("Observers on the Sphere (Mollweide Projection)")
# ax.grid(True)
# ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
# ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
# plt.tight_layout()
# plt.show()

#%% NB DATA ARE NOT SORTED
ph_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}_phidx_fluxes.txt')
snaps, tfb, allindices_ph = ph_data[:, 0].astype(int), ph_data[:, 1], ph_data[:, 2:]
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
mean_rph_nonZero = np.zeros(len(tfb))
mean_rph_weig = np.zeros(len(tfb))
gmean_ph = np.zeros(len(tfb))
median_ph = np.zeros(len(tfb))
median_rph_nonZero = np.zeros(len(tfb))
percentile16 = np.zeros(len(tfb))
percentile84 = np.zeros(len(tfb))

for i, snapi in enumerate(snaps):
        photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snapi}.txt')
        xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
        dim_i = vol_i**(1/3)
        rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
        mean_rph[i] = np.mean(rph_i)
        # mean_rph_nonZero[i] = np.mean(rph_i[rph_i!=0])
        mean_rph_weig[i] = np.sum(rph_i*fluxes[i])/np.sum(fluxes[i])
        gmean_ph[i] = gmean(rph_i)
        median_ph[i] = np.median(rph_i)
        # median_rph_nonZero[i] = np.median(rph_i[rph_i!=0])
        percentile16[i] = np.percentile(rph_i, 16)
        percentile84[i] = np.percentile(rph_i, 84)

plt.figure(figsize = (10, 7))
plt.plot(tfb, mean_rph/apo, c = 'k', label = 'mean')
# plt.plot(tfb, mean_rph_nonZero/apo, c = 'gray', ls = '--', label = 'mean non zero elements')
plt.plot(tfb, mean_rph_weig/apo, c = 'firebrick', label = 'weighted by flux')
plt.plot(tfb, gmean_ph/apo, c = 'deepskyblue', label = 'geometric mean')
plt.plot(tfb, median_ph/apo, c = 'forestgreen', label = 'median')
# plt.plot(tfb, median_rph_nonZero/apo, c = 'yellowgreen', ls = '--', label = 'median non zero elements')
plt.xlabel(r't [$t_{fb}$]')
plt.ylabel(r'$\langle R_{ph}\rangle [R_a]$')
plt.yscale('log')
plt.ylim(1e-2, 10)
plt.xlim(0.4, 1.6)
plt.grid()
plt.legend(fontsize = 16)
#%% 
rph_obs_time = np.zeros((len(tfb), len(indices_axis)))
den_obs_time = np.zeros((len(tfb), len(indices_axis)))
T_obs_time = np.zeros((len(tfb), len(indices_axis)))
Lum_obs_time = np.zeros((len(tfb), len(indices_axis)))
rtr_obs_time = np.zeros((len(tfb), len(indices_axis)))
# alpha_obs_time = np.zeros((len(tfb), len(indices_axis)))
flux_obs_time = np.zeros((len(tfb), len(indices_axis)))

normalize_by = 'Rp'
if normalize_by == 'Rp':
        norm = Rp
else:
        norm = apo
for i, snapi in enumerate(snaps):
        xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph, Pressph, IE_denph, alphaph, _, Lumph, _ = \
                np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snapi}.txt')
        dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snapi}.npz")
        x_tr, y_tr, z_tr = \
            dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr']
        rtr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)

        fluxph = fluxes[i]
        dim = volph**(1/3)
        rph = np.sqrt(xph**2 + yph**2 + zph**2)
        for j, observer in enumerate(indices_axis):
                # nonzero_mask = rph[observer] != 0
                rph_obs_time[i, j] = np.median(rph[observer])
                den_obs_time[i, j] = np.median(denph[observer])
                T_obs_time[i, j] = np.median(Tempph[observer])
                Lum_obs_time[i, j] = np.median(Lumph[observer])
                # alpha_obs_time[i, j] = np.median(alphaph[observer])
                flux_obs_time[i, j] = np.median(fluxph[observer])
                # rtr_obs_time[i, j] = np.median(rtr[observer])
                 
                # nonzero_mask = rtr[observer] != 0
                rtr_obs_time[i, j] = np.median(rtr[observer])

fig, (ax1, ax5) = plt.subplots(1, 2, figsize=(25, 7))
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 7))
figf, ax6 = plt.subplots(1, 1, figsize=(10, 7))
figTr, axTr = plt.subplots(1, 1, figsize=(10, 7))
for i, observer in enumerate(indices_axis):
        # if label_axis[i] in ['z-', 'z+']:
        #         Lum_to_plot = Lum_obs_time[135:, i]
        #         tfb_to_plot = tfb[135:]
        # else:
        Lum_to_plot = Lum_obs_time[:, i]
        tfb_to_plot = tfb
        # if label_axis[i] not in ['x+', 'z-', 'z+']:
        #     continue
        if normalize_by == 'apo': 
                print(f'final Rtr/apo {label_axis[i]}: ',rtr_obs_time[-1, i]/norm)
        ax1.plot(tfb, rph_obs_time[:, i]/norm, label = label_axis[i], c = colors_axis[i], ls = lines_axis[i])
        ax2.plot(tfb, den_obs_time[:, i]*prel.den_converter, label = label_axis[i], c = colors_axis[i], ls = lines_axis[i])
        ax3.plot(tfb, T_obs_time[:, i], label = label_axis[i], c = colors_axis[i], ls = lines_axis[i])
        # ax4.plot(tfb, alpha_obs_time[:, i], label = label_axis[i], c = colors_axis[i])
        ax5.plot(tfb_to_plot, Lum_to_plot/Ledd, label = label_axis[i], c = colors_axis[i], ls = lines_axis[i])
        ax6.plot(tfb, flux_obs_time[:, i], label = label_axis[i], c = colors_axis[i], ls = lines_axis[i])
        if label_axis[i] not in ['y']:
                axTr.plot(tfb, rtr_obs_time[:, i]/norm, label = label_axis[i], c = colors_axis[i], ls = lines_axis[i])
# ax1.axhline(R0/apo, color = 'gray', linestyle = ':', label = r'R$_0$')
# ax1.plot(tfb, mean_rph/apo, c = 'gray', ls = '--', label = 'mean')
# ax1.plot(tfb, median_ph/apo, c = 'gray', ls = ':', label = 'median')
if normalize_by == 'Rp':
        ax1.set_ylabel(r'median R$_{\rm ph}[R_{\rm p}$]')
        ax1.set_ylim(1, 1e2)
if normalize_by == 'apo':
        ax1.axhline(Rp/norm, color = 'gray', linestyle = '-.', label = r'R$_{\rm p}$')
        ax1.set_ylabel(r'median R$_{\rm ph}[R_{\rm a}$]')
        ax1.set_ylim(1e-2, 10)
ax1.legend(fontsize = 16)
ax2.set_ylabel(r'median $\rho_{\rm ph}$[g/cm$^3$]')
ax2.set_ylim(1e-14, 5e-9)
ax3.set_ylabel(r'median T$_{\rm ph}$[K]')
ax3.set_ylim(5e3, 1e10)
ax3.legend(fontsize = 16)
ax5.set_ylabel(r'median L$_{\rm ph}[L_{\rm Edd}]$')
ax5.axhline(y = 1.26e42, color = 'gray', linestyle = '--')
ax5.set_ylim(1e-2, 5)
ax6.set_ylabel(r'median Flux$_{\rm ph}$ [erg/s cm$^2$]')
ax6.legend(fontsize = 16)
if normalize_by == 'apo':
        axTr.axhline(Rp/apo, color = 'gray', linestyle = '-.', label = r'R$_{\rm p}$')
        axTr.set_ylabel(r'$R_{\rm tr} [R_{\rm a}]$')
        # axTr.set_ylim(2e-2, 10)
if normalize_by == 'Rp':
        axTr.set_ylabel(r'$R_{\rm tr} [R_{\rm p}]$')
        axTr.set_ylim(1e-1, 1e2)
axTr.legend(fontsize = 16)
# original_ticks = ax1.get_xticks()
# middle_tick = (original_ticks[:-1] + original_ticks[1:]) / 2
# new_ticks = np.concatenate((original_ticks, middle_tick))
# ticks_label = [f'{np.round(tick,2)}' if tick in original_ticks else "" for tick in new_ticks]
for ax in [ax1, ax2, ax3, ax5, ax6, axTr]:
        # ax.set_xticks(new_ticks)
        # ax.set_xticklabels(ticks_label)
        ax.set_xlim(0.4, 1.6)
        ax.set_xlabel(r't [$t_{\rm fb}$]')
        ax.tick_params(axis='both', which='major', width=1.2, length=9, color = 'k')
        ax.tick_params(axis='both', which='minor', width=1, length=7, color = 'k')
        ax.grid()
        ax.set_yscale('log')
fig.suptitle(f'{check}', fontsize = 30) 
fig2.suptitle(f'{check}', fontsize = 30)
# fig.tight_layout()
fig.subplots_adjust(wspace = 0.2)
fig2.tight_layout()
fig.savefig(f'{abspath}/Figs/next_meeting/{check}/photo_RL_{which_obs}_median.png')
fig2.savefig(f'{abspath}/Figs/next_meeting/{check}/photo_Trho_{which_obs}.png')
figTr.savefig(f'{abspath}/Figs/next_meeting/{check}/trap_profile_{normalize_by}_{which_obs}_median.png')

#%% 3D plot at the chosen snap
if which_obs != 'dark_bright_z':
        idx_snap = -1
        xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph, Pressph, IE_denph, alphaph, _, Lumph, _ = \
                np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snaps[idx_snap]}.txt')
        x_sel = xph[indices_axis[:,0]]
        y_sel = yph[indices_axis[:,0]]
        z_sel = zph[indices_axis[:,0]]
        # z=0 plane
        x_z0 = np.linspace(-5, 2, 60)
        y_z0 = np.linspace(-2, 2, 60)
        X_z0, Y_z0 = np.meshgrid(x_z0, y_z0)
        Z_z0 = np.zeros_like(X_z0)

        # 3d scatter plot
        fig3d = plt.figure(figsize=(15, 15))
        ax3d = fig3d.add_subplot(projection='3d')
        ax3d.scatter(x_sel/apo, y_sel/apo, z_sel/apo, c = colors_axis, s = 100)

        # ax3d.set_xlabel(r'$x/R_{\rm a}$')
        # ax3d.set_ylabel(r'$y/R_{\rm a}$')
        # ax3d.set_zlabel(r'$z/R_{\rm a}$')
        # plane at z=0
        ax3d.plot_surface(X_z0, Y_z0, Z_z0, color= 'gray', linewidth=0, antialiased=True, alpha=0.1)
        ax3d.set_xlim(-5, 2)
        ax3d.set_ylim(-2.1, 2.1)
        ax3d.set_zlim(-2.1, 2.1)

        # Hide the 3D box
        # ax3d.set_axis_off()

        # Set axis limits
        lim = 2
        ax3d.set_xlim(-6, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_box_aspect([1, 1, 1])  # equal aspect ratio

        # Draw custom Cartesian axes
        # X-axis (line + arrow only on +X)
        ax3d.plot([-6, lim], [0, 0], [0, 0], color="k")
        ax3d.quiver(0,0,0, lim,0,0, color="k", arrow_length_ratio=0.05)

        # Y-axis (line + arrow only on +Y)
        ax3d.plot([0, 0], [-lim, lim], [0, 0], color="k")
        ax3d.quiver(0,0,0, 0,lim,0, color="k", arrow_length_ratio=0.05)

        # Z-axis (line + arrow only on +Z)
        ax3d.plot([0, 0], [0, 0], [-lim, lim], color="k")
        ax3d.quiver(0,0,0, 0,0,lim, color="k", arrow_length_ratio=0.05)

        # Add axis labels at positive tips
        ax3d.text(lim, 0, 0, r"$x/R_{\rm a}$", size=20)
        ax3d.text(0, lim, 0, r"$y/R_{\rm a}$", size=20)
        ax3d.text(0, 0, lim, r"$z/R_{\rm a}$", size=20)

        plt.tight_layout()

#%%
idx_snap = -1 #np.argmin(np.abs(snaps - 266))
plt.figure(figsize = (10, 7))
# o order them as you want
if which_obs == 'hemispheres':
    to_plot = [0, 6, 4, 8]
elif which_obs == 'dark_bright_z':
    to_plot = [0, 2, 3, 1]
for i, idx_toplot in enumerate(to_plot):
        plt.scatter(i+1, Lum_obs_time[idx_snap, idx_toplot]/Ledd, label = label_axis[idx_toplot], c = colors_axis[idx_toplot], ls = lines_axis[idx_toplot], s = 100)
plt.legend(fontsize = 16)
plt.ylabel(r'$L_{\rm ph}/L_{\rm Edd}$')
plt.xticks([])
plt.yscale('log') 
plt.ylim(1e-2, 3)
plt.grid()
plt.tick_params(axis='y', which='major', width=1.2, length=9, color = 'k')
plt.tick_params(axis='y', which='minor', width=1, length=7, color = 'k')
plt.title(f't = {np.round(tfb[idx_snap], 2)}' + r' t$_{\rm fb}$', fontsize = 16)
plt.show(block=True)

# #%% compare with other resolutions
# pvalue 
# statL = np.zeros(len(tfbL)) 
# pvalueL = np.zeros(len(tfbL))
# for i, snapi in enumerate(snapsL):
#         # LowRes data
#         photo = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/photo/LowResNewAMR_photo{snapi}.txt')
#         xph_i, yph_i, zph_i, vol_i = photo[0], photo[1], photo[2], photo[3]
#         rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
#         if rph_i.any() < R0:
#                 print('Less than R0:', rph_i[rph_i<R0])
#         # ksL = ks_2samp(rph_i, rph_iFid, alternative='two-sided')
#         # statL[i], pvalueL[i] = ksL.statistic, ksL.pvalue

# %% check opacity
xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph, Pressph, IE_denph, alphaph, _, Lumph, _ = \
        np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo318.txt')
kappaph = alphaph/denph
kappa_invmean = 1/np.mean(1/kappaph)
kappa_bins = np.arange(0, 5, .5)
plt.hist(kappaph, bins = kappa_bins, color = 'forestgreen', alpha = .5)
plt.xlabel(r'$\kappa_{\rm ph}$ [cm$^2$/g]')
plt.ylabel('N')
plt.title(r'1/$<1/\kappa_{\rm ph}$:' + f'{np.round(kappa_invmean, 2)}')
plt.show(block=True)

# %%
