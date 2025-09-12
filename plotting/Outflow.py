""" Investigate the outflow. Look at the velocity and density of the photosphere"""
abspath = '/Users/paolamartire/shocks'
import sys

sys.path.append(f'{abspath}')
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
# import colorcet
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import to_spherical_components, sort_list
from Utilities.sections import make_slices

##
# PARAMETERS
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 
compton = 'Compton'
check = 'NewAMR'
kind_of_plot = 'time_evolution' # 'ratioE' or 'convergence' or 'velocity_components' or 'time_evolution'

conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
t_fall_hour = tfallback * 24 
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

# HEALPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 
indecesorbital = np.concatenate(np.where(latitude_moll==0))
first_idx, last_idx = np.min(indecesorbital), np.max(indecesorbital)

# Load data
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb = data[:, 0], data[:, 1]
snaps, tfb = sort_list([snaps, tfb], snaps)
snaps = np.array([int(snap) for snap in snaps])

if kind_of_plot == 'velocity_components':
    singlesnap = 50
    tfb_single = tfb[np.argmin(np.abs(snaps - singlesnap))]
    xph_s, yph_s, zph_s, volph_s, denph_s, Tempph_s, Rad_denph_s, Vxph_s, Vyph_s, Vzph_s, Pressph_s, IE_denph_s, _, _, _, _ = \
        np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{singlesnap}.txt')
    rph_s = np.sqrt(xph_s**2 + yph_s**2 + zph_s**2)
    v_mag_s = np.sqrt(Vxph_s**2 + Vyph_s**2 + Vzph_s**2)
    v_rad_ph_s, v_theta_ph_s, v_phi_s= to_spherical_components(Vxph_s, Vyph_s, Vzph_s, xph_s, yph_s, zph_s)
    v_rad_ph_s, v_theta_ph_s, v_phi_s, v_mag_s, denph_s, rph_s = sort_list([v_rad_ph_s, v_theta_ph_s, v_phi_s, v_mag_s, denph_s, rph_s], rph_s)
    # r_esc = prel.G*Mbh/(5e3)**2 # R_esc for v= 5000 km/s
    # v_esc_Rp = np.sqrt(2*prel.G*Mbh/Rt)
    # print(f'Escape radius [R_a] for the furthest point: {r_esc/apo}')
    # print(f'Escape velocity at Rp [km/s]: {v_esc_Rp*prel.Rsol_cgs*1e-5/prel.tsol_cgs}')
    # convert the velocty to km/s and the density to g/(scm2)
    v_rad_ph_s_kms = v_rad_ph_s * prel.Rsol_cgs*1e-5/prel.tsol_cgs
    v_theta_ph_s_kms = v_theta_ph_s * prel.Rsol_cgs*1e-5/prel.tsol_cgs
    v_phi_s_kms = v_phi_s * prel.Rsol_cgs*1e-5/prel.tsol_cgs
    v_mag_s_kms = v_mag_s * prel.Rsol_cgs*1e-5/prel.tsol_cgs

    # Plot velocity and radial velocity
    img, ax1 = plt.subplots(1, figsize = (9, 5))
    ax1.plot(rph_s/apo, np.abs(v_rad_ph_s/v_mag_s), c = 'firebrick', label = r'v$_r$')
    ax1.plot(rph_s/apo, np.abs(v_theta_ph_s/v_mag_s), c = 'dodgerblue', label = r'v$_\theta$')
    ax1.plot(rph_s/apo, np.abs(v_phi_s/v_mag_s), c = 'forestgreen', label = r'v$_\phi$')
    ax1.set_ylabel(r'$|v_i/v_{\rm tot}|$')
    ax1.legend(fontsize = 12)
    ax1.set_yscale('log')
    ax1.grid()
    ax1.set_xlabel(r'R$_{\rm ph} [R_a$]')
    ax1.set_title(f't = {np.round(tfb_single, 2)}' + r' t$_{\rm fb}$', fontsize = 15)
    plt.tight_layout()
    # plt.savefig(f'{imgsaving_folder}/velph_{singlesnap}.png')

# ------------------ Evolution in time (bound, unboud, velocity and dispersion)
mean_vel = np.zeros(len(snaps))
percentile16 = np.zeros(len(snaps))
percentile84 = np.zeros(len(snaps))
ratio_unbound_ph = np.zeros(len(snaps))
median_rph = np.zeros(len(snaps))
percentile_vel_ph_16 = np.zeros(len(snaps))
percentile_vel_ph_84 = np.zeros(len(snaps))
xlim_min, xlim_max = -2, 2
ylim_min, ylim_max = -1.9, 1.9
scale_v = 40
for i, snap in enumerate(snaps):
    print(snap)
    x_ph, y_ph, z_ph, vol_ph, den_ph, Temp_ph, Rad_den_ph, Vx_ph, Vy_ph, Vz_ph, Press_ph, IE_den_ph, _, _, _, _ = \
        np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
    vel_ph = np.sqrt(Vx_ph**2 + Vy_ph**2 + Vz_ph**2)
    mass_ph = den_ph * vol_ph

    oe_ph = orb.orbital_energy(r_ph, vel_ph, mass_ph, params, prel.G)
    # bern_ph = orb.bern_coeff(r_ph, vel_ph, den_ph, mass_ph, Press_ph, IE_den_ph, Rad_den_ph, params)
    mean_vel[i] = np.mean(vel_ph)
    ratio_unbound_ph[i] = len(oe_ph[np.logical_and(oe_ph>=0, r_ph!=0)]) / len(oe_ph)  
    median_rph[i] = np.median(r_ph)
    percentile16[i] = np.percentile(r_ph, 16) 
    percentile84[i] = np.percentile(r_ph, 84)
    # Plot
    if kind_of_plot == 'ratioE': # slides of boundness/unboundness with velocity arrows, (if how_amy==3, also time evolution of velocity)
        # if snap != 238:
        #     continue
        # print(median_rph[i]/percentile16[i])
        # print(percentile84[i]/median_rph[i])
        x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, IE_den_mid, Rad_den_mid, VX_mid, VY_mid, VZ_mid, Diss_den_mid, IE_den_mid, Press_mid =\
            np.load(f'{abspath}/data/{folder}/slices/z/z0slice_{snap}.npy')
        r_mid = np.sqrt(x_mid**2 + y_mid**2 + z_mid**2)
        vel_mid = np.sqrt(VX_mid**2 + VY_mid**2 + VZ_mid**2)
        mass_mid = den_mid * dim_mid**3 
        bern_mid = orb.bern_coeff(r_mid, vel_mid, den_mid, mass_mid, Press_mid, IE_den_mid, Rad_den_mid, params)
        # print(np.max(bern_mid), np.min(bern_mid))
        xph_mid, yph_mid, zph_mid, rph_mid = \
            x_ph[indecesorbital], y_ph[indecesorbital], z_ph[indecesorbital], r_ph[indecesorbital]

        x_yz, y_yz, z_yz, dim_yz, den_yz, temp_yz, ie_den_yz, Rad_den_yz, VX_yz, VY_yz, VZ_yz, Diss_den_yz, IE_den_yz, Press_yz =\
            np.load(f'{abspath}/data/{folder}/slices/x/xRpslice_{snap}.npy')
        r_yz = np.sqrt(x_yz**2 + y_yz**2 + z_yz**2)
        vel_yz = np.sqrt(VX_yz**2 + VY_yz**2 + VZ_yz**2)
        mass_yz = den_yz * dim_yz**3
        bern_yz = orb.bern_coeff(r_yz, vel_yz, den_yz, mass_yz, Press_yz, ie_den_yz, Rad_den_yz, params)

        yz_ph = np.abs(x_ph-Rt) < vol_ph**(1/3)
        yph_yz, zph_yz, Vy_ph_yz, Vz_ph_yz, rph_yz = make_slices([y_ph, z_ph, Vy_ph, Vz_ph, r_ph], yz_ph)
        # ratio_bound_ph_mid = len(energy_mid[energy_mid<0]) / len(energy_mid[energy_mid>0])
        long_ph_mid = np.arctan2(yph_mid, xph_mid)          # Azimuthal angle in radians
        falselong_ph_yz = np.arctan2(zph_yz, yph_yz)                # Elevation angle in radians

        fig = plt.figure(figsize=(30, 10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[1, 0.05], hspace=0.25, wspace = 0.2)
        ax3 = fig.add_subplot(gs[0, 2])
        img = ax3.scatter(tfb[:i+1], np.array(median_rph[:i+1])/Rp, s = 25, c = 'k')
        ax3.fill_between(tfb[:i+1], np.array(percentile16[:i+1])/Rp, np.array(percentile84[:i+1])/Rp, color = 'gray', alpha = 0.2)
        ax3.set_xlabel(r't $[t_{\rm fb}]$')
        ax3.set_ylabel(r'Median r$_{\rm ph}$ [$r_{\rm p}$]', fontsize = 25)
        ax3.set_xlim(0.35, 1.55)
        ax3.set_ylim(1, 50)
        ax3.set_yscale('log')
        ax3.grid()          
    
        ax1 = fig.add_subplot(gs[0, 0])
        sorted_indices = np.argsort(long_ph_mid)  # Sorting by y-coordinate
        xph_mid_sorted = xph_mid[sorted_indices]
        yph_mid_sorted = yph_mid[sorted_indices] 
        img = ax1.scatter(x_mid/apo, y_mid/apo, c = bern_mid, s = 20, cmap = 'coolwarm', alpha = 0.5, vmin = -5e2, vmax = 5e2)
        ax1.scatter(xph_mid/apo, yph_mid/apo, facecolor = 'none', s = 60, edgecolors = 'k')
        ax1.plot(xph_mid_sorted/apo, yph_mid_sorted/apo, c = 'k')
        # connect the last and first point
        ax1.plot([xph_mid_sorted[-1]/apo, xph_mid_sorted[0]/apo], [yph_mid_sorted[-1]/apo, yph_mid_sorted[0]/apo], c = 'k', linewidth = 1.5)
        ax1.quiver(xph_mid/apo, yph_mid/apo, Vx_ph[indecesorbital]/scale_v, Vy_ph[indecesorbital]/scale_v, angles='xy', scale_units='xy', scale=0.7, color="k", width=0.003, headwidth = 6)
        ax1.set_xlabel(r'X [$r_{\rm a}$]', fontsize = 25)
        ax1.set_ylabel(r'Y [$r_{\rm a}$]', fontsize = 25)
        ax1.set_xlim(xlim_min, xlim_max)
        ax1.set_ylim(ylim_min, ylim_max)
        ax1.text(.9*xlim_min, 0.85*ylim_max, f't = {np.round(tfb[i],2)}' + r' t$_{\rm fb}$', color = 'k', fontsize = 26)
        ax1.text(.9*xlim_min, 0.85*ylim_min, r'z = 0', fontsize = 25)

        ax2 = fig.add_subplot(gs[0, 1])
        # # Apply sorting
        sorted_indices_yz = np.argsort(falselong_ph_yz)  # Sorting by y-coordinate
        yph_yz_sorted = yph_yz[sorted_indices_yz]
        zph_yz_sorted = zph_yz[sorted_indices_yz]
        img = ax2.scatter(y_yz/apo, z_yz/apo, c = bern_yz, s = 20, cmap = 'coolwarm', alpha = 0.5, vmin = -5e2, vmax = 5e2)
        ax2.scatter(yph_yz/apo, zph_yz/apo, facecolor = 'none', s = 60, edgecolors = 'k')
        ax2.plot(yph_yz_sorted/apo, zph_yz_sorted/apo, c = 'k')
        # if yph_yz_sorted is not empty, connect the last and first point
        if len(yph_yz_sorted) > 0:
            ax2.plot([yph_yz_sorted[-1]/apo, yph_yz_sorted[0]/apo], [zph_yz_sorted[-1]/apo, zph_yz_sorted[0]/apo], c = 'k', linewidth = 1.5)
        ax2.quiver(yph_yz/apo, zph_yz/apo, Vy_ph_yz/scale_v, Vz_ph_yz/scale_v, angles='xy', scale_units='xy', scale=0.7, color="k", width=0.003, headwidth = 6)
        ax2.set_xlabel(r'Y [$r_{\rm a}$]', fontsize = 25)
        ax2.set_ylabel(r'Z [$r_{\rm a}$]', fontsize = 25)
        ax2.set_xlim(xlim_min, xlim_max)
        ax2.set_ylim(ylim_min, ylim_max)
        ax2.text(.9*xlim_min, 0.85*ylim_max, f't = {np.round(tfb[i]*t_fall_hour,1)}' + r' hours', color = 'k', fontsize = 26)
        ax2.text(.9*xlim_min, 0.85*ylim_min, r'x = r$_{\rm p}$', fontsize = 25)

        cbar_ax = fig.add_subplot(gs[1, 0:2])  # Colorbar subplot below the first two
        cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cb.set_label(r'$\mathcal{B}$', fontsize = 28)
        cb.ax.tick_params(labelsize=30)
        cb.set_ticks([-5e2, 0, 5e2])
        cb.set_ticklabels(['Bound', '', 'Unbound'])
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='both', which='major', width=1.5, length=9, color = 'k')
            ax.tick_params(axis='both', which='minor', width=1, length=6, color = 'k')
            if ax != ax3:
                ax.scatter(0,0, c= 'k', marker = 'x', s=100)

        plt.savefig(f'{abspath}/Figs/{folder}/Outflow/B_slice_{snap}.png', bbox_inches='tight')
        plt.close()

if kind_of_plot == 'time_evolution' or kind_of_plot == 'convergence': # only evolution of velocity/boundness in Fid and High res
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    # img = plt.scatter(tfb, mean_vel * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 20, vmin = 0, vmax = 0.8, label = 'Fid')
    # cbar = plt.colorbar(img)
    # cbar.set_label('unbound/tot')
    # plt.plot(tfb, percentile16 * conversion_sol_kms * 1e-4, c = 'yellowgreen', alpha = 0.1, linestyle = '--')
    # plt.plot(tfb, percentile84 * conversion_sol_kms * 1e-4, c = 'yellowgreen', alpha = 0.1, linestyle = '--')
    # plt.fill_between(tfb, percentile16 * conversion_sol_kms * 1e-4, percentile84 * conversion_sol_kms * 1e-4, color = 'yellowgreen', alpha = 0.1)
    # plt.ylabel(r'Mean velocity [$10^4$ km/s] ')
    # plt.ylim(-0.01, 2)

    ax.plot(tfb, ratio_unbound_ph, c = 'yellowgreen', label = 'Fid') #c = mean_vel/v_esc, s = 20, vmin = 0.2, vmax = 1)
    # cbar = plt.colorbar(img)
    # cbar.set_label(r'mean velocity [v$_{\rm esc} (r_{\rm p})$]')
    ax.set_xlim(-0.09, 1.8)
    ax.set_xlabel(r'$t [t_{\rm fb}]$')
    ax.set_ylabel(r'f$\equiv N_{\rm ph, unb}/N_{\rm ph, tot}$')
    # plt.title(f'Photospheric cells', fontsize = 20)
    plt.tight_layout()
    plt.grid() 
    
    if kind_of_plot == 'convergence':
        dataL = np.loadtxt(f'{abspath}/data/{commonfolder}LowResNewAMR/LowResNewAMR_red.csv', delimiter=',', dtype=float)
        snapsL, tfbL = dataL[:, 0], dataL[:, 1]
        snapsL = np.array([int(snap) for snap in snapsL])
        snapsL, tfbL = sort_list([snapsL, tfbL], snapsL)
        mean_velL = np.zeros(len(snapsL))
        percentile16L = np.zeros(len(snapsL))
        percentile84L = np.zeros(len(snapsL))
        ratio_unbound_phL = np.zeros(len(snapsL))

        for j, snap_sH in enumerate(snapsL):
            x_phL, y_phL, z_phL, vol_phL, den_phL, Temp_phL, Rad_den_phL, Vx_phL, Vy_phL, Vz_phL, Press_phL, IE_den_phL, _, _, _, _ = \
                np.loadtxt(f'{abspath}/data/{commonfolder}LowResNewAMR/photo/LowResNewAMR_photo{snap_sH}.txt')
            r_phL = np.sqrt(x_phL**2 + y_phL**2 + z_phL**2)
            vel_phL = np.sqrt(Vx_phL**2 + Vy_phL**2 + Vz_phL**2)
            mean_velL[j] = np.mean(vel_phL)
            percentile16L[j] = np.percentile(vel_phL, 16)
            percentile84L[j] = np.percentile(vel_phL, 84)

            mass_phL = den_phL * vol_phL
            oe_L = orb.orbital_energy(r_phL, vel_phL, mass_phL, params, prel.G)
            # bern_phL = orb.bern_coeff(r_phL, vel_phL, den_phL, mass_phL, Press_phL, IE_den_phL, Rad_den_phL, params)
            ratio_unbound_phL[j] = len(oe_L[np.logical_and(oe_L>=0, r_phL!=0)]) / len(oe_L)

        dataH = np.loadtxt(f'{abspath}/data/{commonfolder}HiResNewAMR/HiResNewAMR_red.csv', delimiter=',', dtype=float)
        snapsH, tfbH = dataH[:, 0], dataH[:, 1]
        snapsH = np.array([int(snap) for snap in snapsH])
        snapsH, tfbH = sort_list([snapsH, tfbH], snapsH)
        mean_velH = np.zeros(len(snapsH))
        percentile16H = np.zeros(len(snapsH))
        percentile84H = np.zeros(len(snapsH))
        ratio_unbound_phH = np.zeros(len(snapsH))

        for j, snap_sH in enumerate(snapsH):
            x_phH, y_phH, z_phH, vol_phH, den_phH, Temp_phH, Rad_den_phH, Vx_phH, Vy_phH, Vz_phH, Press_phH, IE_den_phH, _, _, _, _ = \
                np.loadtxt(f'{abspath}/data/{commonfolder}HiResNewAMR/photo/HiResNewAMR_photo{snap_sH}.txt')
            r_phH = np.sqrt(x_phH**2 + y_phH**2 + z_phH**2)
            vel_phH = np.sqrt(Vx_phH**2 + Vy_phH**2 + Vz_phH**2)
            mean_velH[j] = np.mean(vel_phH)
            percentile16H[j] = np.percentile(vel_phH, 16)
            percentile84H[j] = np.percentile(vel_phH, 84)

            mass_phH = den_phH * vol_phH
            oe_H = orb.orbital_energy(r_phH, vel_phH, mass_phH, params, prel.G)
            # bern_phH = orb.bern_coeff(r_phH, vel_phH, den_phH, mass_phH, Press_phH, IE_den_phH, Rad_den_phH, params)
            ratio_unbound_phH[j] = len(oe_H[np.logical_and(oe_H>=0, r_phH!=0)]) / len(oe_H)
 
            # if snap_sH == 53: #53
            #     fig50, ax50 = plt.subplots()
            #     ax50.scatter(x_phH[bern_phH>0]/apo, r_phH[bern_phH>0]/apo, c = 'r', s = 10, label = 'Unbound')
            #     ax50.scatter(x_phH[bern_phH<0]/apo, r_phH[bern_phH<0]/apo, c = 'b', s = 10, label = 'Bound')
            #     ax50.set_title(f'f = {np.round(ratio_unbound_phH[j],2)}, t = {np.round(tfbH[j],2)}' + r't$_{\rm fb}$')
            #     ax50.set_xlabel(r'x$_{\rm ph}/R_a$')
            #     ax50.set_ylabel(r'r$_{\rm ph}/R_a$')
            #     ax50.set_yscale('log')
            #     ax50.legend()
            #     fig50.savefig(f'{abspath}/Figs/next_meeting/bern_phHiRes_{snap_sH}.png')
                

        ax.plot(tfbL, ratio_unbound_phL, c = 'C1', label = 'Low')
        # plt.scatter(tfbH, mean_velH * conversion_sol_kms * 1e-4, c = ratio_unbound_phH, s = 20, vmin = 0, vmax = 0.8, marker = 's', label = 'High')
        # plt.plot(tfbH, percentile16H * conversion_sol_kms * 1e-4, c = 'darkviolet', alpha = 0.1, linestyle = '--')
        # plt.plot(tfbH, percentile84H * conversion_sol_kms * 1e-4, c = 'darkviolet', alpha = 0.1, linestyle = '--')
        # plt.fill_between(tfbH, percentile16H * conversion_sol_kms * 1e-4, percentile84H * conversion_sol_kms * 1e-4, color = 'darkviolet', alpha = 0.1)
        ax.plot(tfbH, ratio_unbound_phH, c = 'darkviolet', label = 'High') #c = mean_velH/v_esc, s = 20, vmin = 0.2, vmax = 1)
        # plt.axvline(tfbH[np.argmin(np.abs(tfbH-0.5))], c = 'k', linestyle = '--')
        fig.legend(fontsize = 16)
        fig.suptitle(f'Convergence with OE', fontsize = 20)
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/next_meeting/f_conv.png')
        # plt.savefig(f'{abspath}/Figs/next_meeting/velPh_conv.png')
    else:
        fig.suptitle(f'Boundness with OE', fontsize = 20)
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/next_meeting/f{check}.png')

# %%
