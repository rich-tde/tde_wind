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
check = ''
kind_of_plot = 'talk' # 'moll' or 'cart' or 'talk'
how_many = 3
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
t_fall_hour = t_fall * 24

# observers
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
# HEALPIX
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
# Convert to latitude and longitude
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 

# Load data
time = np.loadtxt(f'{abspath}/data/{folder}/slices/z/z0_time.txt')
snaps, tfb = time[0], time[1]
snaps = np.array([int(snap) for snap in snaps])

#%% Look at single snap
singlesnap = 348
tfb_single = tfb[np.argmin(np.abs(snaps - singlesnap))]
xph_s, yph_s, zph_s, volph_s, denph_s, Tempph_s, Rad_denph_s, Vxph_s, Vyph_s, Vzph_s = \
    np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{singlesnap}.txt')
rph_s = np.sqrt(xph_s**2 + yph_s**2 + zph_s**2)
v_mag_s = np.sqrt(Vxph_s**2 + Vyph_s**2 + Vzph_s**2)
long_ph_s = np.arctan2(yph_s, xph_s)          # Azimuthal angle in radians
lat_ph_s = np.arccos(zph_s/ rph_s)            # Elevation angle in radians
v_rad_ph_s, v_theta_ph_s, v_phi_s= to_spherical_components(Vxph_s, Vyph_s, Vzph_s, lat_ph_s, long_ph_s)
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
vrho_cgs = v_rad_ph_s * denph_s * prel.Msol_cgs/ (prel.tsol_cgs * prel.Rsol_cgs**2)
ysqrt = np.sqrt(apo/rph_s)
y2 = (rph_s/apo)**(-2)
y3 = (rph_s/apo)**(-3)
y4 = (rph_s/apo)**(-4)

#%% Plot density
plt.figure()
plt.scatter(rph_s/apo, denph_s*prel.den_converter, c = 'royalblue', label = 'photosphere')
plt.plot(rph_s/apo, 1e-11*y2, c = 'k', linestyle = 'dotted', label = r'$\propto R^{-2}$')
plt.plot(rph_s/apo, 1e-11*y3, c = 'k', linestyle = 'dashed', label = r'$\propto R^{-3}$')
plt.plot(rph_s/apo, 1e-11*y4, c = 'k', linestyle = 'dashdot', label = r'$\propto R^{-4}$')
# plt.ylim(1e-14, 5e-11)
plt.xlabel(r'R [R$_a$]')
plt.ylabel(r'$\rho$ [g/cm$^3$]')
plt.loglog()
plt.title(f'Photosphere, t: {np.round(tfb_single,2)}' + r' t$_{{\rm fb}}$', fontsize = 18)
plt.grid()
plt.legend(fontsize = 12)
plt.tight_layout()
plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/denph_{singlesnap}.png')

# Plot velocity and radial velocity
img, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
ax1.plot(rph_s/apo, np.abs(v_rad_ph_s_kms)*1e-3, c = 'firebrick', label = r'V$_r$')
ax1.plot(rph_s/apo, np.abs(v_theta_ph_s_kms)*1e-3, c = 'dodgerblue', label = r'V$_\theta$')
ax1.plot(rph_s/apo, np.abs(v_phi_s_kms)*1e-3, c = 'forestgreen', label = r'V$_\phi$')
ax1.plot(rph_s/apo, v_mag_s_kms*1e-3, c = 'orange', label = r'$||V||$', linestyle = '--')
ax1.plot(rph_s/apo, 10*ysqrt, c = 'k', linestyle = 'dotted', label = r'$\propto R^{-1/2}$')
ax1.set_ylabel(r'$|v| [10^3 $km/s]')
ax2.scatter(rph_s[vrho_cgs>=0]/apo, np.abs(vrho_cgs[vrho_cgs>0]), c = 'forestgreen', s = 7, label = r'+$\hat{r}$')
ax2.scatter(rph_s[vrho_cgs<0]/apo, np.abs(vrho_cgs[vrho_cgs<0]), c = 'firebrick', s = 7, label = r'-$\hat{r}$')
ax2.plot(rph_s/apo, 1e-3*y2, c = 'k', linestyle = 'dotted')
ax2.set_ylabel(r'$v_r\rho$ [gcm$^{-2}s^{-1}$]')
ax2.text(np.max(rph_s/apo)-1, 4e-6, r'$\propto R^{-2}$', fontsize = 15)
for ax in [ax1, ax2]:
    ax.legend(fontsize = 12)
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel(r'R [R$_a$]')
plt.suptitle(r'Photosphere in spherical $(r,\theta,\phi)$ components,' + f' t: {np.round(tfb_single,2)}' + r' t$_{{\rm fb}}$', fontsize = 18)
plt.tight_layout()
plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/velph_{singlesnap}.png')
#%% Evolution in time (bound, unboud, velocity and dispersion)
if kind_of_plot == 'moll' or kind_of_plot == 'talk':
    ratio_unbound_ph = []
    mean_vel = []
    percentile16 = []
    percentile84 = []
    if kind_of_plot == 'talk':
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

if kind_of_plot == 'cart' or kind_of_plot == 'convergence':
    ratio_unbound_ph = np.zeros(len(snaps))
    mean_vel = np.zeros(len(snaps))
    percentile16 = np.zeros(len(snaps))
    percentile84 = np.zeros(len(snaps))

if kind_of_plot == 'convergence':
    timeH = np.loadtxt(f'{abspath}/data/{folder}HiRes/slices/z/z0_time.txt')
    snapH, tfbH = timeH[0], timeH[1]
    snapH = np.array([int(snap) for snap in snapH])

    ratio_unbound_ph = np.zeros(len(snaps))
    mean_vel = np.zeros(len(snaps))
    percentile16 = np.zeros(len(snaps))
    percentile84 = np.zeros(len(snaps))
    ratio_unbound_phH = np.zeros(len(tfbH))
    mean_velH = np.zeros(len(tfbH))
    percentile16H = np.zeros(len(tfbH))
    percentile84H = np.zeros(len(tfbH))

    for j, snap_sH in enumerate(snapH):
        xphH, yphH, zphH, volphH, denphH, TempphH, Rad_denphH, VxphH, VyphH, VzphH = \
            np.loadtxt(f'{abspath}/data/{folder}HiRes/photo/HiRes_photo{snap_sH}.txt')
        rphH = np.sqrt(xphH**2 + yphH**2 + zphH**2)
        velH = np.sqrt(VxphH**2 + VyphH**2 + VzphH**2)

        mean_velH_sn = np.mean(velH)
        percentile16H_sn = np.percentile(velH, 16)
        percentile84H_sn = np.percentile(velH, 84)
        PE_ph_specH = -prel.G * Mbh / (rphH-Rs)
        KE_ph_specH = 0.5 * velH**2
        energyH = KE_ph_specH + PE_ph_specH
        ratio_unbound_phH_sn = len(energyH[energyH>0]) / len(energyH)

        ratio_unbound_phH[j] = ratio_unbound_phH_sn
        mean_velH[j] = mean_velH_sn
        percentile16H[j] = percentile16H_sn
        percentile84H[j] = percentile84H_sn

for i, snap in enumerate(snaps):
    print(snap)
    xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
        np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    rph = np.sqrt(xph**2 + yph**2 + zph**2)
    vel = np.sqrt(Vxph**2 + Vyph**2 + Vzph**2)

    mean_vel_sn = np.mean(vel)
    percentile16_sn = np.percentile(vel, 16)
    percentile84_sn = np.percentile(vel, 84)
    PE_ph_spec = -prel.G * Mbh / (rph-Rs)
    KE_ph_spec = 0.5 * vel**2
    energy = KE_ph_spec + PE_ph_spec
    ratio_unbound_ph_sn = len(energy[energy>0]) / len(energy)

    # Plot
    if kind_of_plot == 'moll' or kind_of_plot == 'talk': 
        ratio_unbound_ph.append(ratio_unbound_ph_sn)
        mean_vel.append(mean_vel_sn)
        percentile16.append(percentile16_sn)
        percentile84.append(percentile84_sn)

        if kind_of_plot == 'moll': # time evolution side by side of: mollweide to show the (un)bound observers 
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].axis('off') 
            ax1 = fig.add_subplot(121, projection='mollweide')
            ax1.scatter(longitude_moll[energy<0], latitude_moll[energy<0], s = 20, c = 'b', label = 'bound')
            ax1.scatter(longitude_moll[energy>0], latitude_moll[energy>0], s = 10, c = 'r', label = 'unbound')
            ax1.set_xticks(np.radians(np.linspace(-180, 180, 9)))
            # in moll longitude is from -180 to 180. I want the same but cloackwise
            ax1.set_xticklabels(['180°', '135°', '90°', '45°', '0°', '-45°', '-90°', '-135°', '-180°'], fontsize = 14)
            ax1.set_yticks(np.radians(np.linspace(-90, 90, 13)))
            ax1.set_yticklabels(['-90°', '-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '75°', '90°'], fontsize = 14)
            # ax1.legend()

            ax2 = ax[1] 
            img = ax2.scatter(tfb[:len(mean_vel)], np.array(mean_vel) * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 10, vmin = 0, vmax = 0.8)
            cbar = plt.colorbar(img)
            cbar.set_label('unbound/tot')
            ax2.plot(tfb[:len(mean_vel)], np.array(percentile16) * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
            ax2.plot(tfb[:len(mean_vel)], np.array(percentile84) * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
            ax2.fill_between(tfb[:len(mean_vel)], np.array(percentile16) * conversion_sol_kms * 1e-4, np.array(percentile84) * conversion_sol_kms * 1e-4, color = 'chocolate', alpha = 0.1)
            ax2.set_xlabel(r'$t_{\rm fb}$')
            ax2.set_ylabel(r'Mean velocity [$10^4$ km/s] ')
            ax2.set_xlim(0, np.max(tfb))
            ax2.set_ylim(-0.01, 2.3)
            ax2.text(1.35, 2.1, f't = {np.round(tfb[i], 2)}' + r' t$_{\rm fb}$', fontsize = 15)
            for ax in [ax1, ax2]:
                ax.grid()
            plt.tight_layout()
            # plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/unbound/unbound_ph{snap}.png')
            plt.savefig(f'/Users/paolamartire/shocks/Figs/talk/unbound_ph{snap}.png')
            plt.close()

        if kind_of_plot == 'talk': # slides of boundness/unboundness with velocity arrows, time evolution of velocity
            # if int(snap) < 2:
            #     continue
            data_mid = np.load(f'{abspath}/data/{folder}/slices/z/z0slice_{snap}.npy')
            x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid, VX_mid, VY_mid, VZ_mid =\
                data_mid[0], data_mid[1], data_mid[2], data_mid[3], data_mid[4], data_mid[5], data_mid[6], data_mid[7], data_mid[8], data_mid[9], data_mid[10], data_mid[11]
            ie_spec_mid = ie_den_mid / den_mid
            orb_en_spec_mid = orb_en_den_mid / den_mid
            mass_mid = den_mid * dim_mid**3
            V_mid = np.sqrt(VX_mid**2 + VY_mid**2 + VZ_mid**2)
            R_mid = np.sqrt(x_mid**2 + y_mid**2 + z_mid**2)
            KE = 0.5 * mass_mid * V_mid**2
            PE = - prel.G * Mbh * mass_mid / R_mid
            ratioE_mid = KE / np.abs(PE)
            ratio_IO_mid = ie_den_mid / orb_en_den_mid
            orb_en_mid = orb_en_den_mid * dim_mid**3

            data_yz = np.load(f'{abspath}/data/{folder}/slices/x/xRpslice_{snap}.npy')
            x_yz, y_yz, z_yz, dim_yz, den_yz, temp_yz, ie_den_yz, orb_en_den_yz, Rad_den_yz, VX_yz, VY_yz, VZ_yz =\
                data_yz[0], data_yz[1], data_yz[2], data_yz[3], data_yz[4], data_yz[5], data_yz[6], data_yz[7], data_yz[8], data_yz[9], data_yz[10], data_yz[11]
            mass_yz = den_yz * dim_yz**3
            V_yz = np.sqrt(VX_yz**2 + VY_yz**2 + VZ_yz**2)
            R_yz = np.sqrt(x_yz**2 + y_yz**2 + z_yz**2)
            KE = 0.5 * mass_yz * V_yz**2
            PE = - prel.G * Mbh * mass_yz / R_yz
            ratioE_yz = KE / np.abs(PE)
            ratio_IO_yz = ie_den_yz / orb_en_den_yz
            orb_en_yz = orb_en_den_yz * dim_yz**3

            yz_ph = np.abs(xph-Rt) < volph**(1/3)
            yph_yz, zph_yz, Vyph_yz, Vzph_yz, rph_yz = make_slices([yph, zph, Vyph, Vzph, rph], yz_ph)
            xph_mid, yph_mid, zph_mid, rph_mid, energy_mid = xph[indecesorbital], yph[indecesorbital], zph[indecesorbital], rph[indecesorbital], energy[indecesorbital]
            # ratio_bound_ph_mid = len(energy_mid[energy_mid<0]) / len(energy_mid[energy_mid>0])
            long_ph_mid = np.arctan2(yph_mid, xph_mid)          # Azimuthal angle in radians
            falselong_ph_yz = np.arctan2(zph_yz, yph_yz)                # Elevation angle in radians

            if how_many == 3:
                fig = plt.figure(figsize=(30, 10))
                gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[1, 0.05], hspace=0.25, wspace = 0.2)
                ax3 = fig.add_subplot(gs[0, 2])
                img = ax3.scatter(tfb[:len(mean_vel)], np.array(mean_vel) * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 25, vmin = 0, vmax = 0.8)
                cbar = plt.colorbar(img)
                cbar.set_label('Unbound/tot')
                ax3.plot(tfb[:len(mean_vel)], np.array(percentile16) * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
                ax3.plot(tfb[:len(mean_vel)], np.array(percentile84) * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
                ax3.fill_between(tfb[:len(mean_vel)], np.array(percentile16) * conversion_sol_kms * 1e-4, np.array(percentile84) * conversion_sol_kms * 1e-4, color = 'chocolate', alpha = 0.1)
                ax3.set_xlabel(r'$t_{\rm fb}$')
                ax3.set_ylabel(r'Mean velocity [$10^4$ km/s] ', fontsize = 25)
                ax3.set_xlim(0, np.max(tfb))
                ax3.set_ylim(0.1, 2.3)
                # ax3.text(1.25, 2.1, f't = {np.round(tfb[i], 2)}' + r' t$_{\rm fb}$', fontsize = 25)
                ax3.grid()          
            if how_many == 2:
                 fig = plt.figure(figsize=(20, 10))
                 gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1, 0.05], hspace=0.25, wspace = 0.2)
        
            ax1 = fig.add_subplot(gs[0, 0])
            sorted_indices = np.argsort(long_ph_mid)  # Sorting by y-coordinate
            xph_mid_sorted = xph_mid[sorted_indices]
            yph_mid_sorted = yph_mid[sorted_indices]
            img = ax1.scatter(x_mid/apo, y_mid/apo, c = np.abs(ratioE_mid), s = 20, cmap = 'coolwarm', norm = colors.LogNorm(vmin = 9.5e-2, vmax = 10))
            ax1.scatter(xph_mid/apo, yph_mid/apo, facecolor = 'none', s = 60, edgecolors = 'k')
            ax1.plot(xph_mid_sorted/apo, yph_mid_sorted/apo, c = 'k', alpha = 0.5)
            # connect the last and first point
            ax1.plot([xph_mid_sorted[-1]/apo, xph_mid_sorted[0]/apo], [yph_mid_sorted[-1]/apo, yph_mid_sorted[0]/apo], c = 'k', alpha = 0.5)
            ax1.quiver(xph_mid/apo, yph_mid/apo, Vxph[indecesorbital]/40, Vyph[indecesorbital]/40, angles='xy', scale_units='xy', scale=0.7, color="k", width=0.003, headwidth = 6)
            ax1.text(-2.6, -2.8, r'z = 0', fontsize = 25)
            ax1.text(-2.6, 1.65, f't = {np.round(tfb[i],2)}' + r' t$_{\rm fb}$', color = 'k', fontsize = 26)
            ax1.set_xlabel(r'X [$R_{\rm a}$]', fontsize = 25)
            ax1.set_ylabel(r'Y [$R_{\rm a}$]', fontsize = 25)
            ax1.set_xlim(-3, 3)
            ax1.set_ylim(-3, 2)

            ax2 = fig.add_subplot(gs[0, 1])
            # Apply sorting
            sorted_indices_yz = np.argsort(falselong_ph_yz)  # Sorting by y-coordinate
            yph_yz_sorted = yph_yz[sorted_indices_yz]
            zph_yz_sorted = zph_yz[sorted_indices_yz]
            img = ax2.scatter(y_yz/apo, z_yz/apo, c = np.abs(ratioE_yz), s = 20, cmap = 'coolwarm', norm = colors.LogNorm(vmin = 9.5e-2, vmax = 10))
            ax2.scatter(yph_yz/apo, zph_yz/apo, facecolor = 'none', s = 60, edgecolors = 'k')
            ax2.plot(yph_yz_sorted/apo, zph_yz_sorted/apo, c = 'k', alpha = 0.5)
            # if yph_yz_sorted is not empty, connect the last and first point
            if len(yph_yz_sorted) > 0:
                ax2.plot([yph_yz_sorted[-1]/apo, yph_yz_sorted[0]/apo], [zph_yz_sorted[-1]/apo, zph_yz_sorted[0]/apo], c = 'k', alpha = 0.5)
            ax2.quiver(yph_yz/apo, zph_yz/apo, Vyph_yz/40, Vzph_yz/40, angles='xy', scale_units='xy', scale=0.7, color="k", width=0.003, headwidth = 6)
            ax2.text(-2.6, -3.2, r'x = R$_{\rm p}$', fontsize = 25)
            ax2.text(-2.6, 3, f't = {np.round(tfb[i]*t_fall_hour,1)}' + r' hours', color = 'k', fontsize = 26)
            ax2.set_xlabel(r'Y [$R_{\rm a}$]', fontsize = 25)
            ax2.set_ylabel(r'Z [$R_{\rm a}$]', fontsize = 25)
            ax2.set_xlim(-3, 2)
            ax2.set_ylim(-3.5, 3.5)

            cbar_ax = fig.add_subplot(gs[1, 0:2])  # Colorbar subplot below the first two
            cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
            cb.set_label(r'$E_{\rm kin}/E_{\rm pot}$', fontsize = 28)
            cb.ax.tick_params(labelsize=30)
            # add the text "bound" and "unbound" at the extremities of the colorbar
            cb.ax.text(0.085, -2, 'Bound', fontsize = 26, color = 'k')
            cb.ax.text(8, -2, 'Unbound', fontsize = 26, color = 'k')
            for ax in [ax1, ax2]:
                ax.scatter(0,0, c= 'k', marker = 'x', s=80)

            plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/ratioE{check}/{how_many}/E_{snap}.png', bbox_inches='tight')
            plt.close()

    if kind_of_plot == 'cart' or kind_of_plot == 'convergence':
        ratio_unbound_ph[i] = ratio_unbound_ph_sn
        mean_vel[i] = mean_vel_sn
        percentile16[i] = percentile16_sn
        percentile84[i] = percentile84_sn

if kind_of_plot == 'cart': # only evolution of velocity/boundness
    plt.figure(figsize=(10,6))
    img = plt.scatter(tfb, mean_vel * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 7, vmin = 0, vmax = 0.8)
    cbar = plt.colorbar(img)
    cbar.set_label('unbound/tot')
    plt.plot(tfb, percentile16 * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
    plt.plot(tfb, percentile84 * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
    plt.fill_between(tfb, percentile16 * conversion_sol_kms * 1e-4, percentile84 * conversion_sol_kms * 1e-4, color = 'chocolate', alpha = 0.1)
    plt.grid()
    plt.xlabel(r'$t_{\rm fb}$')
    plt.ylabel(r'Mean velocity [$10^4$ km/s] ')
    plt.text(0.08, 2.01, f'{check}', fontsize = 25)
    plt.ylim(-0.01, 2.3)
    plt.xlim(-0.09, 1.8)
    plt.title(f'Photospheric cells', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/unbound/all{check}.png')
    plt.show()

if kind_of_plot == 'convergence': # only evolution of velocity/boundness
    plt.figure(figsize=(10,6))
    img = plt.scatter(tfb, mean_vel * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 7, vmin = 0, vmax = 0.8)
    plt.text(1.5, 0.6, f'Fid', fontsize = 25)
    plt.scatter(tfbH, mean_velH * conversion_sol_kms * 1e-4, c = ratio_unbound_phH, s = 7, vmin = 0, vmax = 0.8)
    plt.text(1, 0.45, f'High', fontsize = 25)
    cbar = plt.colorbar(img)
    cbar.set_label('unbound/tot')
    plt.plot(tfb, percentile16 * conversion_sol_kms * 1e-4, c = 'yellowgreen', alpha = 0.1, linestyle = '--')
    plt.plot(tfb, percentile84 * conversion_sol_kms * 1e-4, c = 'yellowgreen', alpha = 0.1, linestyle = '--')
    plt.fill_between(tfb, percentile16 * conversion_sol_kms * 1e-4, percentile84 * conversion_sol_kms * 1e-4, color = 'yellowgreen', alpha = 0.1)
    plt.plot(tfbH, percentile16H * conversion_sol_kms * 1e-4, c = 'darkviolet', alpha = 0.1, linestyle = '--')
    plt.plot(tfbH, percentile84H * conversion_sol_kms * 1e-4, c = 'darkviolet', alpha = 0.1, linestyle = '--')
    plt.fill_between(tfbH, percentile16H * conversion_sol_kms * 1e-4, percentile84H * conversion_sol_kms * 1e-4, color = 'darkviolet', alpha = 0.1)
    plt.grid()
    plt.xlabel(r'$t_{\rm fb}$')
    plt.ylabel(r'Mean velocity [$10^4$ km/s] ')
    plt.text(0.08, 2.01, f'{check}', fontsize = 25)
    plt.ylim(-0.01, 2)
    plt.xlim(-0.09, 1.8)
    plt.title(f'Photospheric cells', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/unbound/all_conv.png')
    plt.show()

#%% Plot for the talk: last snap
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].axis('off') 
    ax1 = fig.add_subplot(121, projection='mollweide')
    img = ax1.scatter(longitude_moll[energy>0], latitude_moll[energy>0], s = 10, c = vel[energy>0]* conversion_sol_kms * 1e-4, label = 'unbound')
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label('Velocity [$10^4$ km/s]')
    ax1.set_xticks(np.radians(np.linspace(-180, 180, 9)))
    ax1.set_xticklabels(['180°', '135°', '90°', '45°', '0°', '-45°', '-90°', '-135°', '-180°'], fontsize = 14)
    ax1.set_yticks(np.radians(np.linspace(-90, 90, 13)))
    ax1.set_yticklabels(['-90°', '-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '75°', '90°'], fontsize = 14)

    ax2 = ax[1] 
    img = ax2.scatter(tfb, mean_vel * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 7)
    cbar = plt.colorbar(img)
    cbar.set_label('unbound/tot')
    ax2.plot(tfb, percentile16 * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
    ax2.plot(tfb, percentile84 * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
    ax2.fill_between(tfb, percentile16 * conversion_sol_kms * 1e-4, percentile84 * conversion_sol_kms * 1e-4, color = 'chocolate', alpha = 0.1)
    ax2.grid()
    ax2.set_xlabel(r'$t_{\rm fb}$')
    ax2.set_ylabel(r'Mean velocity [$10^4$ km/s] ')
    ax2.set_ylim(-0.01, 2.3)
    for ax in [ax1, ax2]:
        ax.grid()
    plt.suptitle(f'Photospheric cells, t = {np.round(tfb[i], 2)}', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'/Users/paolamartire/shocks/Figs/talk/unbound_ph{snap}{check}.png')
    plt.show()
    # plt.close()
# %%
