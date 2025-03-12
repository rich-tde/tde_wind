""" Ibestigate the outflow. Look at the velocity of the photosphere"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(f'{abspath}')
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
# import colorcet
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb
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
kind_of_plot = 'cart' # 'moll' or 'cart'
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

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
snaps = [int(snap) for snap in snaps]

if kind_of_plot == 'moll':
    ratio_unbound_ph = []
    mean_vel = []
    percentile16 = []
    percentile84 = []
if kind_of_plot == 'cart':
    ratio_unbound_ph = np.zeros(len(snaps))
    mean_vel = np.zeros(len(snaps))
    percentile16 = np.zeros(len(snaps))
    percentile84 = np.zeros(len(snaps))
for i, snap in enumerate(snaps):
    print(snap)
    xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
        np.loadtxt(f'{abspath}/data/{folder}/photo/_photo{snap}.txt')
    rph = np.sqrt(xph**2 + yph**2 + zph**2)
    vel = np.sqrt(Vxph**2 + Vyph**2 + Vzph**2)

    mean_vel_sn = np.mean(vel)
    percentile16_sn = np.percentile(vel, 16)
    percentile84_sn = np.percentile(vel, 84)
    PE_ph_spec = -prel.G * Mbh / (rph-Rs)
    KE_ph_spec = 0.5 * vel**2
    energy = KE_ph_spec + PE_ph_spec
    ratio_unbound_ph_sn = len(energy[energy>0]) / len(energy)

    if kind_of_plot == 'moll': # time evolution side by side of: mollweide to show the (un)bound observers 
        ratio_unbound_ph.append(ratio_unbound_ph_sn)
        mean_vel.append(mean_vel_sn)
        percentile16.append(percentile16_sn)
        percentile84.append(percentile84_sn)

        # Plot
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

    if kind_of_plot == 'cart':
        ratio_unbound_ph[i] = ratio_unbound_ph_sn
        mean_vel[i] = mean_vel_sn
        percentile16[i] = percentile16_sn
        percentile84[i] = percentile84_sn

#%%
if kind_of_plot == 'cart': # only evolution of velocity/boundness
    plt.figure(figsize=(10,6))
    img = plt.scatter(tfb, mean_vel * conversion_sol_kms * 1e-4, c = ratio_unbound_ph, s = 7)
    cbar = plt.colorbar(img)
    cbar.set_label('unbound/tot')
    plt.plot(tfb, percentile16 * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
    plt.plot(tfb, percentile84 * conversion_sol_kms * 1e-4, c = 'chocolate', alpha = 0.1, linestyle = '--')
    plt.fill_between(tfb, percentile16 * conversion_sol_kms * 1e-4, percentile84 * conversion_sol_kms * 1e-4, color = 'chocolate', alpha = 0.1)
    plt.grid()
    plt.xlabel(r'$t_{\rm fb}$')
    plt.ylabel(r'Mean velocity [$10^4$ km/s] ')
    plt.ylim(-0.01, 2.3)
    plt.title(f'Photospheric cells', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/unbound/all.png')
    plt.show()

# Plot for the talk: last snap
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
    plt.savefig(f'/Users/paolamartire/shocks/Figs/talk/unbound_ph{snap}.png')
    plt.show()
    # plt.close()
# %%
