""" Compute the accelerations due to centrifugal force and radial pressure to understand the Eddingotn envelope."""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.coordinates as coord
import Utilities.prelude as prel
from astropy import units as u
import src.orbits as orb
from Utilities.operators import make_tree
from Utilities.sections import make_slices
from Utilities.time_extractor import days_since_distruption
import healpy as hp

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
Rs = 2 * prel.G * Mbh / prel.csol_cgs**2
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps = [348]

Rt = Rstar * (Mbh/mstar)**(1/3)
apo = orb.apocentre(Rstar, mstar, Mbh, beta) 
amin = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
solarR_to_au = 215
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 
indecesorbital = np.concatenate(np.where(latitude_moll==0))

def dPdR(DpDx, DpDy, DpDz, lat, long):
    dP_radial = DpDx * np.sin(lat) * np.cos(long) + DpDy * np.sin(lat) * np.sin(long) + DpDz * np.cos(lat)
    return dP_radial

def v_phi(Vx, Vy, long):
    v_phi = - Vx * np.sin(long) + Vy * np.cos(long)
    return v_phi

def v_theta(Vx, Vy, Vz, lat, long):
    v_theta = Vx * np.cos(lat) * np.cos(long) + Vy * np.cos(lat) * np.sin(long) - Vz * np.sin(lat)
    return v_theta

for snap in snaps:
    dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{snap}.txt')
    xph, yph, zph, volph= dataph[0], dataph[1], dataph[2], dataph[3]
    midph= np.abs(zph) < volph**(1/3)
    xph_mid, yph_mid, zph_mid = make_slices([xph, yph, zph], midph)

    obs = np.load(f'/Users/paolamartire/shocks/data/{folder}/_observers200.npy')
    obs = np.array(obs).T
    x, y, z = obs[0], obs[1], obs[2]
    dataph8 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{snap}_OrbPl200.txt')
    xph8, yph8, zph8, volph8= dataph8[0], dataph8[1], dataph8[2], dataph8[3]
    midph8= np.abs(zph8) < volph8**(1/3)
    xph8_mid, yph8_mid, zph8_mid = make_slices([xph8, yph8, zph8], midph8)

    tfb = days_since_distruption(f'{abspath}/TDE/{folder}/{snap}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    # Load the data
    path = f'{abspath}/TDE/{folder}/{snap}' #f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = False)
    X, Y, Z, Den, Vx, Vy, Vz, Vol, Temp = data.X, data.Y, data.Z, data.Den, data.VX, data.VY, data.VZ, data.Vol, data.Temp
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    R, lat_astro, long = coord.cartesian_to_spherical(X, Y, Z) # r, latitude, longitude 
    lat = lat_astro + (np.pi / 2) * u.rad # so is from 0 to pi
    # orb_en = orb.orbital_energy(R, V, data.Mass, 1, 3e8 / (7e8/t), Mbh)
    DpDx = np.load(f'{path}/DpDx_{snap}.npy')
    DpDy = np.load(f'{path}/DpDy_{snap}.npy')
    DpDz = np.load(f'{path}/DpDz_{snap}.npy')
    dP_radial = dPdR(DpDx, DpDy, DpDz, lat, long)

    # compute accelerations
    a_P = np.abs(dP_radial) / Den
    V_phi = v_phi(Vx, Vy, long)
    a_centrifugal = V_phi**2 / R
    # V_theta = v_theta(Vx, Vy, Vz, lat, long)
    ratio = a_centrifugal / a_P
    a_grav = prel.G * Mbh / (R-Rs)**2
    ratio_env = (a_P + a_centrifugal)/a_grav

    #%%
    mid = np.logical_and(np.abs(Z) < Vol**(1/3), Den > 1e-19)
    x_mid, y_mid, z_mid, den_mid, ratio_mid, ratio_env_mid, temp_mid, vel_mid, Vx_mid, Vy_mid = make_slices([X, Y, Z, Den, ratio, ratio_env, Temp, V, Vx, Vy], mid)

    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(20, 7))
    img = ax1.scatter(x_mid/apo, y_mid/apo, c=Vx_mid, cmap = 'coolwarm', vmin = -1, vmax = 1, s = 7)
    cbar = plt.colorbar(img)
    cbar.set_label(r'$V_x$', fontsize = 20)
    img = ax2.scatter(x_mid/apo, y_mid/apo, c=Vy_mid, cmap = 'coolwarm', vmin = -1, vmax = 1, s = 7)
    cbar = plt.colorbar(img)
    cbar.set_label(r'$V_y$', fontsize = 20)
    img = ax3.scatter(x_mid/apo, y_mid/apo, c = ratio_env_mid, cmap = 'viridis', s = 7, vmin = 1e-2, vmax = 2)#norm=colors.LogNorm(vmin=1e-10, vmax=1e-4))
    cbar = plt.colorbar(img)
    cbar.set_label(r'$(a_{\rm \omega}+a_{\rm P})/a_{\rm g}$', fontsize = 20)
    img = ax4.scatter(x_mid/apo, y_mid/apo, c=ratio_mid, cmap = 'viridis', s = 7, norm=colors.LogNorm(vmin=1e-2, vmax=1e2))
    ax3.plot(xph[indecesorbital]/apo, yph[indecesorbital]/apo, c = 'k', markersize = 10, marker = 'H', label = r'$R_{\rm ph}$')
    ax3.plot(xph8_mid/apo, yph8_mid/apo, c = 'r', markersize = 10, marker = 'H', label = r'$R_{\rm ph}$')

    cbar = plt.colorbar(img)
    cbar.set_label(r'$a_{\rm \omega}/a_{\rm P}$', fontsize = 20)
    for ax in [ax1, ax2, ax3, ax4]:
        if ax in [ax1, ax3]:
            ax.set_ylabel(r'$Y [R_{\rm a}]$', fontsize = 20)
        if ax in [ax3, ax4]:
            ax.set_xlabel(r'$X [R_{\rm a}]$', fontsize = 20)
        ax.scatter(0, 0, c='k', s = 10)
        ax.set_xlim(-6, 2)
        ax.set_ylim(-3,2)#1, 1)
    plt.suptitle(f't = {np.round(tfb,2)}' + r' $t_{\rm fb}$', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/EddingtonEnvelope/accelerations{m}_{snap}big.png', bbox_inches = 'tight')


    

# %%
