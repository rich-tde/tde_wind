""" Compute Mdot fallback and wind"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, single_branch, to_spherical_components
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt/beta
norm_dMdE = Mbh/Rt * (Mbh/Rstar)**(-1/3) # Normalisation (what on the x axis you call \Delta E). It's GM/Rt^2 * Rstar
apo = orb.apocentre(Rstar, mstar, Mbh, beta) 
amin = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)

#%% MAIN
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
tfb_cgs = tfb * tfallback_cgs #converted to seconds
bins = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}_bins.txt')
mid_points = (bins[:-1]+bins[1:])* norm_dMdE/2  # get rid of the normalization
# bins_cgs = bins * (prel.en_converter/prel.Msol_cgs) #  and convert to CGS (they are bins in SPECIFIC orbital energy)
dMdE_distr = np.loadtxt(f'{abspath}data/{folder}/dMdE_{check}.txt')[0] # distribution just after the disruption
bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies

mfall = np.zeros(len(tfb_cgs))
radii = 0.2 * amin
mwind = np.zeros(len(tfb_cgs))
# compute dM/dt = dM/dE * dE/dt
for i, snap in enumerate(snaps):
    if alice:
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
    # convert to code units
    t = tfb_cgs[i] 
    tsol = t / prel.tsol_cgs
    # Find the energy of the element at time t
    energy = orb.keplerian_energy(Mbh, prel.G, tsol)
    # Find the bin that corresponds to the energy of the element and its dMdE (in CGS)
    i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
    if energy/bins_tokeep[i_bin] > 2:
        print('You do not match the data in your time range')
    dMdE_t = dMdE_distr_tokeep[i_bin]
    mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
    mfall[i] = mdot # code units
    mdot_cgs = mdot * prel.Msol_cgs / prel.tsol_cgs # [g/s]

    data = make_tree(path, snap, energy = True)
    X, Y, Z, Mass, Den, VX, VY, VZ = \
        data.X, data.Y, data.Z, data.Mass, data.Den, data.VX, data.VY, data.VZ
    cut = Den > 1e-19
    X, Y, Z, Mass, Den, VX, VY, VZ = \
        make_slices([X, Y, Z, Mass, Den, VX, VY, VZ], cut)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    # xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
    #     np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    # xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
    #     np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    # rph = np.sqrt(xph**2 + yph**2 + zph**2)
    # idx_r_wind = np.argmin(np.abs(rph - radii))
    # x_w, y_w, z_w, r_w, vx_w, vy_w, vz_w = \
    #     xph[idx_r_wind], yph[idx_r_wind], zph[idx_r_wind], rph[idx_r_wind], Vxph[idx_r_wind], Vyph[idx_r_wind], Vzph[idx_r_wind]
    # long_w = np.arctan2(y_w, x_w)          # Azimuthal angle in radians
    # lat_w = np.arccos(z_w / r_w)
    # v_rad_w, _, _ = to_spherical_components(vx_w, vy_w, vz_w, lat_w, long_w)
    long = np.arctan2(Y, X)          # Azimuthal angle in radians
    lat = np.arccos(Z / Rsph)
    v_rad, _, _ = to_spherical_components(VX, VY, VZ, lat, long)
    Den_casted = single_branch(radii, Rsph, Den, weights = Mass)
    v_rad_casted = single_branch(radii, Rsph, v_rad, weights = Mass)
    mwind[i] = 4 * np.pi * radii**2 * Den_casted * v_rad_casted

with open(f'{abspath}/data/{folder}/EddingtonEnvelope/Mdot.txt','a') as file:
    file.write(f'# t/tfb \n')
    file.write(f' '.join(map(str, tfb)) + '\n')
    file.write(f'# Mdot_f \n')
    file.write(f' '.join(map(str, mfall)) + '\n')
    file.write(f'# Mdot_wind \n')
    file.write(f' '.join(map(str, mwind)) + '\n')
    file.close()