""" Plot the photosphere."""
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src import orbits as orb
import matplotlib.colors as colors
import Utilities.prelude as prel
import healpy as hp
from Utilities.sections import make_slices
from sklearn.neighbors import KDTree
from Utilities.operators import make_tree
matplotlib.rcParams['figure.dpi'] = 150

#%%
abspath = '/Users/paolamartire/shocks'
G = 1
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
snap = '348'
compton = 'Compton'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'{abspath}/TDE/{folder}/{snap}'

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
Rp =  Rt / beta
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

path = f'{abspath}/TDE/{folder}/{snap}' #f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
data = make_tree(path, 348, energy = False)
X, Y, Z, Den, Temp = data.X, data.Y, data.Z, data.Den, data.Temp
mid = np.abs(Z) < data.Vol**(1/3)
X_mid, Y_mid, Z_mid, Den_mid, Temp_mid = make_slices([X, Y, Z, Den, Temp], mid)
#%% HEALPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # 3xN
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
# Convert to latitude and longitude
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 
indecesorbital = np.concatenate(np.where(latitude_moll==0))
long_orb, lat_orb = longitude_moll[indecesorbital], latitude_moll[indecesorbital]

print('Longitude from HELAPIX min and max: ' , np.min(theta), np.max(theta))
print('Longitude for mollweide min and max: ', np.min(longitude_moll), np.max(longitude_moll))
print('Latitude from HELAPIX min and max: ' , np.min(phi), np.max(phi))
print('Latitude for mollweide min and max: ', np.min(latitude_moll), np.max(latitude_moll))
print('So we shift of pi/2 the latitude')
# Plot in 2D using a Mollweide projection
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
img = ax.scatter(longitude_moll, latitude_moll, s=20, c=np.arange(len(longitude_moll)), cmap='viridis')
# ax.scatter(longitude_moll[first_eq:final_eq], latitude_moll[first_eq:final_eq], s=10, c='r')
ax.scatter(long_orb, lat_orb, s=10, c='r')
plt.colorbar(img, ax=ax, label='Observer Number')
ax.set_title("Observers on the Sphere (Mollweide Projection)")
ax.grid(True)
ax.set_xticks(np.radians(np.linspace(-180, 180, 9)))
ax.set_xticklabels(['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°', '180°'])
plt.tight_layout()
plt.show()

#%% Mine observers (200 on Orbital plane)
mine_data = np.load(f'{abspath}/data/{folder}/_observers200_{snap}.npy')
mineR_initial, mineObs_x, mineObs_y, mineObs_z = mine_data[0], mine_data[1], mine_data[2], mine_data[3]
mine_Robs = np.sqrt(mineObs_x**2 + mineObs_y**2 + mineObs_z**2)
mine_ph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{snap}_OrbPl200.txt')
xph_mine, yph_mine, zph_mine, volph_mine= mine_ph[0], mine_ph[1], mine_ph[2], mine_ph[3]
rph_mine = np.sqrt(xph_mine**2 + yph_mine**2 + zph_mine**2)
# midplane OBSERVERS 
midObs_mine = np.abs(mineObs_z) == 0
mineR_initial_mid, mineObs_x_mid, mineObs_y_mid, mineObs_z_mid, mine_rObs_mid = \
        make_slices([mineR_initial, mineObs_x, mineObs_y, mineObs_z, mine_Robs], midObs_mine)
xph_mine_mid, yph_mine_mid, zph_mine_mid, rph_mid_mine = \
        make_slices([xph_mine, yph_mine, zph_mine, rph_mine], midObs_mine)
#%% Helapix observers (the ones of the light curve)
dataHealp = np.load(f'{abspath}/data/{folder}/_observersHealpix_{snap}.npy')
healpR_initial, healpObs_x, healpObs_y, healpObs_z = dataHealp[0], dataHealp[1], dataHealp[2], dataHealp[3]
heal_Robs = np.sqrt(healpObs_x**2 + healpObs_y**2 + healpObs_z**2)
# midplane helapix OBSERVERS
photoHeal = np.loadtxt(f'{abspath}/data/{folder}/photo/_photo{snap}.txt')
xph_Heal, yph_Heal, zph_Heal, vol_Heal = photoHeal[0], photoHeal[1], photoHeal[2], photoHeal[3] 
rph_Heal = np.sqrt(xph_Heal**2 + yph_Heal**2 + zph_Heal**2)
mid_Heal = healpObs_z == 0 #np.abs(zph_Heal) < vol_Heal**(1/3)
healpR_initial_Heal, healpObs_x_mid, healpObs_y_mid, healpObs_z_mid, heal_rObs_mid = \
        make_slices([healpR_initial, healpObs_x, healpObs_y, healpObs_z, heal_Robs], mid_Heal)
xph_Heal_mid, yph_Heal_mid, zph_Heal_mid, rph_Heal_mid = \
        make_slices([xph_Heal, yph_Heal, zph_Heal, rph_Heal], mid_Heal)
#%% Plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(mineObs_x/apo, mineObs_y/apo, s = 10, c = 'r', label = 'Mine')
ax.scatter(healpObs_x/apo, healpObs_y/apo, s = 10, c = 'b', label = 'Healpix')
ax.legend()
ax.set_title('Initial position of the observers')
ax.set_xlabel(r'x [R$_a$]')
ax.set_ylabel(r'y [R$_a$]')
plt.savefig(f'{abspath}/Figs/Test/photosphere/observersOrbPl.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.scatter(mineR_initial*mine_Robs/apo, rph_mine/apo, c = 'r', label = 'all mine')
ax.scatter(mineR_initial_mid*mine_rObs_mid/apo, rph_mid_mine/apo, c = 'orange', s = 5, label = 'orbital plane mine')
ax.scatter(healpR_initial*heal_Robs/apo, rph_Heal/apo, c = 'navy', label = 'all Healpix')
ax.scatter(healpR_initial_Heal*heal_rObs_mid/apo, rph_Heal_mid/apo, c = 'dodgerblue', s = 5, label = 'orbital plane Healpix')
# ax.axvline(x = Rt/apo, color = 'k', linestyle = 'dotted', label = r'$R_{\rm t}$')
ax.set_xlabel(r'R$_{initial}$ [R$_a$]')
ax.legend()
ax.set_ylabel(r'R$_{ph}$ [R$_a$]')
plt.savefig(f'{abspath}/Figs/Test/photosphere/initialR&Rph.png', bbox_inches='tight')

#%% find the mine_obs_mid that are nearer (as angles positions) to the healp_obs_mid 
theta_heal = np.arctan2(healpObs_y_mid, healpObs_x_mid)          # Azimuthal angle in radians
phi_heal = np.arccos(healpObs_z_mid / healpR_initial_Heal)
theta_mine = np.arctan2(mineObs_y_mid, mineObs_x_mid)          # Azimuthal angle in radians
phi_mine = np.arccos(mineObs_z_mid / mineR_initial_mid)
points_toquery = np.transpose([theta_heal, phi_heal])
tree = KDTree(np.transpose([theta_mine, phi_mine]))
_, idx = tree.query(points_toquery, k=1)
mineHeal_Rinitial_mid = mineR_initial_mid[idx]
mineHeal_rObs_mid = mine_rObs_mid[idx]
mineHeal_xph_mid, mineHeal_yph_mid, mineHeal_rph_mid = make_slices([xph_mine_mid, yph_mine_mid, rph_mid_mine], idx)

#%%
print('R obs help:', heal_Robs)
print('R obs mine:', mine_Robs)
#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
ax1.scatter(mineHeal_Rinitial_mid*mineHeal_rObs_mid/apo, mineHeal_rph_mid/apo, s = 300, c = np.arange(len(mineHeal_rph_mid)), label = 'Mine', marker = '*', cmap = 'jet', edgecolors = 'k')
ax1.scatter(healpR_initial_Heal*heal_rObs_mid/apo, rph_Heal_mid/apo, s = 200, c = np.arange(len(mineHeal_rph_mid)), label = 'Healpix', marker = 'o', cmap = 'jet', edgecolors = 'k')
ax1.set_xlabel(r'R$_{initial}$ [R$_a$]', fontsize = 27)
ax1.set_ylabel(r'R$_{ph}$ [R$_a$]', fontsize = 27)
ax1.legend(fontsize = 27)

ax2.scatter(X_mid/apo, Y_mid/apo, s = 10, c = Den_mid, cmap = 'gray', norm = colors.LogNorm(vmin=1e-15, vmax=1e-4))
ax2.scatter(mineHeal_xph_mid/apo, mineHeal_yph_mid/apo, c = np.arange(len(mineHeal_rph_mid)), s = 300, marker = '*', cmap = 'jet', edgecolors = 'k')
img = ax2.scatter(xph_Heal_mid/apo, yph_Heal_mid/apo, c = np.arange(len(mineHeal_rph_mid)), s = 200, marker = 'o', cmap = 'jet', edgecolors = 'k')
cbar = plt.colorbar(img, orientation='horizontal', pad = 0.15)
cbar.set_label('Observer Number')
ax2.set_xlim(-9,2.5)
ax2.set_ylim(-5,3)
ax2.set_xlabel(r'X [R$_a$]', fontsize = 27)
ax2.set_ylabel(r'Y [R$_a$]', fontsize = 27)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/Test/photosphere/initialR&Rph_mid.png', bbox_inches='tight')

#%% Helapix observers substituting the ones on the orbital plane with mine
dataH200 = np.load(f'{abspath}/data/{folder}/_observersHealp200_348.npy')
H200R_initial, H200Obs_x, H200Obs_y, H200Obs_z = dataH200[0], dataH200[1], dataH200[2], dataH200[3]
H200_Robs = np.sqrt(H200Obs_x**2 + H200Obs_y**2 + H200Obs_z**2)
data_phH200 = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo348_Healp200.txt')
xph_H200, yph_H200, zph_H200, volph_H200= data_phH200[0], data_phH200[1], data_phH200[2], data_phH200[3]
rph_H200 = np.sqrt(xph_H200**2 + yph_H200**2 + zph_H200**2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
ax1.scatter(mineHeal_Rinitial_mid * mineHeal_rObs_mid/apo, mineHeal_rph_mid/apo, s = 700, c = np.arange(len(mineHeal_rph_mid)), label = 'Mine', marker = '*', cmap = 'jet', edgecolors = 'k')
ax1.scatter(H200R_initial*H200_Robs/apo, rph_H200/apo, c = np.arange(len(mineHeal_rph_mid)), label = 'Healpix with my distances', marker = 'H', cmap = 'jet', edgecolors = 'k')
ax1.set_xlabel(r'R$_{initial}$ [R$_a$]', fontsize = 27)
ax1.set_ylabel(r'R$_{ph}$ [R$_a$]', fontsize = 27)
ax1.legend(fontsize = 27)

ax2.scatter(X_mid/apo, Y_mid/apo, s = 10, c = Den_mid, cmap = 'gray', norm = colors.LogNorm(vmin=1e-15, vmax=1e-4))
ax2.scatter(mineHeal_xph_mid/apo, mineHeal_yph_mid/apo, c = np.arange(len(mineHeal_rph_mid)), s = 700, marker = '*', cmap = 'jet', edgecolors = 'k')
img = ax2.scatter(xph_H200/apo, yph_H200/apo, c = np.arange(len(mineHeal_rph_mid)), s = 200, marker = 'H', cmap = 'jet', edgecolors = 'k')
cbar = plt.colorbar(img, orientation='horizontal', pad = 0.15)
cbar.set_label('Observer Number')
ax2.set_xlim(-9,2.5)
ax2.set_ylim(-5,3)
ax2.set_xlabel(r'X [R$_a$]', fontsize = 27)
ax2.set_ylabel(r'Y [R$_a$]', fontsize = 27)
plt.tight_layout()

plt.savefig(f'{abspath}/Figs/Test/photosphere/initialR&Rph_midMIX.png', bbox_inches='tight')

# %%
