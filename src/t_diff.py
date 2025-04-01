"""Compute diffusion time"""
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = False

import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import nouveau_rich
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb


#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' 
snap = 237
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
Rt = orb.tidal_radius(Rstar, mstar, Mbh)
a_min = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #days
t_fall_cgs = t_fall * 24 * 3600

#%% matlab
eng = matlab.engine.start_matlab()
#%%
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)
N_ray = 5_000
Rt_obs = [Rt, 0, 0]
amin_obs = [a_min, 0, 0]
apo_obs = [apo, 0, 0]
observers_xyz = np.array(Rt_obs)#, amin_obs, apo_obs])

data = make_tree(f'{pre}/{snap}', snap, energy = True)
if alice:
    box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
else:
    box = np.load(f'{pre}/{snap}/box_{snap}.npy')

X, Y, Z, T, Den, Rad, Vol, VX, VY, VZ = \
    data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ
denmask = Den > 1e-19
X, Y, Z, T, Den, Rad, Vol, VX, VY, VZ = make_slices([X, Y, Z, T, Den, Rad, Vol, VX, VY, VZ], denmask)
Rad_den = np.multiply(Rad,Den) # now you have energy density
del Rad   
xyz = np.array([X, Y, Z]).T
R = np.sqrt(X**2 + Y**2 + Z**2)
# for i in range(len(observers_xyz)):
    # print(f'Snap: {snap}, Obs: {i}', flush=False)
    # sys.stdout.flush()

mu_x = observers_xyz[0] #[i][0]
mu_y = observers_xyz[1] #[i][1]
mu_z = observers_xyz[2] #[i][2]

# Box is for dynamic ray making
if mu_x < 0:
    rmax = box[0] / mu_x
    # print('x-', rmax)
else:
    rmax = box[3] / mu_x
    # print('x+', rmax)
if mu_y < 0:
    rmax = min(rmax, box[1] / mu_y)
    # print('y-', rmax)
else:
    rmax = min(rmax, box[4] / mu_y)
    # print('y+', rmax)
if mu_z < 0:
    rmax = min(rmax, box[2] / mu_z)
    # print('z-', rmax)
else:
    rmax = min(rmax, box[5] / mu_z)
    # print('z+', rmax)

r_height = np.logspace(-0.25, np.log10(rmax/2), N_ray)

#go vertically 
x = np.repeat(mu_x, len(r_height))
y = np.repeat(mu_y, len(r_height))
z = r_height
r = np.sqrt(x**2 + y**2 + z**2) #CHANGE r!!!
xyz2 = np.array([x, y, z]).T
# del x, y, z

tree = KDTree(xyz, leaf_size=50)
_, idx = tree.query(xyz2, k=1)
idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
d = Den[idx] * prel.den_converter
t = T[idx]
ray_x = X[idx]
ray_y = Y[idx]
ray_z = Z[idx]
ray_dim = (Vol[idx]**(1/3)) * prel.Rsol_cgs
ray_vz = VZ[idx]
plt.figure()
img = plt.scatter(xyz2[:,1]/Rt, xyz2[:,2]/Rt, c = xyz2[:,0]/Rt, alpha = 0.2, cmap = 'jet', marker = 's', label = 'What we want', vmin = 0.5, vmax = 1.5)
img = plt.scatter(ray_y/Rt, ray_z/Rt, c = ray_x/Rt, cmap = 'jet', label = 'From simulation', vmin = 0.5, vmax = 1.5)
cbar = plt.colorbar(img)
cbar.set_label(r'X [$R_{\rm t}$]')
plt.xlabel(r'Y [$R_{\rm t}$]')
plt.ylabel(r'Z [$R_{\rm t}$]')
# plt.xlim(-5/Rt,5/Rt)
# plt.ylim(-1/Rt,50/Rt)
plt.legend()
rad_den = Rad_den[idx]
volume = Vol[idx]

# Interpolate ----------------------------------------------------------
sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
sigma_rossland = np.array(sigma_rossland)[0]
underflow_mask = sigma_rossland != 0.0
idx = np.array(idx)
d, t, r, ray_x, ray_y, ray_z, sigma_rossland, rad_den, volume, idx = \
    make_slices([d, t, r, ray_x, ray_y, ray_z, sigma_rossland, rad_den, volume, idx], underflow_mask)
sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
del sigma_rossland
gc.collect()

# Optical Depth
r_height_fuT = np.flipud(r_height) #.T
kappa_rossland = np.flipud(sigma_rossland_eval) 
# compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_height_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
los[los <= 0] = 1e-20
ctau = prel.csol_cgs/los #c/tau [units code]
tdiff = (ray_dim * los / prel.c_cgs)/t_fall_cgs # [tfall]
# crete tdiff_cumulative where each element t is the sum tdiff[i] and all the previous ones
tdiff_cumulative = np.cumsum(tdiff)

fig, (ax1, ax2) = plt.subplots(1, 2 , figsize = (12,5))
ax1.plot(ray_z/Rt, tdiff_cumulative, c = 'k')
ax1.set_xlabel(r'$Z [R_{\rm t}]$')
ax1.set_ylabel(r'$t_{\rm diff} [t_{\rm fb}]$')
ax1.set_xlim(0, 50/Rt) # a bit further than the Rtrapp
# ax1.axvline(ray_z[np.argmin(np.abs(ctau[1:]/ray_vz[1:]-1))]/Rt)
img = ax2.scatter(ray_z/Rt, los, c = ctau/ray_vz, cmap = 'rainbow', vmin = 1e-5, vmax = 1)
cbar = plt.colorbar(img)
cbar.set_label(r'c$\tau^{-1}/V_z$')
ax2.set_xlabel(r'$Z [R_{\rm t}]$')
ax2.set_ylabel(r'$\tau$')
ax2.set_xlim(0, 50/Rt)
ax2.set_ylim(0.1,1)
plt.tight_layout()

    
# %%
