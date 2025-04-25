"""Compute trapping radius i.e. R: tau(R) = c/v(R)"""
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
import healpy as hp
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import nouveau_rich
import Utilities.prelude as prel
from Utilities.operators import make_tree, sort_list, to_spherical_components
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb

#%% matlab
eng = matlab.engine.start_matlab()
#%% Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' 
snap = 348
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
pre_saving = f'{abspath}/data/{folder}'

Rt = orb.tidal_radius(Rstar, mstar, Mbh)
a_min = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

# observers
Rt_obs = [Rt, 0, 0]
amin_obs = [-a_min, 0, 0]
apo_obs = [-apo, 0, 0]
observers_xyz = np.array([Rt_obs, amin_obs, apo_obs])
labels_obs = [r'R$_t$', r'a$_{min}$', r'R$_a$']
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
i = 1
mu_x = observers_xyz[i][0]
mu_y = observers_xyz[i][1]
mu_z = observers_xyz[i][2]

# Opacity
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

photo = np.loadtxt(f'{pre_saving}/photo/{check}_photo{snap}.txt')
xph, yph, zph = photo[0], photo[1], photo[2]
rph = np.sqrt(xph**2 + yph**2 + zph**2)
if alice:
    loadpath = f'{pre}/snap_{snap}'
else:
    loadpath = f'{pre}/{snap}'
data = make_tree(loadpath, snap, energy = True)
box = np.load(f'{loadpath}/box_{snap}.npy')
X, Y, Z, T, Den, Vol, VX, VY, VZ = \
    data.X, data.Y, data.Z, data.Temp, data.Den, data.Vol, data.VX, data.VY, data.VZ
denmask = Den > 1e-19
X, Y, Z, T, Den, Vol, VX, VY, VZ = \
    make_slices([X, Y, Z, T, Den, Vol, VX, VY, VZ], denmask)
xyz = np.array([X, Y, Z]).T
N_ray = 500

x_tr = np.zeros(len(observers_xyz))
y_tr = np.zeros(len(observers_xyz))
z_tr = np.zeros(len(observers_xyz))
Temp_tr = np.zeros(len(observers_xyz))
den_tr = np.zeros(len(observers_xyz))
vol_tr = np.zeros(len(observers_xyz))
Vx_tr = np.zeros(len(observers_xyz))
Vy_tr = np.zeros(len(observers_xyz))
Vz_tr = np.zeros(len(observers_xyz))

# Box is for dynamic ray making
# if mu_x < 0:
#     rmax = box[0] / mu_x
# else:
#     rmax = box[3] / mu_x
# if mu_y < 0:
#     rmax = min(rmax, box[1] / mu_y)
# else:
#     rmax = min(rmax, box[4] / mu_y)
# if mu_z < 0:
#     rmax = min(rmax, box[2] / mu_z)
# else:
#     rmax = min(rmax, box[5] / mu_z)
rmin = 0.01
rmax = 5 * apo
r_height = np.logspace(np.log10(rmin), np.log10(rmax), N_ray) # create the z array
# Go vertically 
x = np.repeat(mu_x, len(r_height))
y = np.repeat(mu_y, len(r_height))
z = r_height

xyz2 = np.array([x, y, z]).T
del x, y, z

tree = KDTree(xyz, leaf_size = 50)
_, idx = tree.query(xyz2, k=1)
idx = [ int(idx[i][0]) for i in range(len(idx))] 
idx = np.unique(idx)
ray_x = X[idx]
ray_y = Y[idx]
ray_z = Z[idx]
t = T[idx]
d = Den[idx] * prel.den_converter
ray_vol = Vol[idx]
ray_vx = VX[idx]
ray_vy = VY[idx]
ray_vz = VZ[idx]
ray_x, ray_y, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_z = \
    sort_list([ray_x, ray_y, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_z], ray_z)

# Interpolate ----------------------------------------------------------
sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
sigma_rossland = np.array(sigma_rossland)[0]
underflow_mask = sigma_rossland != 0.0
idx = np.array(idx)
ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, sigma_rossland = \
    make_slices([ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, sigma_rossland], underflow_mask)
sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
del sigma_rossland
gc.collect()

# Optical Depth
ray_fuT = np.flipud(ray_z) 
kappa_rossland = np.flipud(sigma_rossland_eval) #np.flipud(d)*0.34cm2/g
# compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_z. You integrate in the z direction
los_zero = los != 0
ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, los, sigma_rossland_eval = \
    make_slices([ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, los, sigma_rossland_eval], los_zero)
# you integrate for tau as BonnerotLu20, eq.16 for trapping radius
los_scatt = - np.flipud(sci.cumulative_trapezoid(np.flipud(d)*0.34, np.flipud(ray_z), initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_z. YOu integrate in the z direction
c_tau = prel.csol_cgs/los #c/tau [code units]

#%%
# plot to check if you're taking the right thing
plt.figure()
plt.plot(ray_z/apo, c_tau/np.abs(ray_vz), c = 'k', label = r'$c\tau^{-1}/v_z$')
plt.plot(ray_z/apo, los, label = r'$\tau_R$', c = 'r')
plt.plot(ray_z/apo, los_scatt, '--', label = r'$\tau_s$', c = 'r')
# plt.plot(ray_r/apo, np.abs(v_rad)*prel.Rsol_cgs/prel.tsol_cgs, label = r'$|v_r|$ [cm/s]', c = 'b')
plt.axhline(1, c = 'k', linestyle = 'dotted')
plt.xlabel(r'$Z [R_{\rm a}]$')
plt.loglog()
# plt.xlim(1e-1, 6)
# plt.ylabel(r'$c\tau^{-1}/|v_r|$')
plt.title(f'Snap {snap}, observer at x={labels_obs[i]}')
plt.legend()
        
#%% plot to check if you're taking the right thing
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,10))
cutplot = np.logical_and(Den > 1e-19, np.abs(X-mu_x)<Vol**(1/3))
img = ax1.scatter(Y[cutplot], Z[cutplot]/apo, c = X[cutplot]/mu_x,  cmap = 'jet', alpha = 0.7, vmin = 0.5, vmax = 1.5)
img = ax1.scatter(xyz2[:,1], xyz2[:,2]/apo, c = xyz2[:,0]/Rt, cmap = 'jet', marker = 's', edgecolors= 'k', label = 'What we want', vmin = 0.5, vmax = 1.5)
img = ax1.scatter(ray_y, ray_z/apo, c = ray_x/mu_x, cmap = 'jet', edgecolors= 'k', label = 'From simulation', vmin = 0.5, vmax = 1.5)
cbar = plt.colorbar(img, orientation = 'horizontal')
cbar.set_label(r'X [$x_{\rm chosen}$]')
ax1.legend(fontsize = 18)
img = ax2.scatter(ray_y, ray_z/apo, c = los, cmap = 'jet', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e3))
ax2.plot(ray_y, ray_z/apo, c = 'k', alpha = 0.5)
cbar = plt.colorbar(img, orientation = 'horizontal')
cbar.set_label(r'$\tau$')
ax1.set_ylabel(r'Z [$R_{\rm a}$]')
for ax in [ax1, ax2]:
    ax.set_xlabel(r'Y [$R_{\odot}$]')
    ax.set_xlim(-50,50)
    ax.set_ylim(-1/apo, 3.5)
ax1.set_title('Points wanted and selected', fontsize = 18)
ax2.set_title('Optical depth for each cell', fontsize = 18)
plt.suptitle(f'Observer at ({int(mu_x)}, {int(mu_y)}, {int(mu_z)}), snap {snap}', fontsize = 20)
plt.tight_layout()
#%%
Rtr_idx_all = np.where(c_tau/np.abs(ray_vz)<1)[0]
if len(Rtr_idx_all) == 0:
    print(f'No Rtr found anywhere for obs {i}', flush=False)
R_tr_all = ray_z[Rtr_idx_all]
# Rtr < Rph
Rtr_idx_all_inside = np.where(np.abs(R_tr_all)<np.mean(rph))[0]
if len(Rtr_idx_all_inside) == 0:
    print(f'No Rtr inside Rph for obs {i}', flush=False)
Rtr_idx_all = Rtr_idx_all[Rtr_idx_all_inside]
# take the one most outside
Rtr_idx = Rtr_idx_all[-1]

t_dyn = ray_z/np.abs(ray_vz) * prel.tsol_cgs # [s]

# tdiff = \tau*H/c 
#### FOR CUMULATIVE SCATTER OR ROSS
los_fuT = np.flipud(sigma_rossland_eval*ray_z) * prel.Rsol_cgs 
los_scatt_fuT = np.flipud(d*0.34*ray_z) * prel.Rsol_cgs
tdiff_cumulative = - np.flipud(sci.cumulative_trapezoid(los_fuT, np.flipud(ray_z), initial = 0))*prel.Rsol_cgs/ prel.c_cgs # this is the conversion for ray_z. You integrate in the z direction
#### FOR SINGLE
# los_fuT = sigma_rossland_eval * ray_z * prel.Rsol_cgs 
# tdiff = np.zeros(len(ray_z))
# for i in range(len(ray_z)):
#     tdiff[i] = sci.trapezoid(los_fuT[i:], ray_z[i:])* prel.Rsol_cgs/ prel.c_cgs

#%%
plt.figure(figsize = (10,5))
img = plt.scatter(ray_z/apo, sigma_rossland_eval/d, c = t, cmap = 'jet', norm = colors.LogNorm(vmin = 3e3, vmax = 5e5))
cbar = plt.colorbar(img)
cbar.set_label(r'$\tau$')
plt.xlabel(r'$R [R_{\rm a}]$')
plt.ylabel(r'$\kappa$ [cm$^2$/g]')
plt.xlim(0,7)
plt.axhline(0.34, c = 'k', linestyle = '--', label = r'$\kappa_{\rm Th}$')
# plt.ylim(1, 1e4)
plt.yscale('log')

#%%
fig, (ax1, ax2) = plt.subplots(1, 2 , figsize = (12,5))
ax1.plot(ray_z/apo, tdiff_cumulative/tfallback_cgs, c = 'k')
ax1.set_ylabel(r'$t_{\rm diff} [t_{\rm fb}]$')
ax1.axhline(t_dyn[Rtr_idx]/tfallback_cgs, c = 'k', linestyle = '--', label =  r'$t_{\rm dyn}=R_z/v_z$')
# ax1.set_ylim(0, 2) # a bit further than the Rtrapp
img = ax2.scatter(ray_z/apo, los, c = c_tau/np.abs(ray_vz), cmap = 'rainbow', vmin = 0, vmax = 2)
cbar = plt.colorbar(img)
cbar.set_label(r'c$\tau^{-1}/V_z$')
ax2.set_ylabel(r'$\tau$')
ax2.set_yscale('log')
ax2.axvline(np.mean(rph)/apo, c = 'k', linestyle = 'dotted', label =  r'$<R_{\rm ph}>$')
# ax2.set_ylim(0.1, 1e2)
for ax in [ax1, ax2]:
    ax.set_xlabel(r'$Z [R_{\rm a}]$')
    ax.set_xlim(-0.1, 7)
    ax.axvline(ray_z[Rtr_idx]/apo, c = 'b', linestyle = '--', label =  r'$R_{\rm tr} (c/\tau=V_z)$')
    ax.legend(fontsize = 14)
plt.suptitle('Form outside in', fontsize = 20)
plt.tight_layout()

#%%
eng.exit()