""" Finddensity profile"""
#%%
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm

import sys
sys.path.append(abspath)

import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.integrate as sci
import matlab.engine
from sklearn.neighbors import KDTree
import healpy as hp
import Utilities.prelude as prel
import scipy.integrate as spi
import scipy.optimize as spo
from src.Opacity.linextrapolator import nouveau_rich
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import to_spherical_components, sort_list

##
#
##
def CouBegel(r, theta, q, gamma=4/3):
    # I'm missing the normalization
    alpha = (1-q*(gamma-1))/(gamma-1)
    rho = r**(-q) * np.sin(theta)**(2*alpha)
    return rho

def Metzger(r, q, Mbh, mstar, Rstar, k = 0.9, Me = 1, G = prel.G):
    # I'm missing the correct noramlization
    Rv = 2 * Rstar/(5*k) * (Mbh/mstar)**(2/3) * Me/mstar
    norm = Me*(3-q)/(4*np.pi*Rv**3 * (7-2*q))
    if norm == 0.0:
        print('norm is 0')
    if r < Rv:
        rho = (r/Rv)**(-q)
    else:
        rho = np.exp(-(r-Rv)/Rv)
    return (norm * rho)

theta_test = np.linspace(0, np.pi, 50) #latitude
d_test = CouBegel(1, theta_test, 0.5)
d_test2 = CouBegel(1, theta_test, 1)
d_test3 = CouBegel(1, theta_test, 1.5)
# plt.figure()
# plt.plot(theta_test, d_test, label = r'R=1, q = 0.5')
# plt.plot(theta_test, d_test2, label = r'R=1, q = 1')
# plt.plot(theta_test, d_test3, label = r'R=1, q = 1.5')
# plt.ylabel(r'$\rho$')
# plt.xlabel(r'$\theta$')
# plt.legend()
# plt.title(r'$\rho \propto R^{-q} \sin^{2\alpha}(\theta)$, $\alpha = \frac{1-q(\gamma-1)}{\gamma-1}, \gamma = 4/3$')
#%%
##
# MAIN
##
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # '' or 'HiRes'
which_obs = 'healpix' # 'healpix' or 'elliptical'
# Rg = Mbh * prel.G / prel.csol_cgs**2
# print(Rg)
#%%
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 237
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
e_mb = orb.eccentricity(Rstar, mstar, Mbh, beta)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt * beta
x_test = np.arange(1, 3e2)
y_test2 = 1e-15 * x_test**(-2)
y_test3 = 2e-9 * x_test**(-3)
y_test4 = 1e-10 * x_test**(-4)
y_test7 = 1e-10 * x_test**(-7)
r_const = np.array([Rt, 10*Rt, apo])
r_const_label = [r'$R_t$', r'$10R_t$', r'$R_a$']
r_const_colors = ['dodgerblue', 'darkviolet', 'forestgreen']

# Opacity Input (they are ln)
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

#%% MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
box = np.zeros(6)
# Load data -----------------------------------------------------------------
if alice:
    X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
    Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
    Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
    VX = np.load(f'{pre}/snap_{snap}/Vx_{snap}.npy')
    VY = np.load(f'{pre}/snap_{snap}/Vy_{snap}.npy')
    VZ = np.load(f'{pre}/snap_{snap}/Vz_{snap}.npy')
    T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
    Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
    Rad = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy')
    Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
    box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
else:
    X = np.load(f'{pre}/{snap}/CMx_{snap}.npy')
    Y = np.load(f'{pre}/{snap}/CMy_{snap}.npy')
    Z = np.load(f'{pre}/{snap}/CMz_{snap}.npy')
    VX = np.load(f'{pre}/{snap}/Vx_{snap}.npy')
    VY = np.load(f'{pre}/{snap}/Vy_{snap}.npy')
    VZ = np.load(f'{pre}/{snap}/Vz_{snap}.npy')
    T = np.load(f'{pre}/{snap}/T_{snap}.npy')
    Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
    Rad = np.load(f'{pre}/{snap}/Rad_{snap}.npy')
    Vol = np.load(f'{pre}/{snap}/Vol_{snap}.npy')
    box = np.load(f'{pre}/{snap}/box_{snap}.npy')

denmask = Den > 1e-19
X, Y, Z, T, Den, Rad, Vol = make_slices([X, Y, Z, T, Den, Rad, Vol], denmask)
Rad_den = np.multiply(Rad,Den) # now you have energy density
del Rad   
R = np.sqrt(X**2 + Y**2 + Z**2)    

#%% Observers -----------------------------------------------------------------
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
obs_indices = np.arange(len(observers_xyz[0]))
# select only the observers in the orbital plane (will give you a N bool array--> apply to columns)
mid = np.abs(observers_xyz[2]) == 0 # you can do that beacuse healpix gives you the observers also in the orbital plane (Z==0)
observers_xyz_mid, indices_mid = observers_xyz[:,mid], obs_indices[mid]
# observers_xyz = observers_xyz[:,np.arange(0,192,5)]
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
observers_xyz = np.transpose(observers_xyz) #shape: Nx3

r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
long_obs = np.arctan2(y_obs, x_obs)          # Azimuthal angle in radians
lat_obs = np.arccos(z_obs / r_obs)

#%%
xyz = np.array([X, Y, Z]).T
N_ray = 20_000
# d_r = []
# vx_r = []
# vy_r = []
# vz_r = []
print(f'Snap: {snap}')
# with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt','w') as file:
#     file.write(f'# radii, density profile, Vx, Vy, Vz.\n')
#     file.close()
indices_chosen_mid_slice = np.array([np.argmin(np.abs(observers_xyz_mid[0] + 1)),
                            np.argmin(np.abs(observers_xyz_mid[0] - 1)),
                            np.argmin(np.abs(observers_xyz_mid[1] + 1)), 
                            np.argmin(np.abs(observers_xyz_mid[1] - 1))])
indices_chosen_mid = indices_mid[indices_chosen_mid_slice]     
indices_chosen_z = np.array([np.argmin(np.abs(observers_xyz[:,2] + 1)),
                            np.argmin(np.abs(observers_xyz[:,2] - 1))])
indices_chosen = np.concatenate([indices_chosen_mid, indices_chosen_z])

# indices_chosen = np.array([np.argmin(np.abs(observers_xyz[:,0] + 1)),
#                             np.argmin(np.abs(observers_xyz[:,0] - 1)),
#                             np.argmin(np.abs(observers_xyz[:,1] + 1)), 
#                             np.argmin(np.abs(observers_xyz[:,1] - 1)),
#                             np.argmin(np.abs(observers_xyz[:,2] + 1)),
#                             np.argmin(np.abs(observers_xyz[:,2] - 1))])
label_obs = ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']

#%% Load
dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{snap}.txt')
xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
    dataph[0], dataph[1], dataph[2], dataph[3], dataph[4], dataph[5], dataph[6], dataph[7], dataph[8], dataph[9]
rph = np.sqrt(xph**2 + yph**2 + zph**2)
long_ph = np.arctan2(yph, xph)          # Azimuthal angle in radians
lat_ph = np.arccos(zph/ rph)            # Elevation angle in radians
v_rad_ph, v_theta_ph, v_phi_mid = to_spherical_components(Vxph, Vyph, Vzph, lat_ph, long_ph)
v_rad_ph_sorted, rph_sorted = sort_list([v_rad_ph, rph], rph)

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
for i in indices_chosen:
    mu_x = observers_xyz[i][0]
    mu_y = observers_xyz[i][1]
    mu_z = observers_xyz[i][2]

    # Box is for dynamic ray making
    # box gives -x, -y, -z, +x, +y, +z
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

    r = np.logspace(-0.25, np.log10(rmax), N_ray)

    x = r*mu_x
    y = r*mu_y
    z = r*mu_z
    r = r*np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
    xyz2 = np.array([x, y, z]).T
    del x, y, z
    # find the simulation cell corresponding to cells in the wanted ray
    tree = KDTree(xyz, leaf_size = 50) 
    _, idx = tree.query(xyz2, k=1)
    idx = [ int(idx[i][0]) for i in range(len(idx))]
    # Quantity corresponding to the ray
    d = Den[idx] * prel.den_converter
    t = T[idx]
    ray_x = X[idx]
    ray_y = Y[idx]
    ray_z = Z[idx]
    rad_den = Rad_den[idx]
    volume = Vol[idx]
    ray_vx = VX[idx]
    ray_vy = VY[idx]
    ray_vz = VZ[idx]
    
    sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
    sigma_rossland = np.array(sigma_rossland)[0]
    underflow_mask = sigma_rossland != 0.0
    idx = np.array(idx)
    d, t, r, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, idx = \
        make_slices([d, t, r, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, idx], underflow_mask)
        
    v_rad, v_theta, v_phi = to_spherical_components(ray_vx, ray_vy, ray_vz, lat_obs[i], long_obs[i])
    v = np.sqrt(ray_vx**2 + ray_vy**2 + ray_vz**2)
    # find density at a given R for each observer. Since observers are noramlized, you can directly use r to query.
    # r_fortree = r.reshape(-1, 1)
    # tree_r = KDTree(r_fortree, leaf_size=50)
    # _, idx_r = tree_r.query(r_const.reshape(-1,1), k=1)
    # idx_r = [ int(idx_r[i][0]) for i in range(len(idx_r))] # no -1 because we start from 0
    # d_r.append(d * prel.den_converter)
    # vx_r.append(ray_vx)
    # vy_r.append(ray_vy)
    # vz_r.append(ray_vz)

    # with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt','a') as file:
    #     file.write(f'# Observer latitude: {lat_obs[i]}, longitude: {long_obs[i]}\n')
    #     file.write(f' '.join(map(str, r)) + '\n')
    #     file.write(f' '.join(map(str, d)) + '\n')
    #     file.write(f' '.join(map(str, ray_vx)) + '\n')
    #     file.write(f' '.join(map(str, ray_vy)) + '\n')
    #     file.write(f' '.join(map(str, ray_vz)) + '\n')
    #     file.close()

    ax1.plot(r/Rt, d, label = f'Observer {label_obs[np.where(indices_chosen == i)[0][0]]} ({i})')
    # ax1.scatter(rph[i]/Rt, denph[i]*prel.den_converter, c = 'k', s = 20)
    ax2.plot(r/Rt, np.abs(v_rad)*prel.Rsol_cgs*1e-5/prel.tsol_cgs, label = f'Observer {label_obs[np.where(indices_chosen == i)[0][0]]} ({i})')
    # ax1.axvline(x = rph[i]/Rt, c = 'gray', ls = '--')

ax1.plot(x_test, y_test2, c = 'gray', ls = 'dashed', label = r'$\rho \propto R^{-2}$')
ax1.plot(x_test, y_test3, c = 'gray', ls = 'dotted', label = r'$\rho \propto R^{-3}$')
ax1.plot(x_test, y_test4, c = 'gray', ls = '-.', label = r'$\rho \propto R^{-4}$')
ax1.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
ax1.set_ylim(2e-19, 2e-7)
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax2.set_ylabel(r'$|v_r|$ [km/s]')
ax2.legend(loc='lower right', fontsize = 12)
for ax in [ax1, ax2]:
    ax.grid()
    ax.set_xlabel(r'R [R$_t]$')
    ax.set_xlim(0.9, 3e2)
    ax.loglog()
plt.tight_layout()
#%%
time = np.loadtxt(f'{abspath}/data/{folder}/slices/z/z0_time.txt')
snaps = time[0]
snaps = [int(snap) for snap in snaps]
tfb = time[1]
escape_vel_kms = np.sqrt(2*prel.G*Mbh/Rt)*prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_mean = np.zeros(len(snaps))
for i, sn in enumerate(snaps):
    dataph_i = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{sn}.txt')
    xph_i, yph_i, zph_i, volph_i, denph_i, Tempph_i, Rad_denph_i, Vxph_i, Vyph_i, Vzph_i = \
        dataph_i[0], dataph_i[1], dataph_i[2], dataph_i[3], dataph_i[4], dataph_i[5], dataph_i[6], dataph_i[7], dataph_i[8], dataph_i[9]
    rph_i = np.sqrt(xph_i**2 + yph_i**2 + zph_i**2)
    long_ph_i = np.arctan2(yph_i, xph_i)          # Azimuthal angle in radians
    lat_ph_i = np.arccos(zph_i/ rph_i)            # Elevation angle in radians
    v_rad_ph_i, v_theta_ph_i, v_ph_i_mid = to_spherical_components(Vxph_i, Vyph_i, Vzph_i, lat_ph_i, long_ph_i)
    v_mean[i] = np.median(np.abs(v_rad_ph_i))
plt.figure()
plt.plot(tfb, v_mean*prel.Rsol_cgs*1e-5/prel.tsol_cgs, c = 'k')
plt.axhline(y = escape_vel_kms, c = 'gray', ls = '--')
plt.xlabel(r't [t$_{fb}]$')
plt.ylabel(r'median $|v_{\rm r_{\rm ph}}|$ [km/s]')
plt.yscale('log')
plt.grid()

# d_r_trans = np.transpose(d_r)
# img, ax1 = plt.subplots(1,1,figsize = (7, 7)) # this is to check all observers from Healpix on the orbital plane
# for i, radius in enumerate(r_const):
#     Cougel_den = np.zeros(len(lat_obs))
#     for j, lat_single in enumerate(lat_obs):
#         Cougel_den[j] = CouBegel(radius*prel.Rsol_cgs, lat_single, .7, 4/3)
#     # ax1.scatter(lat_obs, d_r_trans[i], c= r_const_colors[i], label = f'R = {r_const_label[i]}')
#     ax1.scatter(lat_obs, d[i], c= r_const_colors[i], label = f'R = {r_const_label[i]}')
#     ax1.plot(lat_obs, Cougel_den, c = r_const_colors[i])
# ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
# ax1.set_xlabel(r'$\theta$ [rad]')
# ax1.set_yscale('log')
# ax1.legend(loc='upper right')
# ax1.set_ylim(2e-15, 2e-5)#2e-9)

# #%% Check density profile from Rin == Rp till Rout= Rph (but should be the colorsphere at tau=1)
# data_loaded = np.loadtxt(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}.txt')
# r_all, d_all, vx_all, vy_all, vz_all = data_loaded[0::5], data_loaded[1::5], data_loaded[2::5], data_loaded[3::5], data_loaded[4::5]

# img, (ax1, ax2) = plt.subplots(1,2,figsize = (12, 5)) # this is to check all observers from Healpix on the orbital plane
# for j, r in enumerate(r_all):
#     d = d_all[j]
#     den_Metzger3 = np.zeros(len(r))
#     den_Metzger7 = np.zeros(len(r))
#     for k in range(len(r)):
#         den_Metzger3[k] = Metzger(r[k], 2.8, Mbh, mstar, Rstar, Me = 0.8*mstar) 
#         den_Metzger7[k] = Metzger(r[k], 7, Mbh, mstar, Rstar, Me = 0.8*mstar)
#     if np.abs(long_obs[j]) < np.pi/2:
#         ax1.plot(r/Rt, d, c = cm.jet(j / len(r_all)))#, label = f'{j}')
#     else:
#         ax2.plot(r/Rt, d, c = cm.jet(j / len(r_all)))#, label = f'{j}')
# ax1.set_title(r'Obs $\theta\leq\pi/2$')
# ax2.set_title(r'Obs $\theta\geq\pi/2$')
# ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
# ax1.plot(x_test, y_test3, c = 'gray', ls = 'dotted', label = r'$\rho \propto R^{-3}$')
# ax1.plot(x_test, y_test4, c = 'gray', ls = '-.', label = r'$\rho \propto R^{-4}$')
# ax1.plot(r/Rt, den_Metzger3*prel.den_converter, c = 'gray', ls = '--', label = r'Metzger $\xi = 2.8, M_e = 0.8m_\star$')
# ax2.plot(r/Rt, den_Metzger7*prel.den_converter, c = 'gray', ls = '--', label = r'Metzger $\xi = 7, M_e = 0.8m_\star$')
# ax2.plot(x_test, y_test7, c = 'gray', ls = 'dotted', label = r'$\rho \propto R^{-7}$')

# for ax in [ax1, ax2]:
#     ax.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
#     ax.set_xlabel(r'R [R$_a]$')
#     ax.loglog()
#     ax.grid()
#     ax.legend(loc='upper right', fontsize = 12)
#     ax.set_xlim(1, 3e2)
# ax1.set_ylim(2e-19, 2e-7)
# ax2.set_ylim(2e-19, 2e-5)
# img.tight_layout()
# img.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/OrbPl/{snap}_denprof.png', bbox_inches='tight')

#%%
vr_all = []
vtheta_all = []
vp_all = []
v_all = []

for i in range(len(observers_xyz)):
    vx_obs = vx_all[i]
    vy_obs = vy_all[i]
    vz_obs = vz_all[i]
    v_r = np.zeros(len(vx_obs))
    v_t = np.zeros(len(vx_obs))
    v_p = np.zeros(len(vx_obs))
    v = np.zeros(len(vx_obs))
    for j in range(len(vx_obs)):
        v_r[j] = radial_vel(vx_obs[j], vy_obs[j], vz_obs[j], lat_obs[i], long_obs[i])
        v_t[j] = orb.v_theta(vx_obs[j], vy_obs[j], vz_obs[j], lat_obs[i], long_obs[i])
        v_p[j] = orb.v_phi(vx_obs[j], vy_obs[j], long_obs[i])
        v[j] = np.sqrt(vx_obs[j]**2 + vy_obs[j]**2 + vz_obs[j]**2)
    vr_all.append(v_r)
    vtheta_all.append(v_t)
    vp_all.append(v_p)
    v_all.append(v)
#%%
plt.figure()
for j in range(0, len(r_all), 10):
    v = vp_all[j]
    r = r_all[j]
    plt.scatter(r/apo, v, c = cm.jet(j / len(r_all)))
plt.loglog()
#%%

#%%
eng.exit()

#%%
