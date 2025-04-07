""" Find density profile"""
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
import k3match
import scipy.integrate as sci
from scipy.stats import gmean
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
which_obs = 'arch'
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
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs

#%% MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()
#%% Observers -----------------------------------------------------------------
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
obs_indices = np.arange(len(observers_xyz[0]))
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
long_obs = np.arctan2(y_obs, x_obs)          # Azimuthal angle in radians
lat_obs = np.arccos(z_obs / r_obs)
# Convert to latitude and longitude
longitude_moll = long_obs              
latitude_moll = np.pi / 2 - lat_obs 
mid = np.concatenate(np.where(latitude_moll==0))
# select only the observers in the orbital plane (will give you a N bool array--> apply to columns)
observers_xyz_mid, indices_mid = observers_xyz[:,mid], obs_indices[mid]

if which_obs == 'arch':
    wanted_obs = [(1,0,0), 
                (1/np.sqrt(2), 0, 1/np.sqrt(2)),  
                (0,0,1),
                (-1/np.sqrt(2), 0 , 1/np.sqrt(2)),
                (-1,0,0)]
    # dot_prod = np.dot(wanted_obs, observers_xyz)
    # indices_chosen = np.argmax(dot_prod, axis=1)
    indices_chosen = []
    for i, obs_sel in enumerate(wanted_obs):   
        _, indices_dist, dist = k3match.cartesian(obs_sel[0], obs_sel[1], obs_sel[2], x_obs, y_obs, z_obs, 1)
        indices_dist, dist = sort_list([indices_dist, dist], dist)
        indices_chosen.append(indices_dist[0:1])
    indices_chosen = np.concatenate(indices_chosen, dtype = int)
    label_obs = ['x+', '45', 'z+', '135', 'x-']
    colors_obs = ['k', 'green', 'orange', 'b', 'r']
else:
    indices_chosen_mid_slice = np.array([np.argmin(np.abs(observers_xyz_mid[0] + 1)),
                                np.argmin(np.abs(observers_xyz_mid[0] - 1)),
                                np.argmin(np.abs(observers_xyz_mid[1] + 1)), 
                                np.argmin(np.abs(observers_xyz_mid[1] - 1))])
    indices_chosen_mid = indices_mid[indices_chosen_mid_slice]     
    indices_chosen_z = np.array([np.argmin(np.abs(z_obs + 1)),
                                np.argmin(np.abs(z_obs - 1))])
    indices_chosen = np.concatenate([indices_chosen_mid, indices_chosen_z])
    label_obs = ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']
    colors_obs = plt.cm.tab10(np.linspace(0, 1, len(indices_chosen)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(x_obs, y_obs, c = 'gray')
ax2.scatter(x_obs, z_obs, c = 'gray')
# scatter plot of x_obs[indices_chosen] with a different color for each different point
for j, idx_list in enumerate(indices_chosen):
    ax1.scatter(x_obs[idx_list], y_obs[idx_list], edgecolors = 'k', s = 50, c = colors_obs[j])
    ax2.scatter(x_obs[idx_list], z_obs[idx_list], edgecolors = 'k', s = 50, c = colors_obs[j])
ax1.set_ylabel(r'$Y$')
ax2.set_ylabel(r'$Z$')
for ax in [ax1, ax2]:
    ax.set_xlabel(r'$X$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
plt.suptitle('Selected observers', fontsize=15)
plt.tight_layout()

observers_xyz = np.transpose(observers_xyz) #shape: Nx3
#%% Opacity Input (they are ln)
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
box = np.zeros(6)

#%% Load data -----------------------------------------------------------------
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
    Mass = np.load(f'{pre}/{snap}/Mass_{snap}.npy')
    Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
    Rad = np.load(f'{pre}/{snap}/Rad_{snap}.npy')
    Vol = np.load(f'{pre}/{snap}/Vol_{snap}.npy')
    box = np.load(f'{pre}/{snap}/box_{snap}.npy')

denmask = Den > 1e-19
X, Y, Z, T, Den, Rad, Vol = make_slices([X, Y, Z, T, Den, Rad, Vol], denmask)
Rad_den = np.multiply(Rad,Den) # now you have energy density
del Rad   
R = np.sqrt(X**2 + Y**2 + Z**2)    

xyz = np.array([X, Y, Z]).T
N_ray = 5_000
with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}{which_obs}avgTree.txt','w') as file:
        file.close()
for i, idx_list in enumerate(indices_chosen):
    if i!= 0:
        continue
    if i==0:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,10))

    mu_x = observers_xyz[idx_list][0]
    mu_y = observers_xyz[idx_list][1]
    mu_z = observers_xyz[idx_list][2]

    # Box is for dynamic ray making
    # box gives -x, -y, -z, +x, +y, +z
    if mu_x < 0:
        rmax = box[0] / mu_x
    else:
        rmax = box[3] / mu_x
    if mu_y < 0:
        rmax = min(rmax, box[1] / mu_y)
    else:
        rmax = min(rmax, box[4] / mu_y)
    if mu_z < 0:
        rmax = min(rmax, box[2] / mu_z)
    else:
        rmax = min(rmax, box[5] / mu_z)

    r = np.logspace(-0.25, np.log10(rmax), N_ray)

    x = r*mu_x
    y = r*mu_y
    z = r*mu_z
    r = r*np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
    xyz2 = np.array([x, y, z]).T
    del x, y, z
    # find the simulation cell corresponding to cells in the wanted ray
    tree = KDTree(xyz, leaf_size = 50) 
    _, idx = tree.query(xyz2, k = 4) # so in the plot along one line of sight you'll have more points
    # idx = [ int(idx[i][j]) for i in range(len(idx))]
    # Quantity corresponding to the ray
    d = Den[idx] * prel.den_converter
    t = T[idx]
    ray_x = X[idx]
    ray_y = Y[idx]
    ray_z = Z[idx]
    ray_Mass = Mass[idx]
    rad_den = Rad_den[idx]
    dim_cell = Vol[idx]**(1/3)
    ray_vx = VX[idx]
    ray_vy = VY[idx]
    ray_vz = VZ[idx]
    # average
    d = np.mean(d, axis=1)
    t = np.mean(t, axis=1)
    ray_x = np.mean(ray_x, axis=1)
    ray_y = np.mean(ray_y, axis=1)
    ray_z = np.mean(ray_z, axis=1)
    ray_Mass = np.mean(ray_Mass, axis=1)
    rad_den = np.mean(rad_den, axis=1)
    dim_cell = np.mean(dim_cell, axis=1)
    ray_vx = np.mean(ray_vx, axis=1)
    ray_vy = np.mean(ray_vy, axis=1)
    ray_vz = np.mean(ray_vz, axis=1)
    
    sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
    sigma_rossland = np.array(sigma_rossland)[0]
    underflow_mask = sigma_rossland != 0.0
    idx = np.array(idx)
    d, t, r, ray_x, ray_y, ray_z, dim_cell, ray_vx, ray_vy, ray_vz, idx = \
        make_slices([d, t, r, ray_x, ray_y, ray_z, dim_cell, ray_vx, ray_vy, ray_vz, idx], underflow_mask)
    v_rad, v_theta, v_phi = to_spherical_components(ray_vx, ray_vy, ray_vz, lat_obs[i], long_obs[i])
    v_tot = np.sqrt(ray_vx**2 + ray_vy**2 + ray_vz**2)
    

    # fig1, axsingle = plt.subplots(1, 1, figsize=(8, 7))
    # img = axsingle.scatter(ray_x[np.abs(v_rad*conversion_sol_kms)<1]/apo, v_rad[np.abs(v_rad*conversion_sol_kms)<1]*conversion_sol_kms, c = ray_y[np.abs(v_rad*conversion_sol_kms)<1]/apo)
    # cbar = plt.colorbar(img)
    # cbar.set_label(r'Y [$R_{\rm a}$]')
    # axsingle.set_xlabel(r'R [$R_{\rm a}$]')
    # axsingle.set_ylabel(r'$|V_r|$ [km/s]')
    # axsingle.set_xlim(0.8, 4)
    # fig1.suptitle(f'Observer {label_obs[j]}, number {i}, snap {snap}')
    # fig1.tight_layout()

    #
    if i == 0:
        ax1.plot(xyz2[:,0]/apo, xyz2[:,1]/apo, c = colors_obs[j], alpha = 0.5)
        img1 = ax1.scatter(ray_x/apo, ray_y/apo, c = np.abs(ray_z)/apo, cmap = 'jet', label = 'From simulation', norm = colors.LogNorm(vmin = 8e-3, vmax = 0.7))
        ax1.set_ylabel(r'Y [$R_{\rm a}$]')
        # ax1.legend(fontsize = 18)
        ax2.plot(ray_x/apo, ray_y/apo, c = 'k', alpha = 0.5)
        img2 = ax2.scatter(ray_x/apo, ray_y/apo, c = np.abs(v_rad)*conversion_sol_kms, cmap = 'jet', norm = colors.LogNorm(vmin = 1e-1, vmax = 2e4))
        cbar = plt.colorbar(img1, orientation = 'horizontal')
        cbar.set_label(r'Z [R$_{\rm a}]$')
        cbar = plt.colorbar(img2, orientation = 'horizontal')
        cbar.set_label(r'$|V_r|$ [km/s]')
        for ax in [ax1, ax2]:
            ax.set_xlabel(r'X [$R_{\rm a}$]')
            ax.set_xlim(5e-2,10)
            ax.set_xscale('log')
            ax.set_ylim(-.5, .5)
        ax1.set_title('Points wanted and selected', fontsize = 18)
        ax2.set_title('Velocity', fontsize = 18)
        plt.suptitle(f'Observer {label_obs[i]}, snap {snap}', fontsize = 20)
        plt.tight_layout()

    with open(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}{which_obs}avgTree.txt','a') as file:
        file.write(f'# Observer latitude: {lat_obs[i]}, longitude: {long_obs[i]}\n')
        file.write(f' '.join(map(str, r)) + '\n')
        file.write(f' '.join(map(str, d)) + '\n')
        file.write(f' '.join(map(str, v_rad)) + '\n')
        file.write(f' '.join(map(str, v_tot)) + '\n')
        file.close()

#%%
x_test = np.arange(1e-3, 1e2)
y_test2 = 1e-17 * (x_test/apo)**(-2)
y_test3 = 2e-21 * (x_test/apo)**(-3)
y_test4 = 1e-24 * (x_test/apo)**(-4)
data_prof = np.loadtxt(f'{abspath}/data/{folder}/EddingtonEnvelope/den_prof{snap}{which_obs}avgTree.txt')
r_arr, d_arr, ray_vr_arr, ray_v_arr = data_prof[0::4], data_prof[1::4], data_prof[2::4], data_prof[3::4]
fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
fig1, ax2 = plt.subplots(1, 1, figsize=(8, 7))
fig2, ax3 = plt.subplots(1, 1, figsize=(8, 7))
for i in range(0,1):#len(r_arr)):
    r = r_arr[i]
    d = d_arr[i]
    ray_vr = ray_vr_arr[i]
    ray_v = ray_v_arr[i]
    ax1.plot(r/apo, d, label = f'Observer {label_obs[i]} ({indices_chosen[i]})', color = colors_obs[i])
    ax2.plot(r/apo, np.abs(ray_vr)*prel.Rsol_cgs*1e-5/prel.tsol_cgs, label = f'Observer {label_obs[i]} ({indices_chosen[i]})', color = colors_obs[i])
    ax3.plot(r/apo, r**2*d*np.abs(ray_vr)*prel.Rsol_cgs**3/prel.tsol_cgs, label = f'Observer {label_obs[i]} ({indices_chosen[i]})', color = colors_obs[i])
    # ax1.axvline(x = rph[i]/apo, c = 'gray', ls = '--')
ax1.plot(x_test, y_test2, c = 'gray', ls = 'dashed', label = r'$\rho \propto R^{-2}$')
ax1.plot(x_test, y_test3, c = 'gray', ls = 'dotted', label = r'$\rho \propto R^{-3}$')
ax1.plot(x_test, y_test4, c = 'gray', ls = '-.', label = r'$\rho \propto R^{-4}$')
ax1.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
ax1.set_ylim(2e-19, 2e-7)
# ax2.set_ylim(500, 2e4)
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax2.set_ylabel(r'$|v_r|$ [km/s]')
ax3.set_ylabel(r'$|v_r| \rho R^2$ [g/s]')
xmin = Rt/apo
xmax = 400*Rt/apo
for ax in [ax1, ax2, ax3]:
    #put the legend outside
    boxleg = ax.get_position()
    ax.set_position([boxleg.x0, boxleg.y0, boxleg.width * 0.8, boxleg.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
    # ax.legend(loc='lower right', fontsize = 12)
    ax.grid()
    ax.set_xlabel(r'R [R$_a]$')
    ax.set_xlim(xmin, xmax)
    ax.loglog()
    # ax4 = ax.twiny()
    # ax4.set_xlim(xmin*apo/Rt, xmax*apo/Rt)
    # ax4.set_xscale('log')
    # ax4.set_xlabel(r'R [R$_t]$')
    # ax.set_title(f'{snap}')

plt.tight_layout()
plt.show()

#%%
eng.exit()

#%%
