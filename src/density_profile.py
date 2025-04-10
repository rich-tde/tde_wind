""" Find density and (radial velocity) profiles"""
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
import matlab.engine
import k3match
from sklearn.neighbors import KDTree
import healpy as hp
import Utilities.prelude as prel
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
which_obs = 'all_rotate' # 'arch', 'all_cartesian', 'all_rotate'
# Rg = Mbh * prel.G / prel.csol_cgs**2
# print(Rg)
#%%
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 348
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
e_mb = orb.eccentricity(Rstar, mstar, Mbh, beta)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt * beta
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs

#%% MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()
#%% Observers
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
obs_indices = np.arange(len(observers_xyz[0]))
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
long_obs = np.arctan2(y_obs, x_obs)          # Azimuthal angle in radians
lat_obs = np.arccos(z_obs / r_obs)

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
        indices_chosen.append(indices_dist[0:4])
    # indices_chosen = np.array(indices_chosen, dtype = int)
    label_obs = ['x+', '45', 'z+', '135', 'x-']
    colors_obs = ['k', 'green', 'orange', 'b', 'r']
if which_obs == 'all_cartesian':
    # Cartesian view    
    indices1 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs >= 0, y_obs >= 0))]
    indices2 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs < 0, y_obs >= 0))]
    indices3 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs < 0, y_obs < 0))]
    indices4 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(x_obs >= 0, y_obs < 0))]
    indices5 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs >= 0, y_obs >= 0))]
    indices6 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs < 0, y_obs >= 0))]
    indices7 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs < 0, y_obs < 0))]
    indices8 = obs_indices[np.logical_and(z_obs<0, np.logical_and(x_obs >= 0, y_obs < 0))]
    indices_chosen = [indices1, indices2, indices3, indices4, indices5, indices6, indices7, indices8]
    label_obs = ['+x+y+z', '-x+y+z', '-x-y+z', '+x-y+z',
                 '+x+y-z', '-x+y-z', '-x-y-z', '+x-y-z',]
    colors_obs = plt.cm.rainbow(np.linspace(0, 1, len(indices_chosen)))
if which_obs == 'all_rotate':
    indices1 = obs_indices[np.logical_and(x_obs >= 0, np.abs(y_obs) < x_obs)]
    indices2 = obs_indices[np.logical_and(y_obs >= 0, y_obs > np.abs(x_obs))]
    indices3 = obs_indices[np.logical_and(x_obs < 0, np.abs(y_obs) < np.abs(x_obs))]
    indices4 = obs_indices[np.logical_and(y_obs < 0, np.abs(y_obs) > np.abs(x_obs))]
    
    indices5 = obs_indices[np.logical_and(z_obs>=0, np.logical_and(z_obs > np.abs(y_obs), z_obs > np.abs(x_obs)))]
    indices6 = obs_indices[np.logical_and(z_obs<0, np.logical_and(np.abs(z_obs) > np.abs(y_obs), np.abs(z_obs) > np.abs(x_obs)))]

    indices_chosen = [indices1, indices2, indices3, indices4, indices5, indices6]#, indices7, indices8]
    colors_obs = plt.cm.rainbow(np.linspace(0, 1, len(indices_chosen)))
    label_obs = ['+x', '+y', '-x', '-y', '+z', '-z']
observers_xyz = np.transpose(observers_xyz) #shape: Nx3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for ax in [ax1, ax2]:
    # ax1.scatter(x_obs, y_obs, c = 'gray')
    # ax2.scatter(x_obs, z_obs, c = 'gray')
    # ax1.scatter(x_obs[z_obs==0], y_obs[z_obs==0], c = 'gray', edgecolors = 'k')
    # ax2.scatter(x_obs[z_obs==0], z_obs[z_obs==0], c = 'gray', edgecolors = 'k')
    ax.set_xlabel(r'$X$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
for j, idx_list in enumerate(indices_chosen):
    ax1.scatter(x_obs[idx_list], y_obs[idx_list], s = 50, edgecolors = 'k', c = colors_obs[j])
    ax2.scatter(x_obs[idx_list], z_obs[idx_list], s = 50, edgecolors = 'k', c = colors_obs[j])
ax1.set_ylabel(r'$Y$')
ax2.set_ylabel(r'$Z$')
plt.suptitle(f'Selected observers {which_obs}', fontsize=15)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/outflow/observers_{which_obs}.png', bbox_inches = 'tight')

#%% Load data -----------------------------------------------------------------
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
box = np.zeros(6)
if alice:
    loadpath = f'{pre}/snap_{snap}'
else:
    loadpath = f'{pre}/{snap}'
X = np.load(f'{loadpath}/CMx_{snap}.npy')
Y = np.load(f'{loadpath}/CMy_{snap}.npy')
Z = np.load(f'{loadpath}/CMz_{snap}.npy')
VX = np.load(f'{loadpath}/Vx_{snap}.npy')
VY = np.load(f'{loadpath}/Vy_{snap}.npy')
VZ = np.load(f'{loadpath}/Vz_{snap}.npy')
T = np.load(f'{loadpath}/T_{snap}.npy')
Den = np.load(f'{loadpath}/Den_{snap}.npy')
Vol = np.load(f'{loadpath}/Vol_{snap}.npy')
box = np.load(f'{loadpath}/box_{snap}.npy')
denmask = Den > 1e-19
X, Y, Z, VX, VY, VZ, T, Den, Vol = \
    make_slices([X, Y, Z, VX, VY, VZ, T, Den, Vol], denmask)
R = np.sqrt(X**2 + Y**2 + Z**2)    
xyz = np.array([X, Y, Z]).T
N_ray = 5_000
with open(f'{abspath}/data/{folder}/outflow/den_prof{snap}{which_obs}.txt','w') as file:
        file.close()
for j, idx_list in enumerate(indices_chosen):
    r_mean = []
    d_mean = []
    v_rad_mean = []
    v_tot_mean = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    for idx_i, i in enumerate(idx_list):  
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # Box 
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
        r = r * np.sqrt(mu_x**2 + mu_y**2 + mu_z**2) # r is the distance from the BH
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size = 50) 
        _, idx = tree.query(xyz2, k=1) #you can do k=4, comment the line afterwards and then do d = np.mean(d, axis=1) and the same for all the quantity. But you doesn't really change since you are averaging already on line of sights
        idx = [ int(idx[i][0]) for i in range(len(idx))]
        idx = np.array(idx)
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        dim_cell = Vol[idx]**(1/3)
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        
        v_rad, v_theta, v_phi = to_spherical_components(ray_vx, ray_vy, ray_vz, lat_obs[i], long_obs[i])
        v_tot = np.sqrt(ray_vx**2 + ray_vy**2 + ray_vz**2)
        r_mean.append(r)
        d_mean.append(d)
        v_rad_mean.append(v_rad)
        v_tot_mean.append(v_tot)

        ax1.plot(xyz2[:,0]/apo, xyz2[:,1]/apo, c = colors_obs[j])
        img1 = ax1.scatter(ray_x/apo, ray_y/apo, c = np.abs(ray_z)/apo, cmap = 'jet', label = 'From simulation', norm = colors.LogNorm(vmin = 8e-3, vmax = 3))
        # # ax1.legend(fontsize = 18)
        ax2.plot(ray_x/apo, ray_y/apo, c = 'k', alpha = 0.5)
        img2 = ax2.scatter(ray_x/apo, ray_y/apo, c = np.abs(v_rad)*conversion_sol_kms, cmap = 'jet', norm = colors.LogNorm(vmin = 1e-1, vmax = 2e4))
    
    ax1.set_ylabel(r'Y [$R_{\rm a}$]')
    cbar = plt.colorbar(img1, orientation = 'horizontal')
    cbar.set_label(r'Z [R$_{\rm a}]$')
    cbar = plt.colorbar(img2, orientation = 'horizontal')
    cbar.set_label(r'$|V_r|$ [km/s]')
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'X [$R_{\rm a}$]')
        ax.set_xlim(np.min(ray_x)/apo, 0.5*np.max(ray_x)/apo)
        ax.set_ylim(np.min(ray_y)/apo, 0.5*np.max(ray_y)/apo)
    ax1.set_title('Points wanted and selected', fontsize = 18)
    ax2.set_title('Velocity', fontsize = 18)
    plt.suptitle(f'Observer {label_obs[j]}, snap {snap}', fontsize = 20)
    plt.tight_layout()
    fig.savefig(f'{abspath}/Figs/outflow/insights/selectedCells{snap}{which_obs}{j}.png', bbox_inches = 'tight')
    
    r_mean = np.mean(r_mean, axis=0)
    d_mean = np.mean(d_mean, axis=0)
    v_rad_mean = np.mean(v_rad_mean, axis=0)
    v_tot_mean = np.mean(v_tot_mean, axis=0)

    with open(f'{abspath}/data/{folder}/outflow/den_prof{snap}{which_obs}.txt','a') as file:
        file.write(f'# Observer latitude: {lat_obs[i]}, longitude: {long_obs[i]}\n')
        file.write(f' '.join(map(str, r_mean)) + '\n')
        file.write(f' '.join(map(str, d_mean)) + '\n')
        file.write(f' '.join(map(str, v_rad_mean)) + '\n')
        file.write(f' '.join(map(str, v_tot_mean)) + '\n')
        file.close()

#%%
x_test = np.arange(1e-3, 1e2)
y_test2 = 1e-17 * (x_test/apo)**(-2)
y_test3 = 2e-21 * (x_test/apo)**(-3)
y_test4 = 1e-24 * (x_test/apo)**(-4)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
fig1, ax2 = plt.subplots(1, 1, figsize=(8, 7))
fig2, ax3 = plt.subplots(1, 1, figsize=(8, 7))

profiles = np.loadtxt(f'{abspath}/data/{folder}/outflow/den_prof{snap}{which_obs}.txt')
r_arr, d_arr, v_rad_arr, v_tot_arr = profiles[0::4], profiles[1::4], profiles[2::4], profiles[3::4]

for i in range(len(indices_chosen)):
    r = r_arr[i]
    d = d_arr[i]
    v_rad = v_rad_arr[i]
    ax1.plot(r/apo, d, color = colors_obs[i], label = f'{label_obs[i]}')#Observer {label_obs[i]} ({indices_chosen[i]})')
    ax2.plot(r/apo, np.abs(v_rad)*prel.Rsol_cgs*1e-5/prel.tsol_cgs, color = colors_obs[i], label = f'{label_obs[i]}')#Observer {label_obs[i]} ({indices_chosen[i]})')
    ax3.plot(r/apo, r**2*d*np.abs(v_rad)*prel.Rsol_cgs**3/prel.tsol_cgs, color = colors_obs[i], label = f'{label_obs[i]}')#Observer {label_obs[i]} ({indices_chosen[i]})')
ax1.plot(x_test, y_test2, c = 'gray', ls = 'dashed', label = r'$\rho \propto R^{-2}$')
ax1.plot(x_test, y_test3, c = 'gray', ls = 'dotted', label = r'$\rho \propto R^{-3}$')
ax1.plot(x_test, y_test4, c = 'gray', ls = '-.', label = r'$\rho \propto R^{-4}$')
# ax1.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
ax1.set_ylim(2e-19, 5e-6)
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax2.set_ylabel(r'$|v_r|$ [km/s]')
ax3.set_ylabel(r'$|v_r| \rho R^2$ [g/s]')
xmin = Rt/apo
xmax = 400*Rt/apo
for ax in [ax1, ax2, ax3]:
    #put the legend if which_obs != 'all_rotate'. Lt it be outside
    boxleg = ax.get_position()
    ax.set_position([boxleg.x0, boxleg.y0, boxleg.width * 0.8, boxleg.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
    ax.grid()
    ax.set_xlabel(r'R [R$_a]$')
    ax.set_xlim(xmin, xmax)
    ax.loglog()
plt.tight_layout()
fig.savefig(f'{abspath}/Figs/outflow/den_prof{snap}{which_obs}.png', bbox_inches = 'tight')
fig1.savefig(f'{abspath}/Figs/outflow/vel_prof{snap}{which_obs}.png', bbox_inches = 'tight')
fig2.savefig(f'{abspath}/Figs/outflow/vel_prof_rhoR2v{snap}{which_obs}.png', bbox_inches = 'tight')
plt.show()

#%%
eng.exit()

#%%
