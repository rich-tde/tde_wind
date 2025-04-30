""" Find density and (radial velocity) profiles for different lines of sight. NB: observers have to be noramlized to 1. """
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
which_part = 'outflow'
# Rg = Mbh * prel.G / prel.csol_cgs**2
# print(Rg)
#%%
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 164
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
    indices1 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(x_obs), np.logical_and(x_obs >= 0, np.abs(y_obs) < x_obs))]
    indices2 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(y_obs), np.logical_and(y_obs >= 0, y_obs > np.abs(x_obs)))]
    indices3 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(x_obs), np.logical_and(x_obs < 0, np.abs(y_obs) < np.abs(x_obs)))]
    indices4 = obs_indices[np.logical_and(np.abs(z_obs) < np.abs(y_obs), np.logical_and(y_obs < 0, np.abs(y_obs) > np.abs(x_obs)))]
    
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
    ax1.scatter(x_obs[z_obs==0], y_obs[z_obs==0], s = 52, c = 'k', edgecolors = 'k')
    ax2.scatter(x_obs[z_obs==0], z_obs[z_obs==0], s = 52, c = 'k', edgecolors = 'k')
    ax.set_xlabel(r'$X$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
for j, idx_list in enumerate(indices_chosen):
    print(f'Observer {j}, len: {len(idx_list)}')
    ax1.scatter(x_obs[idx_list], y_obs[idx_list], s = 50, c = colors_obs[j])
    ax2.scatter(x_obs[idx_list], z_obs[idx_list], s = 50, c = colors_obs[j])
    ax1.scatter(x_obs[90:94], y_obs[90:94], s = 70,  c = 'k')
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
Mass = np.load(f'{loadpath}/Mass_{snap}.npy')
box = np.load(f'{loadpath}/box_{snap}.npy')
denmask = Den > 1e-19
X, Y, Z, VX, VY, VZ, T, Den, Vol, Mass = \
    make_slices([X, Y, Z, VX, VY, VZ, T, Den, Vol, Mass], denmask)
R = np.sqrt(X**2 + Y**2 + Z**2)    
xyz = np.array([X, Y, Z]).T
N_ray = 5_000
rmax = 10*apo
r = np.logspace(-0.25, np.log10(rmax), N_ray)

with open(f'{abspath}/data/{folder}/outflow/den_prof{snap}{which_obs}{which_part}.txt','w') as file:
        file.close()
for j, idx_list in enumerate(indices_chosen):
    print(label_obs[j])
    d_all = []
    v_rad_all = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    # fig2, ax3 = plt.subplots(1, 1, figsize=(8, 8))
    for idx_i, i in enumerate(idx_list):  
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
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
        ray_Mass = Mass[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        
        v_rad, _, _ = to_spherical_components(ray_vx, ray_vy, ray_vz, lat_obs[i], long_obs[i])
        if which_part == 'outflow':
            d[v_rad<0] = 0
            v_rad[v_rad<0] = 0
        if which_part == 'inflow':
            d[v_rad>0] = 0
            v_rad[v_rad>0] = 0
        d_all.append(d)
        v_rad_all.append(v_rad)

        ax1.plot(xyz2[:,0]/apo, xyz2[:,1]/apo, c = colors_obs[j])
        img1 = ax1.scatter(ray_x/apo, ray_y/apo, c = np.abs(ray_z)/apo, cmap = 'jet', label = 'From simulation', norm = colors.LogNorm(vmin = 8e-3, vmax = 3))
        # # ax1.legend(fontsize = 18)
        ax2.plot(ray_x/apo, ray_y/apo, c = 'k', alpha = 0.5)
        img2 = ax2.scatter(ray_x/apo, ray_y/apo, c = np.abs(v_rad)*conversion_sol_kms, cmap = 'jet', norm = colors.LogNorm(vmin = 1e-1, vmax = 2e4))
        # img3 = ax3.scatter(r/apo, d, c = t, s =2, label = f'Obs {i}', cmap = 'jet', norm = colors.LogNorm(vmin = 1e4, vmax = 1e6))

    ax1.set_ylabel(r'Y [$R_{\rm a}$]')
    # ax3.set_xlabel(r'R [$R_{\rm a}$]')
    # ax3.set_ylabel(r'$\rho$ [g/cm$^3]$')
    # ax3.set_xlim(1e-2,1)
    # ax3.set_ylim(1e-11, 1e-5)
    # ax3.loglog()
    # ax3.legend(fontsize = 18)
    # cbar3 = plt.colorbar(img3)
    # cbar3.set_label(r'$T$ [K]')
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
    fig.savefig(f'{abspath}/Figs/outflow/insights/selectedCells{snap}{which_obs}{which_part}{j}.png', bbox_inches = 'tight')

    if which_part == 'outflow' or which_part == 'inflow':
        d_all = np.transpose(d_all) # shape: N_ray, N_obs
        v_rad_all = np.transpose(v_rad_all)
        d_mean = np.zeros(len(d_all))
        v_rad_mean = np.zeros(len(v_rad_all))
        for i_ray in range(len(d_all)):
            n_nonzero = np.count_nonzero(d_all[i_ray])
            d_mean[i_ray] = np.sum(d_all[i_ray])/n_nonzero
            v_rad_mean[i_ray] = np.sum(v_rad_all[i_ray])/n_nonzero
    else:
        d_mean = np.mean(d_all, axis=0)
        v_rad_mean = np.mean(v_rad_all, axis=0)

    with open(f'{abspath}/data/{folder}/outflow/den_prof{snap}{which_obs}{which_part}.txt','a') as file:
        file.write(f'# Observer latitude: {lat_obs[i]}, longitude: {long_obs[i]}. Cut in density and in T>1e4\n')
        file.write(f' '.join(map(str, r)) + '\n')
        file.write(f' '.join(map(str, d_mean)) + '\n')
        file.write(f' '.join(map(str, v_rad_mean)) + '\n')
        file.close()

#%%
x_test = np.arange(1e-3, 1e2)
y_test2 = 4e-17 * (x_test/apo)**(-2)
y_test3 = 2e-20 * (x_test/apo)**(-3)
y_test4 = 3.5e-22 * (x_test/apo)**(-4)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
fig1, ax2 = plt.subplots(1, 1, figsize=(8, 7))
fig2, ax3 = plt.subplots(1, 1, figsize=(8, 7))
which_part = 'outflow'

profiles = np.loadtxt(f'{abspath}/data/{folder}/outflow/den_prof{snap}{which_obs}{which_part}.txt')
r_mean, d_mean, v_rad_mean = profiles[0::3], profiles[1::3], profiles[2::3]

for i in [0,1,4,2,3,5]: #because I'm stupid and I haven't put the observer in a reasonable order
    r = r_mean[i]
    d = d_mean[i]
    v_rad = v_rad_mean[i]
    if np.logical_and(label_obs[i] == '+y', which_part == 'outflow'): # you don't have points there
        d=d[r>1e-1*apo]
        v_rad=v_rad[r>1e-1*apo]
        r=r[r>1e-1*apo]
    ax1.plot(r/apo, d, color = colors_obs[i], label = f'{label_obs[i]}')# Observer {label_obs[i]} ({indices_chosen[i]})')
    ax2.plot(r/apo, np.abs(v_rad)*prel.Rsol_cgs*1e-5/prel.tsol_cgs, color = colors_obs[i], label = f'{label_obs[i]}')#Observer {label_obs[i]} ({indices_chosen[i]})')
    ax3.plot(r/apo, r**2*d*np.abs(v_rad)*prel.Rsol_cgs**3/prel.tsol_cgs, color = colors_obs[i], label = f'{label_obs[i]}')#Observer {label_obs[i]} ({indices_chosen[i]})')
ax1.plot(x_test, y_test2, c = 'gray', ls = 'dashed')#, label = r'$\rho \propto R^{-2}$')
ax1.text(4, 1e-13, r'$\propto R^{-2}$', fontsize = 20, color = 'k', rotation = -20)
ax1.plot(x_test, y_test3, c = 'gray', ls = 'dotted')#, label = r'$\rho \propto R^{-3}$')
ax1.text(.04, 3e-9, r'$\propto R^{-3}$', fontsize = 20, color = 'k', rotation = -32)
ax1.plot(x_test, y_test4, c = 'gray', ls = '-.')#, label = r'$\rho \propto R^{-4}$')
ax1.text(.1, 9e-9, r'$\propto R^{-4}$', fontsize = 20, color = 'k', rotation = -45)
# ax1.axhspan(np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool)), alpha=0.2, color='gray')
ax1.axvline(Rp/apo, c = 'k', ls = '--')
ax1.set_ylim(2e-15, 1e-5)
ax1.set_ylabel(r'$\rho$ [g/cm$^3]$')
ax2.set_ylabel(r'$|v_r|$ [km/s]')
ax3.set_ylabel(r'$|v_r| \rho R^2$ [g/s]')
ax3.set_ylim(1e18, 5e27)
xmin = 0.5*Rt/apo
xmax = 10
for ax in [ax1, ax2, ax3]:
    #put the legend if which_obs != 'all_rotate'. Lt it be outside
    # boxleg = ax.get_position()
    # ax.set_position([boxleg.x0, boxleg.y0, boxleg.width * 0.8, boxleg.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
    ax.legend(loc='lower left', fontsize = 14)
    ax.set_xlabel(r'$R [R_{\rm a}]$')
    ax.set_xlim(xmin, xmax)
    ax.loglog()
    ax.tick_params(axis='both', which='minor', size=4)
    ax.tick_params(axis='both', which='major', size=6)
    ax.grid()
plt.tight_layout()
fig.savefig(f'{abspath}/Figs/outflow/den_prof{snap}{which_obs}{which_part}.pdf', bbox_inches = 'tight')
fig1.savefig(f'{abspath}/Figs/outflow/vel_prof{snap}{which_obs}{which_part}.png', bbox_inches = 'tight')
fig2.savefig(f'{abspath}/Figs/outflow/vel_prof_rhoR2v{snap}{which_obs}{which_part}.png', bbox_inches = 'tight')
plt.show()

#%%
eng.exit()
