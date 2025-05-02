"""Compute diffusion time: tdiff = \tau*H/c"""
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
snap = 267
computation = 'avg'
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
labels_obs = [r'R$_t$', r'-a$_{min}$', r'-R$_a$']
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]


# Opacity
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering', slope_length = 5)

if alice:
    loadpath = f'{pre}/snap_{snap}'
else:
    loadpath = f'{pre}/{snap}'
data = make_tree(loadpath, snap, energy = True)
box = np.load(f'{loadpath}/box_{snap}.npy')
X, Y, Z, T, Den, Vol, Rad_den = \
    data.X, data.Y, data.Z, data.Temp, data.Den, data.Vol, data.Rad
mask = np.logical_and(Den > 1e-19, Z >= 0) # just plane above
X, Y, Z, T, Den, Vol, Rad_den = \
    make_slices([X, Y, Z, T, Den, Vol, Rad_den], mask)
xyz = np.array([X, Y, Z]).T
tree = KDTree(xyz, leaf_size = 50)
N_ray = 500

# query the nearest cell to the observer you want to have its z
rmin = 0.01
rmax = 7 * apo
r_height = np.logspace(np.log10(rmin), np.log10(rmax), N_ray) # create the z array

with open(f'{pre_saving}/{check}_tdiff{snap}{computation}.txt', 'w') as f:
    f.write(f'# Z array [R_\odot] \n' + ''.join(map(str, r_height)) + '\n')
    f.close()

if computation == 'avg':
    fig, axtau = plt.subplots(1,1, figsize = (7,5))
axtau.set_yscale('log')
axtau.set_xlabel(r'$Z [R_\odot]$')
axtau.set_ylabel(r'$\tau$')
axtau.set_xlim(1e-3, 20)
for i in range(len(observers_xyz)):
    mu_x = observers_xyz[i][0]
    mu_y = observers_xyz[i][1]
    mu_z = observers_xyz[i][2]
    # Go vertically 
    x = np.repeat(mu_x, len(r_height))
    y = np.repeat(mu_y, len(r_height))
    z = r_height

    xyz2 = np.array([x, y, z]).T
    del x, y, z

    _, idx = tree.query(xyz2, k=1)
    idx = [ int(idx[j][0]) for j in range(len(idx))] 
    idx = np.unique(idx)
    ray_x = X[idx]
    ray_y = Y[idx]
    ray_z = Z[idx]
    t = T[idx]
    d = Den[idx] * prel.den_converter
    ray_vol = Vol[idx]
    ray_weigh = Rad_den[idx] #prel.alpha_cgs/prel.en_den_converter * t**4 #Rad_den[idx]
    ray_x, ray_y, t, d, ray_vol, ray_weigh, idx, ray_z = \
        sort_list([ray_x, ray_y, t, d, ray_vol, ray_weigh, idx, ray_z], ray_z)

    # Interpolate ----------------------------------------------------------
    sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
    sigma_rossland = np.array(sigma_rossland)[0]
    underflow_mask = sigma_rossland != 0.0
    idx = np.array(idx)
    ray_x, ray_y, ray_z, t, d, ray_vol, ray_weigh, idx, sigma_rossland = \
        make_slices([ray_x, ray_y, ray_z, t, d, ray_vol, ray_weigh, idx, sigma_rossland], underflow_mask)
    sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
    del sigma_rossland
    gc.collect()


    # Optical Depth
    ray_fuT = np.flipud(ray_z) 
    kappa_rossland = np.flipud(sigma_rossland_eval) #np.flipud(d)*0.34cm2/g
    # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_z. You integrate in the z direction
    los_zero = los != 0
    ray_x, ray_y, ray_z, t, d, ray_vol, ray_weigh, idx, los, sigma_rossland_eval = \
        make_slices([ray_x, ray_y, ray_z, t, d, ray_vol, ray_weigh, idx, los, sigma_rossland_eval], los_zero)
    # you integrate for tau as BonnerotLu20, eq.16 for trapping radius
    los_scatt = - np.flipud(sci.cumulative_trapezoid(np.flipud(d)*0.34, np.flipud(ray_z), initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_z. YOu integrate in the z direction
    c_tau = prel.csol_cgs/los #c/tau [code units]
    
    # rough estimate of tdiff
    tenth_denmiind = np.argmin(np.abs(d-0.5*d[0])) #1/10 of orb.plane density
    tdiff_est = los[0] * ray_z[tenth_denmiind] * prel.Rsol_cgs / prel.c_cgs # [s] 

    # plot to check if you're taking the right thing
    # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,10))
    # cutplot = np.logical_and(Den > 1e-19, np.abs(X-mu_x)<Vol**(1/3))
    # img = ax1.scatter(Y[cutplot], Z[cutplot]/apo, c = X[cutplot]/mu_x,  cmap = 'jet', alpha = 0.7, vmin = 0.5, vmax = 1.5)
    # img = ax1.scatter(xyz2[:,1], xyz2[:,2]/apo, c = xyz2[:,0]/Rt, cmap = 'jet', marker = 's', edgecolors= 'k', label = 'What we want', vmin = 0.5, vmax = 1.5)
    # img = ax1.scatter(ray_y, ray_z/apo, c = ray_x/mu_x, cmap = 'jet', edgecolors= 'k', label = 'From simulation', vmin = 0.5, vmax = 1.5)
    # cbar = plt.colorbar(img, orientation = 'horizontal')
    # cbar.set_label(r'X [$x_{\rm chosen}$]')
    # ax1.legend(fontsize = 18)
    # img = ax2.scatter(ray_y, ray_z/apo, c = los, cmap = 'jet', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e3))
    # ax2.plot(ray_y, ray_z/apo, c = 'k', alpha = 0.5)
    # cbar = plt.colorbar(img, orientation = 'horizontal')
    # cbar.set_label(r'$\tau$')
    # ax1.set_ylabel(r'Z [$R_{\rm a}$]')
    # for ax in [ax1, ax2]:
    #     ax.set_xlabel(r'Y [$R_{\odot}$]')
    #     ax.set_xlim(-50,50)
    #     ax.set_ylim(-1/apo, 3.5)
    # ax1.set_title('Points wanted and selected', fontsize = 18)
    # ax2.set_title('Optical depth for each cell', fontsize = 18)
    # plt.suptitle(f'Observer at ({int(mu_x)}, {int(mu_y)}, {int(mu_z)}), snap {snap}', fontsize = 20)
    # plt.tight_layout()
    # Find trapping height
    # Rtr_idx_all = np.where(c_tau/np.abs(ray_vz)<1)[0]
    # if len(Rtr_idx_all) == 0:
    #     print(f'No Rtr found anywhere for obs {i}', flush=False)
    # R_tr_all = ray_z[Rtr_idx_all]
    # # Rtr < Rph
    # Rtr_idx_all_inside = np.where(np.abs(R_tr_all)<np.mean(rph))[0]
    # if len(Rtr_idx_all_inside) == 0:
    #     print(f'No Rtr inside Rph for obs {i}', flush=False)
    # Rtr_idx_all = Rtr_idx_all[Rtr_idx_all_inside]
    # # take the one most outside
    # Rtr_idx = Rtr_idx_all[-1]

    # t_dyn = ray_z/np.abs(ray_vz) * prel.tsol_cgs # [s]

    # FOR CUMULATIVE SCATTER OR ROSS
    los_scatt_fuT = np.flipud(d * 0.34 * ray_z)
    z_fuT = np.flipud(ray_z)
    tau_fuT = np.flipud(sigma_rossland_eval * ray_z)  # tau = sigma_rossland_eval * ray_z (you said earlier)
    #### FOR SINGLE
    if computation == 'avg': # average by internal energy density (don't convert because then you divide)
        ray_weigh_fuT = np.flipud(ray_weigh)
        los_fuT = tau_fuT * prel.Rsol_cgs * ray_weigh_fuT 
        tdiff_cumulative = sci.cumulative_trapezoid(los_fuT, z_fuT, initial = 0) * prel.Rsol_cgs # convert z
        tdiff_cumulative /= sci.cumulative_trapezoid(np.ones_like(ray_weigh), ray_weigh_fuT, initial = ray_weigh_fuT[0])
        #####
        # tdiff_cumulative = np.zeros(len(tdiff_ie_cumulative))
        # for k in range(len(ray_z)):
        #     ie_integ = sci.trapezoid(np.ones_like(ray_weigh[k:]), ray_weigh[k:])
        #     tdiff_cumulative[k] = tdiff_ie_cumulative[k]/ie_integ
        #####
    else:
        los_fuT = tau_fuT * prel.Rsol_cgs 
        # check ho optical depth goes
        if i == 2:
            axtau.plot(z_fuT, los_fuT, '--', label = f'x={labels_obs[i]}')
        else:
            axtau.plot(z_fuT, los_fuT, label = f'x={labels_obs[i]}')
        tdiff_cumulative = sci.cumulative_trapezoid(los_fuT, z_fuT, initial = 0) * prel.Rsol_cgs # this is the conversion for ray_z. You integrate in the z direction

    tdiff_cumulative = - np.flipud(tdiff_cumulative)/ prel.c_cgs

    # save
    with open(f'{pre_saving}/{check}_tdiff{snap}{computation}.txt', 'a') as f:
        f.write(f'# tdiff for obs at x={labels_obs[i]} [s] \n' + ''.join(map(str, tdiff_cumulative)) + '\n')
        f.close()
    
    fig, ax1  = plt.subplots(1, 1 , figsize = (7,5))
    ax1.plot(ray_z/apo, tdiff_cumulative/tfallback_cgs, c = 'k')
    ax1.axhline(tdiff_est/tfallback_cgs, color = 'deepskyblue', linestyle = '--', label = r't$_{\rm diff}\approx \tau_{\rm op} z[\rho=0.5\rho_{\rm op}/c]$')
    ax1.set_ylabel(r'$t_{\rm diff} [t_{\rm fb}]$')
    if i == 0:
        ax1.set_yscale('log')
    # ax1.set_ylim(0, 2) # a bit further than the Rtrapp
    ax1.set_xlabel(r'$Z [R_{\rm a}]$')
    ax1.set_xlim(-0.1, 3)
    ax1.legend(fontsize = 18)
    plt.suptitle(f'{check}, x = {labels_obs[i]}, snap {snap}, {computation}', fontsize = 20)
    plt.tight_layout()
axtau.legend(fontsize = 18)
#%%
eng.exit()