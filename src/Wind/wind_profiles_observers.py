""" Find/plot radial profiles as weighted average on spherical sections. 
Find/plot polar profiles for fixed r and phi_array. 
Written to be run locally."""

import enum
from math import dist
import sys

from sklearn import tree
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import lines as mlines
import scipy.integrate as sci
import healpy as hp
from sklearn.neighbors import KDTree
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
import Utilities.operators as op
from src.Opacity.interpolator_vectorized import calc_ross_opacity_vectorized

#
# PARAMS
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs

# Load opacity tables
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt')

#%% FUNCTIONS
def split_observers(X, Y, Z, dim_cell, which_obs, r_chosen, params_obs):
    x_obs, y_obs, z_obs, indices_obs, colors_obs, lines_obs = params_obs
    xyz = np.transpose([X/r_chosen, Y/r_chosen, Z/r_chosen]) #/Rtalize to r_chosen
    tree = KDTree(xyz) 
    sections_tocheck = op.choose_sections(X, Y, Z, which_obs)
    indices_all = np.arange(len(X))
    indices_sec_tocheck = []
    for key in sections_tocheck.keys():
        cond_sec_tocheck = sections_tocheck[key]['cond']
        indices_sec_tocheck.append(indices_all[cond_sec_tocheck])

    indices_sec = []
    for j, indices in enumerate(indices_obs):
        x_obs_sec = x_obs[indices]
        y_obs_sec = y_obs[indices]
        z_obs_sec = z_obs[indices]
        dist, idx = tree.query(np.transpose([x_obs_sec, y_obs_sec, z_obs_sec]), k = 80)
        dist = dist.flatten()
        idx = idx.flatten()
        cut_dim = dist < dim_cell[idx]
        idx = idx[cut_dim]
        correct_idx = np.intersect1d(idx, indices_sec_tocheck[j])
        indices_sec.append(correct_idx)
    return indices_sec

def profiles(loadpath, snap, ray_params, params_obs, which_part = ''):
    x_obs, y_obs, z_obs, _, _, _ = params_obs
    rmin, rmax, Nray = ray_params
    ray_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)

    data = op.make_tree(loadpath, snap)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
    dim_cell = Vol**(1/3)

    cut_wind, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
    if which_part == 'wind':
        cut = cut_wind
    elif which_part == 'outflow':
        cut = V_r >= 0 
    elif which_part == 'all':
        cut = Den > 1e-19
    X, Y, Z, Rsph, Vol, Den, Mass, V_r, T, Rad_den, dim_cell = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, V_r, T, Rad_den, dim_cell], cut)  
    
    xyz = np.transpose([X, Y, Z]) 
    tree = KDTree(xyz) 

    fig, (axmid, ax_xz) = plt.subplots(1, 2, figsize=(8, 4))
    for ax in [axmid, ax_xz]:
        ax.set_aspect('equal')
        ax.set_xlabel(r'x (r$_t$)', fontsize = 18)
        ax.set_ylabel(r'y (r$_t$)', fontsize = 18)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    all_outflows = {}
    d_prof = []
    v_rad_prof = []
    m_prof = []
    t_prof = []
    Mdot_prof = []
    L_kin_prof = []
    L_adv_prof = []
    for i in range(len(x_obs)):
        d_prof = np.zeros(len(ray_array))
        v_rad_prof = np.zeros(len(ray_array))
        m_prof = np.zeros(len(ray_array))
        t_prof = np.zeros(len(ray_array))
        Mdot_prof = np.zeros(len(ray_array))
        L_kin_prof = np.zeros(len(ray_array))
        L_adv_prof = np.zeros(len(ray_array))
        points = np.vstack([ray_array * x_obs[i], ray_array * y_obs[i], ray_array * z_obs[i]]).T
        dist, idx = tree.query(points, k = 10)
        for j, r in enumerate(ray_array):
            cut = dist[j] < dim_cell[idx[j]]
            idx_r = np.array(idx[j][cut])
            if len(idx_r) == 0:
                continue

            ray_d = Den[idx_r]
            ray_m = Mass[idx_r]
            ray_V_r = V_r[idx_r] 
            ray_rad_den = Rad_den[idx_r]
            ray_vol = Vol[idx_r]
            ray_t = (ray_rad_den * prel.en_den_converter / prel.alpha_cgs)**(1/4)
            L_adv = ray_V_r * ray_rad_den

            # if j == 100:
            #     axmid.scatter(X[idx_r]/Rt, Y[idx_r]/Rt, s = 5, c = ray_d * prel.den_converter, norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-9))
            #     img = ax_xz.scatter(X[idx_r]/Rt, Z[idx_r]/Rt, s = 5, c = ray_d * prel.den_converter, norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-9))

            d_prof[j] = np.sum(ray_d*ray_m)/ np.sum(ray_m) 
            v_rad_prof[j] = np.sum(ray_V_r*ray_m) / np.sum(ray_m)  
            m_prof[j] = np.mean(ray_m)
            t_prof[j] = np.sum(ray_t*ray_vol) / np.sum(ray_vol)
            L_adv_prof[j] = 4 * np.pi * r**2 * np.mean(L_adv) # surface of each helpix cell: 4pir^2/192
            Mdot_prof[j] = 4 * np.pi * r**2 * np.mean(ray_d * ray_V_r)
            L_kin_prof[j] = 4 * np.pi * r**2 * np.mean(ray_d * ray_V_r**3)
        
        alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rossland, scattering, np.log(t_prof), np.log(d_prof))
        alpha_rossland = np.array(alpha_rossland)
        # underflow_mask = np.log(alpha_rossland) != 0.0
        # idx = np.array(idx)

        # Optical depth
        r_fuT = np.flipud(ray_array) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        outflow = {
            'r': ray_array,
            'd_prof': d_prof, 
            'v_rad_prof': v_rad_prof,
            'm_prof': m_prof,
            't_prof': t_prof,
            'tau': los,
            'Mdot_prof': Mdot_prof,
            'L_kin_prof': L_kin_prof,
            'L_adv_prof': L_adv_prof
        }

        key = f"{i}"
        all_outflows[key] = outflow

    return all_outflows


#   
## MAIN
#
compute = False
which_part = 'wind' # 'outflow' or 'all' or 'wind' to have the wind
which_obs = 'split_stream' # 'left_right_z', 'all' or 'in_out_z'
NSIDE = 4 # observers = 12 * NSIDE **2
NPIX = hp.nside2npix(NSIDE)
observers_xyz = np.array(hp.pix2vec(NSIDE, range(NPIX))) # shape is 3,N
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
indices_obs, label_obs, colors_obs, lines_obs, _ = op.choose_observers(observers_xyz, which_obs)
params_obs = [x_obs, y_obs, z_obs, indices_obs, colors_obs, lines_obs]

if compute:
    snaps = [151]
    for snap in snaps: 
        path = f'{pre}/{snap}'
        ray_params = [Rt, 1e3*Rt, 300]
        r_chosen_name = ''

        all_outflows = profiles(path, snap, ray_params, params_obs, which_part)
        out_path = f"{abspath}/data/{folder}/wind/r_profile/Obs_profSec{snap}_{which_part}.npy"
        np.save(out_path, all_outflows, allow_pickle=True)

else:
    snap = 151
    # Load data Rph and Rtr
    path = f'{pre}/{snap}'
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
    fig, axA = plt.subplots(1, 1, figsize = (8, 8))
    figcheck, (axd, axM) = plt.subplots(2,1, figsize = (8, 16))
    # Load profiles
    profiles = np.load(f'{abspath}/data/{folder}/wind/r_profile/Obs_profSec{snap}_{which_part}.npy', allow_pickle=True).item()
    Mdot_all = []
    Mdot_sec = []
    d_sec = []
    figtau, axtau = plt.subplots(1, 1, figsize = (8, 8))
    axtau.set_ylim(1e-2, 1e2)
    axtau.loglog()
    for j, idx in enumerate(indices_obs):
        d_temp = []
        m_temp = []
        Mdot_temp = []
        for i in idx:
            lab = f'{i}'
            Mdot = profiles[lab]['Mdot_prof'] #Mdot_prof
            Mdot_all.append(Mdot)
            r_plot = profiles[lab]['r'] 
            m_prof = profiles[lab]['m_prof']
            d_prof = profiles[lab]['d_prof']
            tau = profiles[lab]['tau']
            d_temp.append(d_prof)
            m_temp.append(m_prof)
            Mdot_temp.append(Mdot)
            if i in [0, 84, 90]:
                axtau.plot(r_plot/Rt, tau, color = colors_obs[j], label = f'Observer {i}', linewidth = 2)
        d_temp = np.array(d_temp) 
        m_temp = np.array(m_temp)
        Mdot_temp = np.array(Mdot_temp)
        Mdot_sec.append(48 * np.mean(Mdot_temp, axis = 0)) # you imagined it for a quarter of as sphere
        d_sec.append(np.sum(d_temp * m_temp, axis = 0)/np.sum(m_temp, axis = 0)) 
        # d_sec.append(np.mean(d_temp, axis = 0))

    axtau.legend(fontsize = 18)
    Mdot_sec = np.array(Mdot_sec) # shape:4, 300
    d_sec = np.array(d_sec)
    Mdot_all = np.array(Mdot_all) # shape is Nobs, Nray 
    Mdot_mean = np.mean(Mdot_all, axis = 0) 
    variance = 1/192 * np.sum(Mdot_all**2 - Mdot_mean**2, axis = 0) 
    A_param =  variance/Mdot_mean**2

    axA.plot(r_plot/Rt, A_param, linewidth = 2)

    for j, idx in enumerate(indices_obs):
        axd.plot(r_plot/Rt, d_sec[j] * prel.den_converter, color = colors_obs[j], linewidth = 2)
        axM.plot(r_plot/Rt, Mdot_sec[j]/Medd_sol, color = colors_obs[j])
    
    for ax in [axd, axM, axA]:
        ax.loglog()
        ax.set_xlim(1.5, 1.4e2)
        ax.grid()
        ax.tick_params(axis = 'both', which = 'major', length = 8, width = 1.5)
        ax.tick_params(axis = 'both', which = 'minor', length = 6, width = 1)
    axM.set_xlabel(r'$r /r_{\rm t}$', fontsize = 28)
    axd.set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 28)
    axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 28)
    axA.set_ylabel(r'A', fontsize = 28)
    axd.set_ylim(2e-13, 1e-5)
    axM.set_ylim(1e2, 1e7)


