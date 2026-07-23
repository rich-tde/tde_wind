""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
#%% import resource
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import healpy as hp
from sklearn.neighbors import KDTree
from src.Opacity.interpolator_vectorized import calc_planck_opacity_vectorized, calc_ross_opacity_vectorized, calc_scattering_opacity_vectorized
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
from Utilities.operators import make_tree, choose_observers, to_spherical_components
from src import orbits as orb
from src.Wind.Rtrapp_tdiff import load_and_smooth_rtrap

def single_fld(loadpath, snap, observers_xyz, N_ray):
    num_obs = len(observers_xyz)
    data = make_tree(loadpath, snap)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den = \
        data.X, data.Y, data.Z, data.Temp, data.Den
    denmask = Den > 1e-19
    X, Y, Z, T, Den = \
        make_slices([X, Y, Z, T, Den], denmask)
    xyz = np.array([X, Y, Z]).T

    all_obs = {}
    for i in range(num_obs):
        # if i not in [0, 90]:
        #     continue
        print(f'Obs: {i}', flush=True)

        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # be sure is normalized, but it should be because it's a healpy vector
        norm_mu = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
        if norm_mu != 1.:
            print('normalizing observers direction. norm:', norm_mu, flush=True)
            mu_x /= norm_mu
            mu_y /= norm_mu
            mu_z /= norm_mu

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
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size=50) 
        _, idx = tree.query(xyz2, k=1)
        idx = idx.ravel()
        d = Den[idx] * prel.den_converter
        t = T[idx]

        alpha_scatter = calc_scattering_opacity_vectorized(T_cool, Rho_cool, scattering, np.log(t), np.log(d))
        alpha_scatter = np.array(alpha_scatter)
        alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rossland, scattering, np.log(t), np.log(d))
        alpha_rossland = np.array(alpha_rossland)
        alpha_planck = calc_planck_opacity_vectorized(T_cool, Rho_cool, planck, np.log(t), np.log(d))
        alpha_planck = np.array(alpha_planck)
        
        underflow_mask = np.logical_and(np.logical_and(np.log(alpha_rossland) != 0.0, np.log(alpha_planck) != 0.0), np.log(alpha_scatter) != 0.0)
        d, t, r, alpha_rossland, alpha_planck, alpha_scatter = \
            make_slices([d, t, r, alpha_rossland, alpha_planck, alpha_scatter], underflow_mask)
        
        single_obs = {'r': r, 
                      'd': d, 
                      't': t, 
                      'alpha_rossland': alpha_rossland, 
                      'alpha_planck': alpha_planck, 
                      'alpha_scatter': alpha_scatter}

        key = f'obs_{i}'
        all_obs[key] = single_obs

    return all_obs

#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 151
N_ray = 1_000
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rt = things['Rt']
apo = things['apo']
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre_saving = f'{abspath}/data/{folder}'
loadpath = f'{pre}/{snap}'
which_obs = 'split_stream' 
isoent = 'isoent'
compute = False 

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
indices_obs, label_obs, colors_obs, _, _= choose_observers(observers_xyz, which_obs)
observers_xyz = np.array(observers_xyz).T

# Load opacity tables
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
planck = np.loadtxt(f'{opac_path}/planck.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm

#%%
if compute:
    all_obs = single_fld(loadpath, snap, observers_xyz, N_ray) 
    np.save(f'{pre_saving}/wind/kappa_fromFLD{snap}.npy', all_obs, allow_pickle=True)
else:
    
    all_obs = np.load(f'{pre_saving}/wind/kappa_fromFLD{snap}.npy', allow_pickle=True).item()
    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    xph, yph, zph = photo['x'], photo['y'], photo['z']
    rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
    pathtrap = f"{abspath}/data/{folder}/trap"
    dataRtr = load_and_smooth_rtrap(pathtrap, check, snap)
    x_tr, y_tr, z_tr, d_tr, Vr_tr, radden_tr = \
        dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['Vr_tr'], dataRtr['Rad_den_tr']
    r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    # check xi from Miller15 (eq.6)
    profiles = np.load(f'{abspath}/data/{folder}/wind/r_profile/r{isoent}_profSec{snap}_{which_obs}_wind.npy', allow_pickle=True).item()

    kappa_tr = np.zeros(len(r_tr_all))
    Mdot_tr = 4 * np.pi * r_tr_all**2 * d_tr* Vr_tr
    Mdot_tr *= prel.Msol_cgs/prel.tsol_cgs

    # they will be of shape = (len(indices_obs), len(r))
    r_all = []
    d_all = []
    t_all = []
    kappa_all = []

    rph_medians = np.zeros(len(indices_obs))
    rph_nonzero_medians = np.zeros(len(indices_obs))
    rtr_medians = np.zeros(len(indices_obs))
    rtr_nonzero_medians = np.zeros(len(indices_obs))
    figs, axs = plt.subplots(1, len(indices_obs), figsize=(7*len(indices_obs), 7))
    for k, indices in enumerate(indices_obs):
        # if label_obs[k] == 'South pole':
        #     continue
        # print(label_obs[k],indices)
        rph_medians[k] = np.median(rph_all[indices])
        rtr_medians[k] = np.median(r_tr_all[indices])
        non_zero = indices[r_tr_all[indices]> Rt]
        print(f'Obs: {label_obs[k]}: non zero = {len(non_zero)/len(indices)*100:.1f}%')
        rph_nonzero_medians[k] = np.median(rph_all[non_zero]) if len(non_zero) > 0 else rph_medians[k]
        rtr_nonzero_medians[k] = np.median(r_tr_all[non_zero]) if len(non_zero) > 0 else 0
        r_sec = [] 
        d_sec = []
        t_sec = []
        kappa_sec = []
        for i in range(len(observers_xyz)):
            axs[k].set_title(label_obs[k])
            if r_tr_all[i] == 0:
                continue
            r = all_obs[f'obs_{i}']['r']
            d = all_obs[f'obs_{i}']['d']
            t = all_obs[f'obs_{i}']['t']
            alpha_ross = all_obs[f'obs_{i}']['alpha_rossland']
            kappa = alpha_ross/d
            idx_tr = np.argmin(np.abs(r - r_tr_all[i]))
            kappa_tr[i] = alpha_ross[idx_tr]/d[idx_tr]
            if i not in indices:
                continue
            # print(label_obs[k], i)
            # if label_obs[k] == 'Stream side':
            axs[k].plot(r/Rt, kappa, label = f'Obs {i}')
            axs[k].scatter(r[idx_tr]/Rt, kappa[idx_tr], marker='d', edgecolors='k', s=60, zorder=3)

            r_sec.append(r)
            d_sec.append(d)
            t_sec.append(t)
            kappa_sec.append(kappa) #if len(d)==0 else np.zeros((0,len(r)))
        
        # np.shape(r_sec) = (len(indices[nonzero]), len(r)) where len(r) is the same for all observers in the every section
        r_all.append(np.median(np.array(r_sec), axis=0))
        d_all.append(np.median(np.array(d_sec), axis=0))
        t_all.append(np.median(np.array(t_sec), axis=0))
        kappa_all.append(np.median(np.array(kappa_sec), axis=0))
    
    fig, (axx, axk) = plt.subplots(2,1, figsize=(7, 10))
    axes = np.concatenate([[axx, axk], [axs[l] for l in range(len(indices_obs))]])
    for i, lab in enumerate(profiles.keys()):
        r_arr = profiles[lab]['r'] 
        v_rad = np.abs(profiles[lab]['v_rad_prof'])
        if isoent == 'isoent': 
            print('Isoentropic Mdot and Lum for xi')
            area = profiles[lab]['area']
            Mdot = (4 * np.pi * r_arr**2) /area * profiles[lab]['Mdot_prof']
            L_adv = (4 * np.pi * r_arr**2) / area * profiles[lab]['L_adv_prof']
        else:
            Mdot = profiles[lab]['Mdot_prof'] 
            L_adv = profiles[lab]['L_adv_prof']
        n_e = Mdot/(prel.m_p_cgs/prel.Msol_cgs * 4 * np.pi * r_arr**2 * v_rad)
        xi = L_adv/(n_e * r_arr**2) 
        xi_cgs = xi * prel.en_converter * prel.Rsol_cgs/prel.tsol_cgs
        # delete the nan values in xi_cgs and the corresponding r_arr values
        nan_mask = np.isnan(xi_cgs)
        xi_cgs = xi_cgs[~nan_mask]
        r_arr = r_arr[~nan_mask]

        if lab == 'South pole' or rtr_nonzero_medians[i] == 0:
            continue
        idx_stop_xi = np.argmin(np.abs(r_arr - rtr_nonzero_medians[i]))
        idx_rtr = np.argmin(np.abs(r_all[i] - rtr_nonzero_medians[i]))
        idx_rph = np.argmin(np.abs(r_all[i] - rph_nonzero_medians[i]))

        axk.plot(r_all[i]/Rt, kappa_all[i], label=lab, color=colors_obs[i])
        axk.scatter(rph_nonzero_medians[i]/Rt, kappa_all[i][idx_rph], color=colors_obs[i], marker='o', s=60, edgecolors = 'k', zorder=3)
        axk.scatter(rtr_nonzero_medians[i]/Rt, kappa_all[i][idx_rtr], color=colors_obs[i], marker='d', s=60, edgecolors = 'k', zorder=3)
        axx.plot(r_arr[:idx_stop_xi]/Rt, xi_cgs[:idx_stop_xi], color=colors_obs[i],  label=lab)
        axx.scatter(rtr_nonzero_medians[i]/Rt, xi_cgs[idx_stop_xi], color=colors_obs[i], marker='d', s=60, edgecolors = 'k', zorder=3)
    
    axk.axhline(0.34, color='k', ls='-.')
    axk.text(2.7e2, 0.4, r'$\kappa_{\rm es}$', fontsize=24)
    axk.set_ylim(.1, 25)
    axk.set_ylabel(r'$\kappa$ (cm$^2$/g)')
    axk.set_xlabel(r'$r /r_{\rm t}$')
    axx.axhline(5000, color='k', ls='-.')
    axx.set_ylim(1e1, 1.2e5)
    axx.set_ylabel(r'$\xi$ (erg cm/s)')
    axx.legend(fontsize=14)
    for ax in axes:
        ax.loglog()
        ax.set_xlim(1, 5e2)
        ax.tick_params(axis='both', which='major', length=12, width=1.2)
        ax.tick_params(axis='both', which='minor', length=6, width=1)
        ax.grid()
        plt.tight_layout()
    
    fig.savefig(f'{abspath}/Figs/2.paperWind/opacity.pdf', dpi=300, bbox_inches='tight')
    figs.savefig(f'{abspath}/Figs/{folder}/wind/opacity_Rprof{snap}_{which_obs}ALL.png', dpi=300, bbox_inches='tight')

    # %% test for photosphere
    gamma = 1/7
    d_ph, alphaRoss_ph, Vx_ph, Vy_ph, Vz_ph, radden_ph, Lumph = \
        photo['den'], photo['alpha_rossland'], photo['vx'], photo['vy'], photo['vz'], photo['radden'], photo['Lum']
    Trad_ph = (radden_ph * prel.en_den_converter/prel.alpha_cgs)**(1/4)
    kappa_ph = alphaRoss_ph/d_ph
    Vr_ph, _, _ = to_spherical_components(Vx_ph, Vy_ph, Vz_ph, xph, yph, zph) 
    Vr_ph = np.array(Vr_ph) * prel.Rsol_cgs/prel.tsol_cgs
    # rph_rtr_approx = Vr_tr * ratios_k / prel.csol_cgs * d_tr/d_ph 
    rph_rtr_approx = (7*gamma-4)/(7*gamma-6) * kappa_ph/kappa_tr * prel.csol_cgs  /Vr_tr

    fig, ((axRratio, axT), (axR, axL)) = plt.subplots(2, 2, figsize=(15, 15))
    axRratio.scatter(rph_all/r_tr_all, rph_rtr_approx, color='k', s=60, edgecolors='k')
    axRratio.plot([0, 10], [0, 10], color='r', ls='--', lw=1.5)
    axRratio.set_xlabel(r'$r_{\rm ph}/r_{\rm tr}$ from simulation')
    axRratio.set_ylabel(r'$\frac{c}{v_{\rm tr}} \frac{\kappa_{\rm ph}}{\kappa_{\rm tr}} \frac{7\gamma-4}{7\gamma-6}$')
    axRratio.set_xlim(0, 5)
    axRratio.set_ylim(0, 5)

    rtr_approx = kappa_tr * Mdot_tr / (4 * np.pi * prel.c_cgs * np.abs(3.5*gamma-2))
    Mdot_ph = 4 * np.pi * (rph_all*prel.Rsol_cgs)**2 * d_ph * Vr_ph
    rph_approx = kappa_ph * Mdot_ph / (4 * np.pi * Vr_ph * np.abs(3.5*gamma-3))
    axR.scatter(np.arange(len(rph_all)), rph_all * prel.Rsol_cgs / rph_approx, color='k', s=60, label = r'$r_{\rm ph}$')
    axR.scatter(np.arange(len(r_tr_all)), r_tr_all * prel.Rsol_cgs / rtr_approx, color='b', s=60, label = r'$r_{\rm tr}$')
    axR.axhline(1, color='r', ls='--', lw=1.5)
    axR.set_ylabel(r'$r_{\rm sim}/ r_{\rm approx}$')
    axR.set_xlabel(r'$N_{\rm obs}$')
    axR.set_yscale('log')

    Lum_tr = 4 * np.pi * r_tr_all**2 * radden_tr * Vr_tr * prel.en_converter/prel.tsol_cgs
    Tph_approx = (4 * np.pi * Vr_ph**2 * Lum_tr * beta**2 / (kappa_ph**2 * Mdot_ph**2 * prel.sigmaB_cgs))**(1/4)
    axT.scatter(np.arange(len(rph_all[r_tr_all != 0])), (Trad_ph/Tph_approx)[r_tr_all != 0], color='k', s=60, edgecolors='k')
    axT.axhline(1, color='r', ls='--', lw=1.5)
    axT.set_ylabel(r'$T_{\rm ph, sim}/ T_{\rm ph, approx}$')
    axT.set_xlabel(r'$N_{\rm obs}$')
    axT.set_yscale('log')
    # axR.set_ylim(5e-3, 2)

    axL.scatter(Lumph, Lum_tr, color='k', s=60, edgecolors='k')
    axL.plot([0, 1e45], [0, 1e45], color='r', ls='--', lw=1.5)
    axL.loglog()
    axL.set_xlabel(r'$L_{\rm ph, sim}$')
    axL.set_ylabel(r'$L_{\rm tr, sim}$')
    axL.set_xlim(1e40, 1e43)
    axL.set_ylim(1e40, 1e43)
    axR.legend(fontsize=20)
    for ax in [axRratio, axR, axT, axL]:
        ax.tick_params(axis='both', which='major', length=8, width=1.2)
        ax.tick_params(axis='both', which='minor', length=5, width=1)
        ax.grid()
    plt.suptitle(f'snap {snap}, gamma = {gamma:.2f}', fontsize=20)
    plt.tight_layout()

# %%
