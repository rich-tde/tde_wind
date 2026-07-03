"""Compute trapping radius i.e. R: tau(R) = c/v(R) and diffusion and dynamical time in the radial direction"""
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    compute = True

import gc
import numpy as np
# from scipy.integrate import cumulative_trapezoid
import healpy as hp
from scipy.integrate import cumulative_trapezoid
from sklearn.neighbors import KDTree
from src.Opacity.interpolator_vectorized import calc_ross_opacity_vectorized
import Utilities.prelude as prel
from Utilities.operators import make_tree, sort_list, choose_observers, to_spherical_coordinate
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
check = 'HiResNewAMR' 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
pre_saving = f'{abspath}/data/{folder}'

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
norm = things['E_mb']
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
t_fb_sol = tfallback_cgs/prel.tsol_cgs

# Opacity
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
planck = np.loadtxt(f'{opac_path}/planck.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt')

def r_trapp(data, ray_params):
    rmin, Nray = ray_params
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)
    
    X, Y, Z, T, Den, Mass, Vol, VX, VY, VZ, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Mass, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE, data.Rad
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)
    cut, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
    mask = np.logical_and(Den > 1e-19, cut)
    X, Y, Z, T, Den, Vol, V_r, vel, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, T, Den, Vol, V_r, vel, Press, IE_den, Rad_den], mask)
    xyz = np.array([X, Y, Z]).T
    tree = KDTree(xyz, leaf_size = 50)
    
    data_ph = np.load(f'{pre_saving}/photo/{check}_photo{snap}.npz')
    # denph/= prel.den_converter #it was saved in cgs
    xph, yph, zph = data_ph['x'], data_ph['y'], data_ph['z']
    rph = np.sqrt(xph**2 + yph**2 + zph**2)

    x_tr = np.zeros(len(observers_xyz))
    y_tr = np.zeros(len(observers_xyz))
    z_tr = np.zeros(len(observers_xyz))
    r_tr = np.zeros(len(observers_xyz))
    vol_tr = np.zeros(len(observers_xyz))
    den_tr = np.zeros(len(observers_xyz))
    Temp_tr = np.zeros(len(observers_xyz))
    Vr_tr = np.zeros(len(observers_xyz))
    V_tr = np.zeros(len(observers_xyz))
    P_tr = np.zeros(len(observers_xyz))
    IEden_tr = np.zeros(len(observers_xyz))
    Rad_den_tr = np.zeros(len(observers_xyz))
    kappa_tr = np.zeros(len(observers_xyz))
    ratio_kept = np.zeros(len(observers_xyz))

    # if plot:
    #     fig_all, ax_all = plt.subplots(1, len(indices_sorted), figsize = (len(indices_sorted)*5,6))
    indices_bigVol = []
    indices_overRph = []
    
    fig_p, (ax_T, axd, axk) = plt.subplots(3,1,figsize = (8,24))
    for i in range(len(observers_xyz)):
        if i not in [0, 70]:
            continue
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]
        
        # Box is for dynamic ray making
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
        r = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
        x = r*mu_x
        y = r*mu_y
        z = r*mu_z

        xyz2 = np.array([x, y, z]).T
        del x, y, z

        dist, idx = tree.query(xyz2, k=1)
        dist = np.concatenate(dist)
        idx = np.array([ int(idx[i][0]) for i in range(len(idx))])
        # initial_r_indices = np.arange(len(r[idx]))

        # pick them just if near enough and iterate
        # check_dist = np.abs(r_sim - radii2) < Vol[idx]**(1/3)
        # r_sim = np.sqrt(X[idx]**2 + Y[idx]**2 + Z[idx]**2)
        check_dist = dist <= Vol[idx]**(1/3) #np.logical_and(dist <= Vol[idx]**(1/3), r_sim >= Rt)
        ratio_kept[i] = np.sum(check_dist)/len(check_dist)
        idx = idx[check_dist]
        ray_r = r[check_dist] 

        if len(idx) <= 1:
            print(f'No wind cells for observers {i}', flush=True)
            # count_i += 1
            continue

        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        ray_t = T[idx]
        ray_d = Den[idx] * prel.den_converter
        ray_vol = Vol[idx]
        ray_V = vel[idx]
        ray_vr = V_r[idx] 
        ray_P = Press[idx]
        ray_ieDen = IE_den[idx]
        ray_radDen = Rad_den[idx]

        # Interpolate ----------------------------------------------------------
        alpha_rossland = calc_ross_opacity_vectorized(T_cool, Rho_cool, rossland, scattering, np.log(ray_t), np.log(ray_d))
        alpha_rossland = np.array(alpha_rossland)
        
        underflow_mask = np.log(alpha_rossland)!= 0.0
        ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, ray_P, ray_ieDen, ray_radDen, idx], underflow_mask)

        # Optical Depth
        # compute the optical depth from outside in: tau = - int alpha dr. Then reverse the order to have it from the inside to out, so can query.
        ray_fuT = np.flipud(ray_r)
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        tau = - np.flipud(cumulative_trapezoid(alpha_rossland_fuT, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_r. 
        tau_zero = tau != 0
        ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, tau, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, tau, ray_P, ray_ieDen, ray_radDen, idx], tau_zero)
        c_tau = prel.csol_cgs/tau # code units, since tau is adimensional
        ray_kappa = alpha_rossland/ray_d

        # if plot:
        #     j = next(j for j in range(len(indices_sorted)) if i in indices_sorted[j]) 
        #     _, theta, phi = to_spherical_coordinate(mu_x, mu_y, mu_z)
        #     phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
        #     phi = -phi
        tdyn_single = ray_r / ray_vr * prel.tsol_cgs # cgs
        tdiff_single = tau * ray_r * prel.Rsol_cgs / prel.c_cgs # cgs                

        #     fig, ax1 = plt.subplots(1,1,figsize = (8,6))
        #     ax1.plot(ray_r/Rt, tdyn_single/tfallback_cgs, c = 'k', label = r'$t_{\rm dyn}=r/v_r$')             
        #     img = ax1.scatter(ray_r/Rt, tdiff_single/tfallback_cgs, c = tau, cmap = 'turbo', s = 10, label = r'$t_{\rm diff}=\tau r/c$' , norm = colors.LogNorm(5e-1, 1e2)) #np.percentile(tau, 5), np.percentile(tau, 95)))
        #     cbar = plt.colorbar(img)#, orientation = 'horizontal')
        #     cbar.set_label(r'$\tau$', fontsize = 20)
        #     cbar.ax.tick_params(which = 'major', length=6, width=1)
        #     cbar.ax.tick_params(which = 'minor', length=4, width=0.8)
        #     ax1.axvline(Rt/Rt, c = 'k', linestyle = '-.', label = r'$r_{\rm t}$')
        #     ax1.set_xlabel(r'$r [r_{\rm t}]$')
        #     ax1.set_ylabel(r'$t [t_{\rm fb}]$')
        #     ax1.loglog()    
        #     ax1.set_xlim(R0/Rt, 2*rph[i]/Rt)
        #     # ax1.axvline(rph[i]/Rt, c = 'k', linestyle = 'dotted', label =  r'$r_{\rm ph}$')
        #     # ax1.set_xlim(1e-5, 8)
        #     ax1.set_ylim(1e-6, 1e2)
        #     ax1.tick_params(axis='both', which='major', length=8, width=1.2)
        #     ax1.tick_params(axis='both', which='minor', length=5, width=1)
        #     ax1.legend(fontsize = 14)
        #     plt.suptitle(f'Section: {label_obs[j]}, ' + r'$(\theta, \phi)$ = ' + f'({theta:.2f}, {phi:.2f})', fontsize = 16) #phi according to pur convention (apocenter at -pi, clockwise), \theta from Npole to Spole 
        #     plt.tight_layout()

        #     ax_all[j].plot(ray_r/Rt, ray_kappa)
        fig, axt = plt.subplots(1,1,figsize = (8,6))
        axt.plot(ray_r/Rt, tdyn_single/tfallback_cgs, c = 'k', label = r'$t_{\rm dyn}=r/v_r$')             
        img = axt.scatter(ray_r/Rt, tdiff_single/tfallback_cgs, c = ray_kappa, cmap = 'rainbow', s = 10, label = r'$t_{\rm diff}=\tau r/c$' , norm = colors.LogNorm(1e-1, 10)) #np.percentile(tau, 5), np.percentile(tau, 95)))
        cbar = plt.colorbar(img)#, orientation = 'horizontal')
        cbar.set_label(r'$\kappa$ (cm$^2$/g)', fontsize = 20)
        cbar.ax.tick_params(which = 'major', length=6, width=1)
        cbar.ax.tick_params(which = 'minor', length=4, width=0.8)
        axt.set_ylabel(r'$t (t_{\rm fb})$')
        axt.set_xlabel(r'$r (r_{\rm t})$')
        axt.tick_params(axis='both', which='major', length=8, width=1.2)
        axt.tick_params(axis='both', which='minor', length=5, width=1)

        
        ax_T.plot(ray_r/Rt, ray_t, label = f'Obs {i}')
        ax_T.set_ylabel(r'$T$')
        axd.plot(ray_r/Rt, ray_d * prel.den_converter, label = f'Obs {i}')
        axd.set_ylabel(r'$\rho$ (g/cm$^3$)')
        axk.plot(ray_r/Rt, ray_kappa, label = f'Obs {i}')
        axk.set_xlabel(r'$r (r_{\rm t})$')
        axk.set_ylabel(r'$\kappa$ (cm$^2$/g)')

        # select the inner part, where tau big --> c/tau < v (i.e. tdyn<tdiff)
        Rtr_idx_all = np.where(c_tau/ray_vr <= 1)[0]
        if len(Rtr_idx_all) == 0:
            print(f'For obs {i}, tdiff < tdyn always, no Rtr', flush=True)
            if plot:
                # fig.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{snap}/{label_axis[j]}_tdiff_Obs{i}.png')
                plt.close(fig)
            continue
        else: # take the one most outside 
            Rtr_idx = Rtr_idx_all[-1] # so if you have a gap, it takes the before point

        if ray_r[Rtr_idx]/rph[i] >= 1:
            indices_overRph.append(i)
            print(f'For obs {i}, Rtr is outside Rph', flush=True)

        # check you don't have a huge gap, otherwise it's just numerics: you don't really have 2 regimes
        if ray_vol[Rtr_idx+1]/ray_vol[Rtr_idx] > 1e3:
            indices_bigVol.append(i)
            print(f'For obs {i}, huge gap, so I skip, vol ratio: {int(ray_vol[Rtr_idx+1]/ray_vol[Rtr_idx])}', flush=True)

        x_tr[i] = ray_x[Rtr_idx]
        y_tr[i] = ray_y[Rtr_idx]
        z_tr[i] = ray_z[Rtr_idx]
        r_tr[i] = ray_r[Rtr_idx]
        vol_tr[i] = ray_vol[Rtr_idx]
        den_tr[i] = ray_d[Rtr_idx]/prel.den_converter # so is in code units
        Temp_tr[i] = ray_t[Rtr_idx]
        Vr_tr[i] = ray_vr[Rtr_idx]
        V_tr[i] = ray_V[Rtr_idx]
        P_tr[i] = ray_P[Rtr_idx]
        IEden_tr[i] = ray_ieDen[Rtr_idx]
        Rad_den_tr[i] = ray_radDen[Rtr_idx]
        kappa_tr[i] = ray_kappa[Rtr_idx]/prel.Rsol_cgs**2 * prel.Msol_cgs # to have it in sol units
        # M_dot_tr[i] = 4 * np.pi * ray_r[Rtr_idx]**2 * np.abs(Vr_tr[i]) * prel.Rsol_cgs**3/prel.tsol_cgs * den_tr[i] # den is already in cgs
        if plot:
            axt.set_ylim(1e-4, 10)
            axt.set_title(f'Obs {i}', fontsize = 16)
            for ax in [axt, ax_T, axd, axk]:
                ax.axvline(r_tr[i]/Rt, linestyle = '--')
                if ax == axt:
                    ax.axvline(rph[i]/Rt, linestyle = ':', c = 'k')
                ax.legend(fontsize = 14)
                ax.loglog()
                ax.grid()
            del ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, tau, ray_P, ray_ieDen, ray_radDen, idx, ray_kappa

    ax_T.set_ylim(1e4, 1e7)
    axd.set_ylim(1e-16, 1e-10)
    axk.set_ylim(1e-1, 10)
    axk.axhline(0.34, linestyle = '--', c = 'k')
    ax_T.legend(fontsize = 14)
    #     if plot:
    #         # search in which list of indices_sorted, which is a list of lists, is i and call it j
    #         for j in range(len(indices_sorted)):
    #             if i in indices_sorted[j]:
    #                 break
    #         ax_all[j].set_xlabel(r'$r [r_{\rm t}]$')
    #         ax_all[j].loglog()
    #         ax_all[j].set_xlim(R0/Rt, 2*apo/Rt)
    #         ax_all[j].set_ylim(1e-1, 2e2)
    #         ax_all[j].set_title(f'Observers section: {label_obs[j]}')

    # if plot:  
    #     ax_all[0].set_ylabel(r'$\kappa$ [cm$^2$/g]') 
    #     fig_all.tight_layout()
        # fig_all.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{snap}/kappa_all_{snap}.png')
    
    r_trapp = {
        'x_tr': x_tr,
        'y_tr': y_tr,
        'z_tr': z_tr,
        'r_tr': r_tr,
        'vol_tr': vol_tr,
        'den_tr': den_tr, 
        'Temp_tr': Temp_tr, 
        'Vr_tr': Vr_tr,
        'V': V_tr,
        'P_tr': P_tr,
        'IE_den_tr': IEden_tr,
        'Rad_den_tr': Rad_den_tr,
        'indices_bigVol': indices_bigVol,
        'indices_overRph': indices_overRph,
        'kappa_tr': kappa_tr,
        'ratio_kept': ratio_kept
    }
    del X, Y, Z, T, Den, Vol, vel, V_r, Press, IE_den, Rad_den
    gc.collect()

    return r_trapp

##
# MAIN
## 
#%% estimates for times
v_r_est = 5e3 #km/s
r_est = 60 * Rt * prel.Rsol_cgs 
t_dyn_est = r_est*1e-5 / (v_r_est ) # in seconds
t_dyn_est /= (3600 * 24) # in days
print(f'Estimated t_dyn: {t_dyn_est:.2f} days')
print(f'Estimated t_fb: {t_dyn_est/tfallback:.2f}')
kappa_est = 10 #cm^2/g
d_est = 1e-11 #g/cm^3
t_diff_est =  kappa_est * d_est * r_est**2 / prel.c_cgs # in seconds
t_diff_est /= (3600 * 24) # in days
print(f'Estimated t_diff: {t_diff_est:.2f} days')


#%%
if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True)
else:
    snaps = [151]

if compute:
    for snap in snaps: 
        # if snap <= 120:
        #     continue
        if alice:
            loadpath = f'{pre}/snap_{snap}'
            print(snap, flush=True)
        else: 
            choice = 'left_right_in_out_z'
            loadpath = f'{pre}/{snap}'
            observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
            indices_sorted, label_obs, colors_obs, _ = choose_observers(observers_xyz, choice)
        
        data = make_tree(loadpath, snap)
        box = np.load(f'{loadpath}/box_{snap}.npy')
        
        #%%
        r_trap = r_trapp(data, [Rt, 5000])

        # if alice:
        # np.savez(f"{pre_saving}/trap/TEST{check}_Rtr{snap}.npz", **r_trap)
#%%
if plot:
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps, tfbs, Lums = data[:, 0], data[:, 1], data[:, 2]
    tfbs, snaps, Lums = sort_list([tfbs, snaps, Lums], snaps, unique=True)
    snaps = snaps.astype(int)
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
    observers_xyz = np.array(observers_xyz)
    indices_axis, label_axis, colors_axis, lines_axis = choose_observers(observers_xyz, 'left_right_in_out_z')

    # almost = [109]
    # dataRtr = np.load(f"{abspath}/data/{folder}/wind/trap/{check}_Rtr{snap}.npz")
    # x_tr, y_tr, z_tr , den_tr, Vr_tr, kappa_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['Vr_tr'], dataRtr['kappa_tr']
    # radius_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    # Mdot_w = 4 * np.pi * radius_tr**2 * Vr_tr * den_tr
    # Mdot_Edd_k = 4 * np.pi * prel.G * Mbh / kappa_tr
    # # Mdot_w_cgs = Mdot_w * prel.Msol_cgs/prel.tsol_cgs
    # # t_dyn = radius_tr / np.abs(Vr_tr)
    # fig, ax = plt.subplots(1, 1, figsize = (8, 8)) 
    # # ax.scatter(np.arange(len(radius_tr)), t_dyn/t_fb_sol, s = 5, c = 'k')
    # # ax.set_xlabel(r'Observer index')
    # # ax.set_ylabel(r'$t_{\rm dyn}$ [t$_{\rm fb}$]')
    # ax.scatter(radius_tr/Rg, Mdot_w/Mdot_Edd_k, s = 10, c = 'k')
    # ax.scatter(radius_tr[almost]/Rg, (Mdot_w/Mdot_Edd_k)[almost], s = 20, c = 'r')
    # ax.set_xlabel(r'$r_{\rm tr} / r_{\rm g}$')
    # ax.set_ylabel(r'$\dot{M}_{\rm w} / \dot{M}_{\rm Edd}$')
    # ax.set_xlim(1e2, 1e7)
    # ax.set_ylim(1e2, 1e7)
    # ax.loglog()
    # ax.set_title(f'Snap {snap}')
    # ax.grid()
    # ax.tick_params(axis='both', which='minor', length=6, width=1)
    # ax.tick_params(axis='both', which='major', length=10, width=1.5)
    # plt.tight_layout()
    
    r_tr_sec = np.zeros((len(indices_axis), len(snaps)))
    r_trnonzero_sec = np.zeros((len(indices_axis), len(snaps)))
    NbigV_sec = np.zeros((len(indices_axis), len(snaps)))
    r_trBigV_sec = np.zeros((len(indices_axis), len(snaps)))
    r_trnonzeroBigV_sec = np.zeros((len(indices_axis), len(snaps)))
    NoverRph_sec = np.zeros((len(indices_axis), len(snaps)))
    r_trOverRph_sec = np.zeros((len(indices_axis), len(snaps)))
    r_trnonzeroOverRph_sec = np.zeros((len(indices_axis), len(snaps)))
    r_tr_tokeep = np.zeros((len(indices_axis), len(snaps)))
    unbound_ratio = np.zeros((len(indices_axis), len(snaps)))

    for s, snap in enumerate(snaps): 
        if snap != 151:
            continue
        dataRtr = np.load(f"{abspath}/data/{folder}/trap/TEST{check}_Rtr{snap}.npz") # NB it is selected to be only done by wind cells
        x_tr, y_tr, z_tr, den_tr, Vr_tr, Temp_tr, Rad_den_tr, vol_tr, kappa_tr = \
                dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['Vr_tr'], dataRtr['Temp_tr'], dataRtr['Rad_den_tr'], dataRtr['vol_tr'], dataRtr['kappa_tr']
        kappa_tr = kappa_tr * prel.Rsol_cgs**2 / prel.Msol_cgs # to have it in cgs units
        indices_bigVol, indices_overRph, ratio_kept = dataRtr['indices_bigVol'], dataRtr['indices_overRph'], dataRtr['ratio_kept']
        r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        alpha_ph, d_ph = photo['alpha_rossland'], photo['den']
        kappa_ph = alpha_ph/d_ph

        plt.figure(figsize = (8, 6))
        plt.plot(r_tr/Rt, kappa_tr, 'o', c = 'k')
        print(kappa_tr/kappa_ph, flush=True)

        for i, observer in enumerate(indices_axis):  
                unbound_ratio[i][s] = np.median(ratio_kept[observer])      
                exist_rtr = r_tr[observer] > 1.5*Rt 
                indices_nonzero = observer[exist_rtr]
                r_tr_sec[i][s] = np.median(r_tr[observer])  
                r_trnonzero_sec[i][s] = np.median(r_tr[indices_nonzero]) 

                sec_bigVol = np.array(np.intersect1d(indices_bigVol, observer), dtype = int)
                NbigV_sec[i][s] = len(sec_bigVol)
                r_tr[sec_bigVol] = 0  # to compare with old results, where you were skipping the big vol, and so was 0
                r_trBigV_sec[i][s] = np.median(r_tr[observer])
                indices_sec_NObigVol_nonzero = observer[r_tr[observer] > 1.5*Rt]
                r_trnonzeroBigV_sec[i][s] = np.median(r_tr[indices_sec_NObigVol_nonzero]) 
                
                sec_overRph = np.array(np.intersect1d(indices_overRph, observer), dtype = int)
                NoverRph_sec[i][s] = len(sec_overRph)
                r_tr[sec_overRph] = 0  # to compare with old results, where you were skipping the over Rph, and so was 0
                r_trOverRph_sec[i][s] = np.median(r_tr[observer])
                indices_sec_NOoverRph_nonzero = observer[r_tr[observer] > 1.5*Rt]
                r_trnonzeroOverRph_sec[i][s] = np.median(r_tr[indices_sec_NOoverRph_nonzero]) 
                
                r_tr_tokeep[i][s] = np.median(r_tr[observer])  

fig, (axratiokept, axr, axnonzero) = plt.subplots(1, 3, figsize=(24, 6))
figBigV, (axBigVperc, axBigV, axnonzeroBigV) = plt.subplots(1, 3, figsize=(24, 6))
figOverRph, (axOverRphperc, axOverRph, axnonzeroOverRph) = plt.subplots(1, 3, figsize=(24, 6))
figfin, axfin = plt.subplots(1, 1, figsize=(8, 6))
for i, observer in enumerate(indices_axis):
        if label_axis[i] == 'south pole':
               continue
        axratiokept.plot(tfbs, unbound_ratio[i], c = colors_axis[i], label = label_axis[i])
        axr.plot(tfbs, r_tr_sec[i]/Rt, c = colors_axis[i], label = label_axis[i])
        axnonzero.plot(tfbs, r_trnonzero_sec[i]/Rt, c = colors_axis[i]) 

        axBigVperc.plot(tfbs, NbigV_sec[i]/len(observer), c = colors_axis[i], label = label_axis[i])
        axBigV.plot(tfbs, r_trBigV_sec[i]/Rt, c = colors_axis[i])
        axnonzeroBigV.plot(tfbs, r_trnonzeroBigV_sec[i]/Rt, c = colors_axis[i]) 

        axOverRphperc.plot(tfbs, NoverRph_sec[i]/len(observer), c = colors_axis[i], label = label_axis[i])
        axOverRph.plot(tfbs, r_trOverRph_sec[i]/Rt, c = colors_axis[i])
        axnonzeroOverRph.plot(tfbs, r_trnonzeroOverRph_sec[i]/Rt, c = colors_axis[i])

        axfin.plot(tfbs, r_tr_tokeep[i]/Rt, c = colors_axis[i], label = label_axis[i])

for ax in [axratiokept, axr, axnonzero, axBigVperc, axBigV, axnonzeroBigV, axOverRphperc, axOverRph, axnonzeroOverRph, axfin]:
    ax.set_xlabel(r'$t/t_{\rm fb}$')
    ax.tick_params(axis='both', which='major', width=1.2, length=9, color = 'k')
    ax.tick_params(axis='both', which='minor', width=1, length=7, color = 'k')
    ax.set_xlim(0, np.max(tfbs))
    ax.grid()
    if ax in [axr, axBigV, axOverRph]:
        ax.set_title(r'All observers', fontsize = 20)
        ax.set_ylabel(r'median $r_{\rm tr} / r_{\rm t}$')
    elif ax in [axnonzero, axnonzeroBigV, axnonzeroOverRph]:
        ax.set_title(r'Non zeros', fontsize = 20)
        ax.legend(fontsize = 16)
    if ax not in [axratiokept, axBigVperc, axOverRphperc]:
        ax.set_yscale('log')
        ax.set_ylim(1, 100)
    ax.legend(fontsize = 16)
axratiokept.set_ylabel('Ratio unbound material', fontsize = 20)
axBigVperc.set_ylabel('Ratio obs with big gap', fontsize = 20)
axOverRphperc.set_ylabel(r'Ratio obs with $r_{\rm tr} > r_{\rm ph}$', fontsize = 20)
fig.tight_layout()
figBigV.tight_layout()
figOverRph.tight_layout()
# %%
