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
    compute = True

#%%
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import cumulative_trapezoid
import healpy as hp
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import opacity_linear, opacity_extrap
import Utilities.prelude as prel
from Utilities.operators import make_tree, sort_list, to_spherical_components, choose_observers, to_spherical_coordinate
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
scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
_, _, scatter2 = opacity_linear(T_cool, Rho_cool, scattering, slope_length = 7, highT_slope = 0)
T_cool2, Rho_cool2, rossland2 = opacity_extrap(T_cool, Rho_cool, rossland, which_opacity = 'rossland', scatter = scatter2)

def r_trapp(loadpath, snap, ray_params):
    rmin, Nray = ray_params
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)

    data = make_tree(loadpath, snap, energy = True)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE, data.Rad
    v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    mass = Den * Vol
    bern = orb.bern_coeff(R, vel, Den, mass, Press, IE_den, Rad_den, params)
    mask = np.logical_and(Den > 1e-19, np.logical_and(bern > 0, v_rad >= 0)) 

    X, Y, Z, T, Den, Vol, vel, v_rad, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, T, Den, Vol, vel, v_rad, Press, IE_den, Rad_den], mask)
    xyz = np.array([X, Y, Z]).T
    tree = KDTree(xyz, leaf_size = 50)

    data_ph = np.loadtxt(f'{pre_saving}/photo/{check}_photo{snap}.txt')
    # denph/= prel.den_converter #it was saved in cgs
    xph, yph, zph = data_ph[0], data_ph[1], data_ph[2]
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

    if plot:
        fig_all, ax_all = plt.subplots(1, len(indices_sorted), figsize = (len(indices_sorted)*5,6))
    indices_bigVol = []
    indices_overRph = []
    for i in range(len(observers_xyz)):
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
        # discarded_idx = initial_r_indices[~check_dist]
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
        ray_vr = v_rad[idx]
        ray_P = Press[idx]
        ray_ieDen = IE_den[idx]
        ray_radDen = Rad_den[idx]
        # ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_idx_sim, ray_r = \
        #     sort_list([ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_idx_sim, ray_r], ray_r)
        # idx = np.array(idx)

        # Interpolate ----------------------------------------------------------
        alpha_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(ray_t), np.log(ray_d), 'linear', 0)
        alpha_rossland = np.array(alpha_rossland)[0]
        underflow_mask = alpha_rossland != 0.0
        ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland, ray_P, ray_ieDen, ray_radDen, idx], underflow_mask)
        alpha_rossland_eval = np.exp(alpha_rossland) # [1/cm]
        del alpha_rossland
        gc.collect()

        # Optical Depth
        # compute the optical depth from outside in: tau = - int alpha dr. Then reverse the order to have it from the inside to out, so can query.
        ray_fuT = np.flipud(ray_r)
        alpha_rossland_fuT = np.flipud(alpha_rossland_eval) 
        tau = - np.flipud(cumulative_trapezoid(alpha_rossland_fuT, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_r. 
        tau_zero = tau != 0
        ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx], tau_zero)
        c_tau = prel.csol_cgs/tau # code units, since tau is adimensional
        ray_kappa = alpha_rossland_eval/ray_d

        if plot:
            _, theta, phi = to_spherical_coordinate(mu_x, mu_y, mu_z)
            phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
            phi = -phi
            tdyn_single = ray_r / ray_vr * prel.tsol_cgs # cgs
            tdiff_single = tau * ray_r * prel.Rsol_cgs / prel.c_cgs # cgs                

            fig, ax1 = plt.subplots(1,1,figsize = (8,6))
            ax1.plot(ray_r/Rt, tdyn_single/tfallback_cgs, '.', c = 'k', label = r'$t_{\rm dyn}=r/v_r$')
            # add a twin y axis to show ray_vr 
            # ax2 = ax1.twinx()
            # ax2.plot(ray_r/Rt, ray_vr, c = 'r')
            # ax2.set_ylabel(r'$v_R$ [cm/s]', fontsize = 20, c = 'r')
            # ax2.tick_params(axis='y', labelcolor='r')
            # ax2.set_yscale('log')                
            img = ax1.scatter(ray_r/Rt, tdiff_single/tfallback_cgs, c = tau, cmap = 'turbo', s = 10, label = r'$t_{\rm diff}=\tau r/c$' , norm = colors.LogNorm(5e-1, 1e2)) #np.percentile(tau, 5), np.percentile(tau, 95)))
            cbar = plt.colorbar(img)#, orientation = 'horizontal')
            cbar.set_label(r'$\tau$', fontsize = 20)
            cbar.ax.tick_params(which = 'major', length=6, width=1)
            cbar.ax.tick_params(which = 'minor', length=4, width=0.8)
            ax1.axvline(Rt/Rt, c = 'k', linestyle = '-.', label = r'$r_{\rm t}$')
            ax1.set_xlabel(r'$r [r_{\rm t}]$')
            ax1.set_ylabel(r'$t [t_{\rm fb}]$')
            ax1.loglog()    
            ax1.set_xlim(R0/Rt, 2*rph[i]/Rt)
            # ax1.axvline(rph[i]/Rt, c = 'k', linestyle = 'dotted', label =  r'$r_{\rm ph}$')
            # ax1.set_xlim(1e-5, 8)
            ax1.set_ylim(1e-5, 5)
            ax1.tick_params(axis='both', which='major', length=8, width=1.2)
            ax1.tick_params(axis='both', which='minor', length=5, width=1)
            ax1.legend(fontsize = 14)
            # plt.suptitle(f'Snap {snap}, observer {label_obs[count_i]} (number {i})', fontsize = 16)
            plt.suptitle(f'Section: {label_obs[j]}, ' + r'$(\theta, \phi)$ = ' + f'({theta:.2f}, {phi:.2f})', fontsize = 16) #phi according to pur convention (apocenter at -pi, clockwise), \theta from Npole to Spole 
            plt.tight_layout()

            ax_all[j].plot(ray_r/Rt, ray_kappa)
        
        # select the inner part, where tau big --> c/tau < v (i.e. tdyn<tdiff)
        Rtr_idx_all = np.where(c_tau/ray_vr <= 1)[0]
        if len(Rtr_idx_all) == 0:
            print(f'For obs {i}, tdiff < tdyn always, no Rtr', flush=True)
            # plt.close(fig)
            continue
        else: # take the one most outside 
            Rtr_idx = Rtr_idx_all[-1]+1 # so if you have a gap, it takes the next point

        # check you don't have a huge gap, otherwise it's just numerics: you don't really have 2 regimes
        if ray_vol[Rtr_idx+1]/ray_vol[Rtr_idx] > 1e3:
            indices_bigVol.append(i)
            print(f'For obs {i}, huge gap, so I skip, vol ratio: {int(ray_vol[Rtr_idx+1]/ray_vol[Rtr_idx])}', flush=True)

        if ray_r[Rtr_idx]/rph[i] >= 1:
            indices_overRph.append(i)
            print(f'For obs {i}, Rtr is outside Rph', flush=True)

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
            # alpha_rossland_eval_sol = alpha_rossland_eval[Rtr_idx] * prel.Rsol_cgs # to have it in 1/[sol length]
            # check_line = prel.csol_cgs/(den_tr[i]*kappa_tr[i]*Vr_tr[i]) 
            # check_line_alpha = prel.csol_cgs/(alpha_rossland_eval_sol*Vr_tr[i]) 
            # ax1.axvline(check_line/Rt, c = 'r', label =  r'$\frac{c}{\rho \kappa v_{\rm r}}$')
            # ax1.axvline(check_line_alpha/Rt, c = 'gold', ls = '--', label =  r'$\frac{c}{\alpha v_{\rm r}}$')
            ax1.axvline(r_tr[i]/Rt, c = 'k', linestyle = '--', label =  r'$r_{\rm tr}$')
            ax1.legend(fontsize = 14)
            # fig.savefig(f'{abspath}/Figs/{folder}/Wind/{choice}/{snap}/tdiff_{snap}Obs{i}.png')
            # plt.close(fig)
            
            # count_i += 1
            del ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx, ray_kappa

        if plot:
            # search in which list of indices_sorted, which is a list of lists, is i and call it j
            for j in range(len(indices_sorted)):
                if i in indices_sorted[j]:
                    break
            ax_all[j].set_xlabel(r'$r [r_{\rm t}]$')
            ax_all[j].loglog()
            ax_all[j].set_xlim(R0/Rt, 2*apo/Rt)
            ax_all[j].set_ylim(1e-1, 2e2)
            ax_all[j].set_title(f'Observers section: {label_obs[j]}')

    if plot:  
        ax_all[0].set_ylabel(r'$\kappa$ [cm$^2$/g]') 
        fig_all.tight_layout()
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
    }
    del X, Y, Z, T, Den, Vol, vel, v_rad, Press, IE_den, Rad_den
    gc.collect()

    return r_trapp

##
# MAIN
## 
#%% matlab
if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True)
else:
    snaps = [109]

for snap in snaps: 
    if compute:
        eng = matlab.engine.start_matlab()
        if alice:
            loadpath = f'{pre}/snap_{snap}'
            print(snap, flush=True)
        else: 
            choice = 'left_right_in_out_z'
            loadpath = f'{pre}/{snap}'
            observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
            indices_sorted, label_obs, colors_obs, _ = choose_observers(observers_xyz, choice)
            # test_idx = indices_sorted[1]
            # take just the first one for each direction
            # label_obs = np.array(label_obs)
            # colors_obs = np.array(colors_obs)
            # test_idx = np.array(test_idx)
            # label_obs, colors_obs, test_idx = sort_list([label_obs, colors_obs, test_idx], test_idx)
            r_tr = r_trapp(loadpath, snap, [Rt, 5000])

        np.savez(f"{pre_saving}/wind/trap/{check}_Rtr{snap}_test.npz", **r_tr)
        # if alice:
        eng.exit()

    if plot:
        almost = [109]
        dataRtr = np.load(f"{abspath}/data/{folder}/wind/trap/{check}_Rtr{snap}_test.npz")
        x_tr, y_tr, z_tr , den_tr, Vr_tr, kappa_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['Vr_tr'], dataRtr['kappa_tr']
        radius_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        Mdot_w = 4 * np.pi * radius_tr**2 * Vr_tr * den_tr
        Mdot_Edd_k = 4 * np.pi * prel.G * Mbh / kappa_tr
        # Mdot_w_cgs = Mdot_w * prel.Msol_cgs/prel.tsol_cgs
        # t_dyn = radius_tr / np.abs(Vr_tr)
        fig, ax = plt.subplots(1, 1, figsize = (8, 8)) 
        # ax.scatter(np.arange(len(radius_tr)), t_dyn/t_fb_sol, s = 5, c = 'k')
        # ax.set_xlabel(r'Observer index')
        # ax.set_ylabel(r'$t_{\rm dyn}$ [t$_{\rm fb}$]')
        ax.scatter(radius_tr/Rg, Mdot_w/Mdot_Edd_k, s = 10, c = 'k')
        ax.scatter(radius_tr[almost]/Rg, (Mdot_w/Mdot_Edd_k)[almost], s = 20, c = 'r')
        ax.set_xlabel(r'$r_{\rm tr} / r_{\rm g}$')
        ax.set_ylabel(r'$\dot{M}_{\rm w} / \dot{M}_{\rm Edd}$')
        ax.set_xlim(1e2, 1e7)
        ax.set_ylim(1e2, 1e7)
        ax.loglog()
        ax.set_title(f'Snap {snap}')
        ax.grid()
        ax.tick_params(axis='both', which='minor', length=6, width=1)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        plt.tight_layout()
    

#%%
# if plot:
#     photo = np.loadtxt(f'{pre_saving}/photo/{check}_photo{snap}.txt')
#     xph, yph, zph = photo[0], photo[1], photo[2]
#     rph = np.sqrt(xph**2 + yph**2 + zph**2)
#     rph = rph[test_idx]

#     dataRtrNOun = np.load(f"{pre_saving}/Rtrap_tests/{check}_Rtr{snap}_NOunique.npz")
#     x_tr_i_NOun, y_tr_i_NOun, z_tr_i_NOun = \
#         dataRtrNOun['x_tr'], dataRtrNOun['y_tr'], dataRtrNOun['z_tr']
#     R_tr_i_NOun = np.sqrt(x_tr_i_NOun**2 + y_tr_i_NOun**2 + z_tr_i_NOun**2)
#     R_tr_i_NOun = R_tr_i_NOun[test_idx]

#     dataRtr = np.load(f"{pre_saving}/Rtrap_tests/{check}_Rtr{snap}.npz")
#     x_tr_i, y_tr_i, z_tr_i, idx_tr = \
#         dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['idx_tr']
#     R_tr_i = np.sqrt(x_tr_i**2 + y_tr_i**2 + z_tr_i**2)
#     R_tr_i = R_tr_i[test_idx]

#     plt.figure(figsize = (8, 6))
#     plt.axvline(R_tr_i_NOun[4]/apo, label = r'$R_{\rm tr}$ without unique/sort', c = 'b')
#     plt.axvline(R_tr_i[4]/apo, label = r'$R_{\rm tr}$ with unique/sort', ls = '--', c = 'C1')
#     plt.axvline(rph[4]/apo, label = r'$R_{\rm ph}$', c = 'k')
#     plt.legend(fontsize = 16)
#     plt.xlim(0, 2.5)
#     plt.xlabel(r'$R [R_{\rm a}]$')
#     plt.title(f'Snap {snap}, observer {test_idx[1]}')


    # to check that the indices are correct are correct
    # idx_tr = np.array([int(idx_tr[i]) for i in range(len(idx_tr))])
    ## where_zero = np.where(idx_tr == 0)[0]
    # loadpath = f'{pre}/{snap}'
    # data = make_tree(loadpath, snap)
    # X, Y, Z, Den = data.X, data.Y, data.Z, data.Den
    # cut = Den > 1e-19   
    # X, Y, Z = make_slices([X, Y, Z], cut)
    # x_sim, y_sim, z_sim = X[idx_tr], Y[idx_tr], Z[idx_tr]

    # plt.scatter(np.arange(len(x_tr_i)), x_tr_i/x_sim, label = 'x')
    # plt.scatter(np.arange(len(y_tr_i)), y_tr_i/y_sim, label = 'y')
    # plt.scatter(np.arange(len(z_tr_i)), z_tr_i/z_sim, label = 'z')
    # plt.legend() 




