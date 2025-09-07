"""Compute trapping radius i.e. R: tau(R) = c/v(R) and diffusion and dynamical time in the radial direction"""
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    import k3match
    abspath = '/Users/paolamartire/shocks'
    save = False

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
from Utilities.operators import make_tree, sort_list, to_spherical_components, choose_observers
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
check = 'NewAMR' 
which_part = 'outflow' # 'outflow' or ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
pre_saving = f'{abspath}/data/{folder}'

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
norm = things['E_mb']
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

# Opacity
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
_, _, scatter2 = opacity_linear(T_cool, Rho_cool, scattering, slope_length = 7, highT_slope = 0)
T_cool2, Rho_cool2, rossland2 = opacity_extrap(T_cool, Rho_cool, rossland, scatter = scatter2, slope_length = 7, highT_slope = 0)

def r_trapp(loadpath, snap):
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) #shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)

    data = make_tree(loadpath, snap, energy = True)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE, data.Rad
    v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)
    if which_part == 'outflow':
        R = np.sqrt(X**2 + Y**2 + Z**2)
        mass = Den * Vol
        bern = orb.bern_coeff(R, vel, Den, mass, Press, IE_den, Rad_den, params)
        mask = np.logical_and(Den > 1e-19, np.logical_and(bern > 0, v_rad >= 0)) 
    elif which_part == '': 
        mask = Den > 1e-19
    X, Y, Z, T, Den, Vol, vel, v_rad, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, T, Den, Vol, vel, v_rad, Press, IE_den, Rad_den], mask)
    xyz = np.array([X, Y, Z]).T
    tree = KDTree(xyz, leaf_size=50)
    N_ray = 5000

    xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph, Pressph, IE_denph, _, _, _, _ = \
        np.loadtxt(f'{pre_saving}/photo/{check}_photo{snap}.txt')
    denph/= prel.den_converter #it was saved in cgs
    rph = np.sqrt(xph**2 + yph**2 + zph**2)

    x_tr = np.zeros(len(observers_xyz))
    y_tr = np.zeros(len(observers_xyz))
    z_tr = np.zeros(len(observers_xyz))
    vol_tr = np.zeros(len(observers_xyz))
    den_tr = np.zeros(len(observers_xyz))
    Temp_tr = np.zeros(len(observers_xyz))
    Vr_tr = np.zeros(len(observers_xyz))
    V_tr = np.zeros(len(observers_xyz))
    P_tr = np.zeros(len(observers_xyz))
    IEden_tr = np.zeros(len(observers_xyz))
    Rad_den_tr = np.zeros(len(observers_xyz))

    count_i = 0
    for i in range(len(observers_xyz)):
        if not alice:
            if i not in test_idx:
                continue
            else:
                print(i, flush=True)
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
        r = np.logspace(-0.25, np.log10(rmax), N_ray)
        x = r*mu_x
        y = r*mu_y
        z = r*mu_z

        xyz2 = np.array([x, y, z]).T
        # radii2 = np.sqrt(x**2 + y**2 + z**2)
        del x, y, z

        dist, idx = tree.query(xyz2, k=1)
        dist = np.concatenate(dist)
        idx = np.array([ int(idx[i][0]) for i in range(len(idx))])

        # pick them just if near enough and iterate
        # r_sim = np.sqrt(X[idx]**2 + Y[idx]**2 + Z[idx]**2)
        # check_dist = np.abs(r_sim - radii2) < Vol[idx]**(1/3)
        check_dist = dist <= Vol[idx]**(1/3)
        idx = idx[check_dist]
        ray_r = r[check_dist]
 
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
        idx = np.array(idx)

        if len(idx) == 0:
            print(f'No points found for observer {i}', flush=True)
            count_i += 1
            continue

        # check which points your are taking
        if plot:
            plt.figure()
            x_plot = xyz2[:,0]
            y_plot = xyz2[:,1]
            z_plot = xyz2[:,2]
            plt.plot(x_plot[check_dist]/apo, y_plot[check_dist]/apo, c = 'k')
            img = plt.scatter(ray_x/apo, ray_y/apo, c = np.abs(z_plot[check_dist]/ray_z), s = 8, cmap = 'rainbow', norm = colors.LogNorm(vmin = 5e-1, vmax = 5))
            plt.colorbar(img, label = r'$|z_{\rm wanted}/z_{\rm sim}|$')
            plt.xlim(-8, 8)
            plt.ylim(-8, 8)
            plt.xlabel(r'$x/R_{\rm a}$')
            plt.ylabel(r'$y/R_{\rm a}$')
            plt.title(label_obs[count_i])

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
        ray_fuT = np.flipud(ray_r)
        alpha_rossland_fuT = np.flipud(alpha_rossland_eval) 
        # compute the optical depth from outside in: tau = - int alpha dr. Then reverse the order to have it from the inside to out, so can query.
        tau = - np.flipud(cumulative_trapezoid(alpha_rossland_fuT, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_r. 
        tau_zero = tau != 0
        ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, ray_t, ray_d, ray_vol, ray_vr, ray_V, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx], tau_zero)
        c_tau = prel.csol_cgs/tau # code units, since tau is adimensional

        if plot:
            tdyn_single = ray_r / np.abs(ray_vr) * prel.tsol_cgs # cgs
            tdiff_single = tau * ray_r * prel.Rsol_cgs / prel.c_cgs # cgs

            fig, ax1 = plt.subplots(1,1,figsize = (8,6))
            ax1.plot(ray_r/apo, tdyn_single/tfallback_cgs, c = 'k', label = r'$t_{\rm dyn}=R/v_R$')
            # add a twin y axis to show ray_vr 
            # ax2 = ax1.twinx()
            # ax2.plot(ray_r/apo, ray_vr, c = 'r')
            # ax2.set_ylabel(r'$v_R$ [cm/s]', fontsize = 20, c = 'r')
            # ax2.tick_params(axis='y', labelcolor='r')
            # ax2.set_yscale('log')                
            img = ax1.scatter(ray_r/apo, tdiff_single/tfallback_cgs, c = tau, cmap = 'turbo', s = 10, norm = colors.LogNorm(5e-1, 1e2)) #np.percentile(tau, 5), np.percentile(tau, 95)))
            cbar = plt.colorbar(img)#, orientation = 'horizontal')
            cbar.set_label(r'$\tau$', fontsize = 20)
            ax1.axvline(Rt/apo, c = 'k', linestyle = '-.', label = r'$R_{\rm t}$')
            # ax1.axvline(rph[i]/apo, c = 'k', linestyle = 'dotted', label =  r'$R_{\rm ph}$')
            ax1.set_xlabel(r'$R [R_{\rm a}]$')
            ax1.set_ylabel(r'$t [t_{\rm fb}]$')
            ax1.loglog()    
            ax1.set_xlim(R0/apo, 5)
            # ax1.set_ylim(1e-4, 20)
            ax1.tick_params(axis='both', which='major', length=8, width=1.2)
            ax1.tick_params(axis='both', which='minor', length=5, width=1)
            ax1.legend(fontsize = 14)
            plt.suptitle(f'Snap {snap}, observer {label_obs[count_i]} (number {i}) from R0 to its photosphere', fontsize = 16)
            plt.tight_layout()
        
        # select the inner part, where tau big --> c/tau < v (i.e. tdyn<tdiff)
        Rtr_idx_all = np.where(c_tau/np.abs(ray_vr)<1)[0]
        if len(Rtr_idx_all) == 0:
            # print(f'No Rtr found anywhere for obs {i}', flush=False)
            Rtr_idx = 0
            print(f'For obs {i}, tdiff < tdyn always, so I pick the first wind cell')
        # take the one most outside 
        else:
            Rtr_idx = Rtr_idx_all[-1] +1 # so if you have a gap, it takes the next point

        if ray_r[Rtr_idx]/rph[i] > 1:
            v_rad_ph, _, _ = to_spherical_components(Vxph[i], Vyph[i], Vzph[i], xph[i], yph[i], zph[i])
            V_ph = np.sqrt(Vxph[i]**2 + Vyph[i]**2 + Vzph[i]**2)
            mass_ph = denph[i] * volph[i] 
            bern_ph = orb.bern_coeff(rph[i], V_ph, denph[i], mass_ph, Pressph[i], IE_denph[i], Rad_denph[i], params)
            if np.logical_and(bern_ph > 0, v_rad_ph >= 0):
                x_tr[i], y_tr[i], z_tr[i], vol_tr[i], den_tr[i], Temp_tr[i], Vr_tr[i], V_tr[i], P_tr[i], IEden_tr[i], Rad_den_tr[i] = \
                    xph[i], yph[i], zph[i], volph[i], denph[i], Tempph[i], v_rad_ph, V_ph, Pressph[i], IE_denph[i], Rad_denph[i]
                print(f'For obs {i}, big gap in time comparison. Rtr is outside Rph, so I take it')
                if plot:
                    ax1.axvline(rph[i]/apo, c = 'k', linestyle = '--', label =  r'$R_{\rm tr}$')
                    ax1.legend(fontsize = 14)
                    plt.savefig(f'{abspath}/Figs/next_meeting/tdiff_{which_part}{snap}{label_obs[count_i]}.png')
            else:
                print(f'For obs {i}, big gap in time comparison. Rtr is outside Rph and Rph is not outflowing. I skip.')
                count_i += 1
                if plot:
                    ax1.legend(fontsize = 14)
                    plt.savefig(f'{abspath}/Figs/next_meeting/tdiff_{which_part}{snap}{label_obs[count_i]}.png')
                continue
        else:
            x_tr[i] = ray_x[Rtr_idx]
            y_tr[i] = ray_y[Rtr_idx]
            z_tr[i] = ray_z[Rtr_idx]
            vol_tr[i] = ray_vol[Rtr_idx]
            den_tr[i] = ray_d[Rtr_idx]/prel.den_converter # so is in code units
            Temp_tr[i] = ray_t[Rtr_idx]
            Vr_tr[i] = ray_vr[Rtr_idx]
            V_tr[i] = ray_V[Rtr_idx]
            P_tr[i] = ray_P[Rtr_idx]
            IEden_tr[i] = ray_ieDen[Rtr_idx]
            Rad_den_tr[i] = ray_radDen[Rtr_idx]
            if plot:
                print(label_obs[count_i], ray_r[Rtr_idx]/rph[i])
                ax1.axvline(ray_r[Rtr_idx]/apo, c = 'k', linestyle = '--', label =  r'$R_{\rm tr}$')
                ax1.axvline(rph[i]/apo, c = 'k', linestyle = 'dotted', label =  r'$R_{\rm ph}$')
                ax1.legend(fontsize = 14)
                plt.savefig(f'{abspath}/Figs/next_meeting/tdiff_{which_part}{snap}{label_obs[count_i]}.png')
        
        count_i += 1
 
    r_trapp = {
        'x_tr': x_tr,
        'y_tr': y_tr,
        'z_tr': z_tr,
        'vol_tr': vol_tr,
        'den_tr': den_tr,
        'Temp_tr': Temp_tr,
        'Vr_tr': Vr_tr,
        'V': V_tr,
        'P_tr': P_tr,
        'IE_den_tr': IEden_tr,
        'Rad_den_tr': Rad_den_tr,
    }

    return r_trapp

##
# MAIN
## 
#%% matlab
eng = matlab.engine.start_matlab()
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True)

for snap in snaps:
    if alice:
        loadpath = f'{pre}/snap_{snap}'
        print(snap, flush=True)
    else: 
        if snap != 318:
            continue
        loadpath = f'{pre}/{snap}'
        observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
        indices_sorted, label_obs, colors_obs, _ = choose_observers(observers_xyz, 'hemispheres')
        test_idx = indices_sorted[:,0]
        # take just the first one for each direction
        label_obs = np.array(label_obs)
        colors_obs = np.array(colors_obs)
        test_idx = np.array(test_idx)
        label_obs, colors_obs, test_idx = sort_list([label_obs, colors_obs, test_idx], test_idx)
    r_tr = r_trapp(loadpath, snap)

    if save:
        np.savez(f"{pre_saving}/trap/{check}_Rtr{snap}.npz", **r_tr)

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


#%%
eng.exit()

