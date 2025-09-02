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

# Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'NewAMR' 
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
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, T, Den, Vol, VX, VY, VZ, Press, IE_den, Rad_den], denmask)
    idx_sim = np.arange(len(X))
    xyz = np.array([X, Y, Z]).T
    N_ray = 5000

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
    idx_tr = np.zeros(len(observers_xyz))

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
        del x, y, z

        tree = KDTree(xyz, leaf_size=50)
        _, idx = tree.query(xyz2, k=1)
        idx = np.array([ int(idx[i][0]) for i in range(len(idx))])
        # idx = np.unique(idx)
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        t = T[idx]
        d = Den[idx] * prel.den_converter
        ray_vol = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_idx_sim = idx_sim[idx]
        ray_r = r #np.sqrt(ray_x**2 + ray_y**2 + ray_z**2) 
        ray_P = Press[idx]
        ray_ieDen = IE_den[idx]
        ray_radDen = Rad_den[idx]
        # ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_idx_sim, ray_r = \
        #     sort_list([ray_x, ray_y, ray_z, t, d, ray_vol, ray_vx, ray_vy, ray_vz, idx, ray_idx_sim, ray_r], ray_r)
        idx = np.array(idx)
        v_rad, _, _ = to_spherical_components(ray_vx, ray_vy, ray_vz, ray_x, ray_y, ray_z)
        vel = np.sqrt(ray_vx**2 + ray_vy**2 + ray_vz**2)

        # Interpolate ----------------------------------------------------------
        alpha_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
        alpha_rossland = np.array(alpha_rossland)[0]
        underflow_mask = alpha_rossland != 0.0
        ray_x, ray_y, ray_z, ray_r, t, d, ray_vol, v_rad, vel, ray_idx_sim, alpha_rossland, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, t, d, ray_vol, v_rad, vel, ray_idx_sim, alpha_rossland, ray_P, ray_ieDen, ray_radDen, idx], underflow_mask)
        alpha_rossland_eval = np.exp(alpha_rossland) # [1/cm]
        del alpha_rossland
        gc.collect()

        # Optical Depth
        ray_fuT = np.flipud(ray_r)
        alpha_rossland_fuT = np.flipud(alpha_rossland_eval) 
        # compute the optical depth from outside in: tau = - int alpha dr. Then reverse the order to have it from the inside to out, so can query.
        tau = - np.flipud(cumulative_trapezoid(alpha_rossland_fuT, ray_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for ray_r. 
        tau_zero = tau != 0
        ray_x, ray_y, ray_z, ray_r, t, d, ray_vol, v_rad, vel, ray_idx_sim, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx = \
            make_slices([ray_x, ray_y, ray_z, ray_r, t, d, ray_vol, v_rad, vel, ray_idx_sim, alpha_rossland_eval, tau, ray_P, ray_ieDen, ray_radDen, idx], tau_zero)
        c_tau = prel.csol_cgs/tau # code units, since tau is adimensional

        if plot:
            photo = np.loadtxt(f'{pre_saving}/photo/{check}_photo{snap}.txt')
            xph, yph, zph = photo[0], photo[1], photo[2]
            rph = np.sqrt(xph**2 + yph**2 + zph**2)

            tdyn_single = ray_r / np.abs(v_rad) * prel.tsol_cgs # cgs
            tdiff_single = tau * ray_r * prel.Rsol_cgs / prel.c_cgs # cgs

            fig, ax1 = plt.subplots(1,1,figsize = (8,6))
            ax1.plot(ray_r/apo, tdyn_single/tfallback_cgs, c = 'k', label = r'$t_{\rm dyn}=R/v_R$')
            # add a twin y axis to show v_rad 
            # ax2 = ax1.twinx()
            # ax2.plot(ray_r/apo, v_rad, c = 'r')
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
            ax1.set_xlim(R0/apo, rph[i]/apo)
            ax1.set_ylim(1e-3, 1e-1)
            ax1.tick_params(axis='both', which='major', length=8, width=1.2)
            ax1.tick_params(axis='both', which='minor', length=5, width=1)
            ax1.legend(fontsize = 14)
            plt.suptitle(f'Snap {snap}, observer {i} from R0 to its photosphere', fontsize = 16)
            plt.tight_layout()
        
        # select the inner part, where tau big --> c/tau < v
        Rtr_idx_all = np.where(c_tau/np.abs(v_rad)<1)[0]
        if len(Rtr_idx_all) == 0:
            print(f'No Rtr found anywhere for obs {i}', flush=False)
            sys.stdout.flush()
            continue
        # take the one most outside 
        Rtr_idx = Rtr_idx_all[-1]
        x_tr[i] = ray_x[Rtr_idx]
        y_tr[i] = ray_y[Rtr_idx]
        z_tr[i] = ray_z[Rtr_idx]
        vol_tr[i] = ray_vol[Rtr_idx]
        den_tr[i] = d[Rtr_idx]/prel.den_converter
        Temp_tr[i] = t[Rtr_idx]
        Vr_tr[i] = v_rad[Rtr_idx]
        V_tr[i] = vel[Rtr_idx]
        P_tr[i] = ray_P[Rtr_idx]
        IEden_tr[i] = ray_ieDen[Rtr_idx]
        Rad_den_tr[i] = ray_radDen[Rtr_idx]
        idx_tr[i] = ray_idx_sim[Rtr_idx]
        
        if plot:
            ax1.axvline(ray_r[Rtr_idx]/apo, c = 'k', linestyle = '--', label =  r'$R_{\rm tr}$')
            ax1.legend(fontsize = 14)

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
        'idx_tr': idx_tr,
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
        if snap != 238:
            continue
        loadpath = f'{pre}/{snap}'
        observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
        indices_sorted, label_obs, colors_obs = choose_observers(observers_xyz, 'arch')
        test_idx = indices_sorted[:,0]

    r_tr = r_trapp(loadpath, snap)
    # np.savez(f"{pre_saving}/Rtrap_tests/{check}_Rtr{snap}_NOunique.npz", **r_tr)

    if save:
        np.savez(f"{pre_saving}/trap/{check}_Rtr{snap}.npz", **r_tr)

#%%
if plot:
    photo = np.loadtxt(f'{pre_saving}/photo/{check}_photo{snap}.txt')
    xph, yph, zph = photo[0], photo[1], photo[2]
    rph = np.sqrt(xph**2 + yph**2 + zph**2)
    rph = rph[test_idx]

    dataRtrNOun = np.load(f"{pre_saving}/Rtrap_tests/{check}_Rtr{snap}_NOunique.npz")
    x_tr_i_NOun, y_tr_i_NOun, z_tr_i_NOun = \
        dataRtrNOun['x_tr'], dataRtrNOun['y_tr'], dataRtrNOun['z_tr']
    R_tr_i_NOun = np.sqrt(x_tr_i_NOun**2 + y_tr_i_NOun**2 + z_tr_i_NOun**2)
    R_tr_i_NOun = R_tr_i_NOun[test_idx]

    dataRtr = np.load(f"{pre_saving}/Rtrap_tests/{check}_Rtr{snap}.npz")
    x_tr_i, y_tr_i, z_tr_i, idx_tr = \
        dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['idx_tr']
    R_tr_i = np.sqrt(x_tr_i**2 + y_tr_i**2 + z_tr_i**2)
    R_tr_i = R_tr_i[test_idx]

    plt.figure(figsize = (8, 6))
    plt.axvline(R_tr_i_NOun[4]/apo, label = r'$R_{\rm tr}$ without unique/sort', c = 'b')
    plt.axvline(R_tr_i[4]/apo, label = r'$R_{\rm tr}$ with unique/sort', ls = '--', c = 'C1')
    plt.axvline(rph[4]/apo, label = r'$R_{\rm ph}$', c = 'k')
    plt.legend(fontsize = 16)
    plt.xlim(0, 2.5)
    plt.xlabel(r'$R [R_{\rm a}]$')
    plt.title(f'Snap {snap}, observer {test_idx[1]}')


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

