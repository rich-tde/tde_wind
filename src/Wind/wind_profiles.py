""" Find/plot radial profiles as weighted average on spherical sections. 
Find/plot polar profiles for fixed r and phi_array. 
Written to be run locally."""

import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import lines as mlines
import healpy as hp
from sklearn.neighbors import KDTree
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
import Utilities.operators as op

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

#%% FUNCTIONS
def radial_profiles(loadpath, snap, ray_params, which_obs, which_part = ''):
    rmin, rmax, Nray = ray_params
    r_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
    data = op.make_tree(loadpath, snap)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
    dim_cell = Vol**(1/3)

    # split in sections the wind cells
    Rsph_all = Rsph.copy()
    dim_cell_all = dim_cell.copy()
    sections = op.choose_sections(X, Y, Z, which_obs)
    indices_all = np.arange(len(Rsph_all))
    cond_sec_all = []
    for key in sections.keys():
        cond_sec_all.append(sections[key]['cond'])

    cut, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
    if which_part == 'outflow':
        cut = V_r >= 0 
    if which_part == 'all':
        cut = Den > 1e-19
    X, Y, Z, Rsph, Vol, Den, Mass, V_r, T, Press, IE_den, Rad_den, dim_cell = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, V_r, T, Press, IE_den, Rad_den, dim_cell], cut)       
    indices = np.arange(len(X))

    # split in sections the wind cells
    sections = op.choose_sections(X, Y, Z, which_obs)
    cond_sec = []
    colors_obs = []
    label_obs = []
    lines_obs = []
    for key in sections.keys():
        cond_sec.append(sections[key]['cond'])
        colors_obs.append(sections[key]['color'])
        label_obs.append(sections[key]['label'])
        lines_obs.append(sections[key]['line'])

    shell_indices = []
    shell_all_indices = []
    for i, r in enumerate(r_array): 
        # find cells at r
        ind_r = np.abs(Rsph-r) < dim_cell
        shell_indices.append(indices[ind_r])
        
        ind_r = np.abs(Rsph_all-r) < dim_cell_all
        shell_all_indices.append(indices_all[ind_r])
    # Convert to arrays for faster later indexing
    shell_indices = [np.asarray(s, dtype=int) for s in shell_indices]
    shell_all_indices = [np.asarray(s, dtype=int) for s in shell_all_indices]

    all_outflows = {}
    const_C = 4/len(cond_sec)
    for j, cond in enumerate(cond_sec):
        cond_all = cond_sec_all[j]
        t_prof = np.zeros(Nray)
        v_rad_prof = np.zeros(Nray)
        d_prof = np.zeros(Nray)
        Mdot_prof = np.zeros(Nray)
        L_kin_prof = np.zeros(Nray)
        L_adv_prof = np.zeros(Nray)
        Mdotmean_prof = np.zeros(Nray)
        L_kinmean_prof = np.zeros(Nray)
        L_advmean_prof = np.zeros(Nray)
        ratio_un = np.zeros(Nray)

        # Rsph_initial_j = Rsph_initial[j]
        # dim_cell_initial_j = dim_cell_initial[j]
        
        for i, r in enumerate(r_array): 
            # find cells in the shell at r
            shell = shell_indices[i]
            shell_all = shell_all_indices[i]
            if shell.size == 0:
                continue
            # restrict shell to section j
            mask = cond[shell]
            mask_all = cond_all[shell_all]
            if not np.any(mask):
                continue
            idx = shell[mask]
            len_all = len(shell_all[mask_all])

            ray_V_r = V_r[idx] 
            ray_d = Den[idx] 
            ray_m = Mass[idx]
            ray_rad_den = Rad_den[idx]
            ray_vol = Vol[idx]
            ray_dim = dim_cell[idx]
            ray_t = (ray_rad_den * prel.en_den_converter / prel.alpha_cgs)**(1/4) 
            L_adv =  ray_V_r * ray_rad_den
            t_prof[i] = np.sum(ray_t*ray_vol) / np.sum(ray_vol)
            v_rad_prof[i] = np.sum(ray_V_r*ray_m) / np.sum(ray_m)
            d_prof[i] = np.sum(ray_d*ray_m)/ np.sum(ray_m)
            Mdot_prof[i] = const_C * r**2 / np.sum(ray_dim**2) * np.pi * np.sum(ray_dim**2 * ray_d * ray_V_r)
            L_kin_prof[i] = const_C * r**2 / np.sum(ray_dim**2)* 0.5 * np.pi *np.sum(ray_dim**2 * ray_d * ray_V_r**3)
            L_adv_prof[i] = const_C * r**2 / np.sum(ray_dim**2) * np.pi * np.sum(ray_dim**2 * L_adv)
            ratio_un[i] = len(ray_d) / len_all if len_all > 0 else 0

            L_advmean_prof[i] =  const_C * np.pi * r**2 * np.mean(L_adv)
            Mdotmean_prof[i] = const_C * np.pi * r**2 * np.mean(ray_d * ray_V_r) if ray_V_r.size > 0 else 0 
            L_kinmean_prof[i] = const_C * np.pi * r**2 * np.mean(ray_d * ray_V_r**3) if ray_V_r.size > 0 else 0
       
        outflow = {
            'r': r_array,
            't_prof': t_prof,
            'v_rad_prof': v_rad_prof,
            'd_prof': d_prof,
            'Mdot_prof': Mdot_prof,
            'L_advmean_prof': L_advmean_prof,
            'Mdotmean_prof': Mdotmean_prof,
            'L_adv_prof': L_adv_prof,
            'L_kin_prof': L_kin_prof,
            'L_kinmean_prof': L_kinmean_prof,
            'ratio_un': ratio_un,
            'colors_obs': colors_obs[j],
            'lines_obs': lines_obs[j]
        }

        key = f"{label_obs[j]}"
        all_outflows[key] = outflow
    
    return all_outflows

def polar_profiles(loadpath, snap, ray_params, which_material = 'wind'):
    r_chosen, phis, Nray = ray_params
    theta_array = np.linspace(0, np.pi/2, Nray)
    data = op.make_tree(loadpath, snap)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)      
    dim_cell = Vol**(1/3)
    cut = np.logical_and(Den > 1e-19, np.abs(Rsph - r_chosen) < dim_cell)
    X, Y, Z, Rsph, Vol, dim_cell, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Rsph, Vol, dim_cell, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)
    len_allX_neg = len(X[X < 0])
    len_allX_pos = len(X[X >= 0])
    
    if which_material == 'wind':
        cut, bern, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params)
        X, Y, Z, Rsph, Vol, dim_cell, Den, Mass, V_r, T, Press, IE_den, Rad_den, bern = \
            make_slices([X, Y, Z, Rsph, Vol, dim_cell, Den, Mass, V_r, T, Press, IE_den, Rad_den, bern], cut)  
    else:
        V_r, _, _ = op.to_spherical_components(VX, VY, VZ, X, Y, Z)    
    _, lat, long = op.to_spherical_coordinate(X, Y, Z, r_frame = 'us') #lat in [0, pi] with North pole at 0, orbital plane at pi/2, long counterclockwise in [0, 2pi] with direction of positive x at 0 
    
    all_outflows = {}  
    if which_material == 'wind':
        unbound_ratio_Nhemi = np.zeros(Nray)
        unbound_ratio_Phemi = np.zeros(Nray)
    for j, phi in enumerate(phis):
        t_prof = np.zeros(Nray)
        v_rad_prof = np.zeros(Nray)
        d_prof = np.zeros(Nray)
        cut_phi = np.abs(long - phi) < 0.1

        for i, theta in enumerate(theta_array): 
            cut_theta = np.abs(lat - theta) < 0.1
            cut_angles = np.logical_and(cut_theta, cut_phi)
            if len(cut_angles) == 0:
                continue
            ray_V_r = V_r[cut_angles] 
            ray_d = Den[cut_angles] 
            ray_m = Mass[cut_angles]
            ray_rad_den = Rad_den[cut_angles]
            ray_vol = Vol[cut_angles]
            ray_t = (ray_rad_den * prel.en_den_converter / prel.alpha_cgs)**(1/4) 

            t_prof[i] = np.sum(ray_t*ray_vol) / np.sum(ray_vol)
            v_rad_prof[i] = np.sum(ray_V_r*ray_m) / np.sum(ray_m)
            d_prof[i] = np.sum(ray_d*ray_m)/ np.sum(ray_m)
            if np.logical_and(j==0, which_material == 'wind'): # so you do it just once, since there's no dependece on phi
                unbound_ratio_Nhemi[i] = len(X[np.logical_and(X<0, cut_theta)]) / len_allX_neg
                unbound_ratio_Phemi[i] = len(X[np.logical_and(X>=0, cut_theta)]) / len_allX_pos
        
        outflow = {
            'phi': phi,
            't_prof': t_prof,
            'v_rad_prof': v_rad_prof,
            'd_prof': d_prof,
        }

        key = f"{j}"
        all_outflows[key] = outflow
    
    all_outflows['theta_array'] = theta_array
    if which_material == 'wind':
        all_outflows['unbound_ratio_Nhemi'] = unbound_ratio_Nhemi 
        all_outflows['unbound_ratio_Phemi'] = unbound_ratio_Phemi 


    return all_outflows

#
## MAIN
#
compute = False
what = 'radial'
which_part = 'wind' # 'outflow' or 'all' or 'wind' to have the wind
snap = 151
idx_stop_d_unb = [7.2e1, 1.2e2] 

if what == 'polar':
    which_material = 'wind' # 'wind' or ''
    rchosen = apo
    rchose_lab = 'apo'
    if compute:
        path = f'{pre}/{snap}'
        ray_params = [rchosen, [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4], 50] 
        all_outflows = polar_profiles(path, snap, ray_params, which_material)
        out_path = f"{abspath}/data/{folder}/wind/theta_prof{snap}{which_material}_{rchose_lab}.npy"
        np.save(out_path, all_outflows, allow_pickle=True)
    
    else:
        from Utilities.basic_units import radians
        figd, (ax0, axd) = plt.subplots(1, 2, figsize=(18, 7)) 
        figVT, (axV, axT) = plt.subplots(1, 2, figsize=(18, 7)) 
        figU, axunb = plt.subplots(1, 1, figsize = (10, 6))
        x_line = np.arange(-40*Rt, 40*Rt, dtype=complex)
        line_xz = op.draw_line(x_line, np.arcsin(2/3), 'line')
        line_xz_neg = op.draw_line(x_line, np.pi-np.arcsin(2/3), 'line')
        # Load data and search for cell at cirularization radius = 2Rp
        path = f'{pre}/{snap}'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
        # data = op.make_tree(path, snap, energy = True)
        # if which_material == 'wind':
        #     X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        #         data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
        #     Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
        #     cut, _, _ = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params)
        #     cut = np.logical_and(cut, Den > 1e-19)
        # if which_material == '':
        #     X, Y, Z, Den, Vol  = data.X, data.Y, data.Z, data.Den, data.Vol
        #     cut = Den > 1e-19
            
        # X, Y, Z, Den, Vol = make_slices([X, Y, Z, Den, Vol], cut)
        # dim_cell = Vol**(1/3)
        # if which_material == 'wind':
        #     dmax_plot = 5*np.max(Den[np.abs(Y)<dim_cell]) * prel.den_converter    
        # if which_material == '':
        #     dmax_plot = 1e-6
        # xyz = np.array([X, Y, Z]).T
        # tree = KDTree(xyz, leaf_size = 50) 
        # Rc = 2*Rp
        # _, idx = tree.query(np.array([[Rc, 0, 0]])) 
        # idx = np.concatenate(idx)
        # norm = (Rc, Den[idx])   

        # plot xz plane
        # y_cut = np.abs(Y) < dim_cell
        # X_cut, Z_cut, Den_cut = X[y_cut], Z[y_cut], Den[y_cut]
        # img = axs.scatter(X_cut/Rt, Z_cut/Rt, c = Den_cut * prel.den_converter, norm = colors.LogNorm(vmin = 1e-15, vmax = dmax_plot), cmap = 'rainbow', s = 1)
        # cbar = figd.colorbar(img)
        # axs.plot(x_line, line_xz, c = 'k', ls = 'dashed')
        # axs.plot(x_line, line_xz_neg, c = 'k', ls = 'dashed')
        # cbar.set_label(r'$\rho$ (g/cm$^3)$')
        # axs.set_xlabel(r'x ($r_{\rm t}$)')
        # axs.set_ylabel(r'z ($r_{\rm t}$)')
        # axs.set_xlim(-apo/Rt, apo/Rt)
        # axs.set_ylim(-apo/Rt, apo/Rt)

        profiles = np.load(f'{abspath}/data/{folder}/wind/theta_prof{snap}{which_material}_{rchose_lab}.npy', allow_pickle=True).item()
        theta_plot = profiles['theta_array']
        if which_material == 'wind':
            unbound_ratio_Nhemi = profiles['unbound_ratio_Nhemi']
            unbound_ratio_Phemi = profiles['unbound_ratio_Phemi']
            axunb.plot(theta_plot * radians, unbound_ratio_Nhemi, linewidth = 2, label = r'$X<0$')
            axunb.plot(theta_plot * radians, unbound_ratio_Phemi, linewidth = 2, label = r'$X>0$')

        for key in profiles.keys():
            if key not in ['theta_array', 'unbound_ratio_Nhemi', 'unbound_ratio_Phemi']:
                phi = profiles[key]['phi']
                x_phi, y_phi = op.from_cylindric(phi, 1)
                d = profiles[key]['d_prof']
                v_rad = profiles[key]['v_rad_prof'] 
                t = profiles[key]['t_prof']

                ax0.scatter(x_phi, y_phi, s = 100, linewidth = 2, label = r'$\phi = $' + f'{phi:.2f} rad')
                axd.plot(theta_plot * radians, d * prel.den_converter, linewidth = 2, label = r'$\phi = $' + f'{phi:.2f} rad')
                axV.plot(theta_plot * radians, v_rad * conversion_sol_kms, linewidth = 2, label = r'$\phi = $' + f'{phi:.2f} rad')
                axT.plot(theta_plot * radians, t, linewidth = 2, label = r'$\phi = $' + f'{phi:.2f} rad')
        
        # rho_Cou = CouBegel(rchosen, theta_plot, 0, norm, gamma=4/3)
        # axd.plot(theta_plot * radians, rho_Cou * prel.den_converter, ls = ':', c = 'k', label = 'Coughlin+14')
        
        for ax in [ax0, axd, axV, axT, axunb]:
            ax.legend(fontsize = 16)
            ax.tick_params(axis='both', which='minor', length = 6, width = 1)
            ax.tick_params(axis='both', which='major', length = 10, width = 1.5)
            if ax != ax0:
                ax.set_xlabel(r'$\theta$')
                ax.set_yscale('log')
                ax.grid()

        axV.set_ylim(2e3, 1e5)
        axT.set_ylim(5e3, 5e5)
        axd.set_ylabel(r'$\rho$ [g/cm$^3]$')
        axV.set_ylabel(r'v$_{\rm r}$ [km/s]')
        axT.set_ylabel(r'$T_{\rm rad}$ [K]')
        axunb.set_ylabel(r'Ratio unbound materal')
        axunb.set_ylim(5e-3, 1)
        figd.suptitle(f'{which_material} at t = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 20)
        figd.tight_layout()
        figd.savefig(f'{abspath}/Figs/{folder}/Wind/polar_view/thetaD_prof{snap}{which_material}_{rchose_lab}.png', dpi = 300)
        figVT.suptitle(f'{which_material} at t = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 20)
        figVT.tight_layout()
        figVT.savefig(f'{abspath}/Figs/{folder}/Wind/polar_view/thetaVT_prof{snap}{which_material}_{rchose_lab}.png', dpi = 300)
        
        if which_material == 'wind':
            figU.suptitle(f'{which_material} at t = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 20)
            figU.tight_layout()
            figU.savefig(f'{abspath}/Figs/{folder}/Wind/polar_view/thetaUnb_prof{snap}{which_material}_{rchose_lab}.png', dpi = 300)


if what == 'radial':
    which_obs = 'left_right_z' # 'left_right_z', 'all' or 'in_out_z'

    if compute:
        path = f'{pre}/{snap}'
        ray_params = [Rt, 1e3*Rt, 300]
        all_outflows = radial_profiles(path, snap, ray_params, which_obs, which_part)
        out_path = f"{abspath}/data/{folder}/wind/rad_profSec{snap}_{which_obs}_{which_part}.npy"
        np.save(out_path, all_outflows, allow_pickle=True)

    else:
        which_parts = ['outflow', 'wind']
        labels_parts = ['Outflow', 'Wind']
        line_styles_parts = ['--', '-']

        x_test = np.arange(1., 300)
        y_testplus1 = op.draw_line(x_test, [3.5, 1], 'powerlaw')
        y_test1 = op.draw_line(x_test, [9e4, -1], 'powerlaw')
        y_test23 = op.draw_line(x_test, [3.5e5, -2/3], 'powerlaw')
        y_test2 = op.draw_line(x_test, [2e-7, -2], 'powerlaw')

        path = f'{pre}/{snap}'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
        # To have an idea of where is the trapping radius
        observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
        x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
        indices_obs, label_obs, colors_obs, _ = op.choose_observers(observers_xyz, which_obs)
        # Load data Rph and Rtr
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        xph, yph, zph = photo['x'], photo['y'], photo['z']
        rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
        dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
        x_tr, y_tr, z_tr, den_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr']
        r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        # sections_ph = op.choose_sections(xph, yph, zph, which_obs)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        # ax1.scatter(x_obs, y_obs, facecolor = 'none', edgecolors = 'k', linewidths = 1)
        # ax2.scatter(x_obs, z_obs, facecolor = 'none', edgecolors = 'k', linewidths = 1)
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Z')
        rph_medians = []
        rph_nonzero_medians = []
        rtr_medians = []
        rtr_nonzero_medians = []
        for i, idx_list in enumerate(indices_obs): 
            rph_medians.append(np.median(rph_all[idx_list]))
            rtr_medians.append(np.median(r_tr_all[idx_list]))
            non_zero = idx_list[r_tr_all[idx_list]> Rt]
        #     # if non_zero.any():
        #     print(f'{label_obs[i]}: Rtr in {len(non_zero)/len(idx_list)*100:.2f}%')
            rph_nonzero_medians.append(np.median(rph_all[non_zero]))
            rtr_nonzero_medians.append(np.median(r_tr_all[non_zero]))
        #     # Plot the observers with trapping radius non zero
        #     ax1.scatter(x_obs[non_zero], y_obs[non_zero], color = colors_obs[i], linewidths = 1)
        #     ax2.scatter(x_obs[non_zero], z_obs[non_zero], color = colors_obs[i], linewidths = 1, label = r'r$_{\rm tr}\neq0$' if i == 0 else '')
        # plt.tight_layout()
        
        
        fig, (axd, axV, axM, axLkin) = plt.subplots(4, 1, figsize=(8, 22)) 
        figM, (axT, axLadv) = plt.subplots(1, 2, figsize=(15, 7))
        figr, axratio = plt.subplots(1, 1, figsize=(12, 10))
        all_axes = [axd, axV, axT, axM, axLadv, axLkin, axratio]
        
        handles_color = []
        labels_color = []
        # Load profiles
        for k, which_part in enumerate(which_parts):
            profiles = np.load(f'{abspath}/data/{folder}/wind/rad_profSec{snap}_{which_obs}_{which_part}.npy', allow_pickle=True).item()
            for i, lab in enumerate(profiles.keys()):
                print(lab)
                if label_obs[i] == 'South pole':
                    continue 
                r_plot = profiles[lab]['r'] 
                d = profiles[lab]['d_prof']
                v_rad = profiles[lab]['v_rad_prof'] 
                t = profiles[lab]['t_prof']
                Mdot = profiles[lab]['Mdot_prof'] #Mdotmean_prof
                L_adv = profiles[lab]['L_adv_prof'] #L_advmean_prof
                L_kin = profiles[lab]['L_kin_prof'] #L_kinmean_prof
                ratio_un = profiles[lab]['ratio_un']
                colors_sec = profiles[lab]['colors_obs']
                # Mdot = d * r_plot**2 * v_rad
                not_zero = np.where(d != 0)
                r_plot, d, v_rad, t, Mdot, L_adv, L_kin, ratio_un = make_slices([r_plot, d, v_rad, t, Mdot, L_adv, L_kin, ratio_un], not_zero)
                idx_rtr = np.argmin(np.abs(r_plot - rtr_nonzero_medians[i]))
                idx_rph = np.argmin(np.abs(r_plot - rph_nonzero_medians[i]))
                if label_obs[i] == r'Stream side': # just to cut the initially unbound material
                    idx_stop_d = np.argmin(np.abs(r_plot - idx_stop_d_unb[k]*Rt)) 
                    d[idx_stop_d:] = 1e-20
                    Mdot[idx_stop_d:] = 1e-20
                    L_kin[idx_stop_d:] = 1e-20
                    ratio_un[np.argmin(np.abs(r_plot - idx_stop_d_unb[0]*Rt)):] = 1e-20
                else:
                    idx_stop_d = -1


                line = axd.plot(r_plot/Rt, d * prel.den_converter, label = f'{lab}' if which_part == 'wind' else None, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)[0]
                axV.plot(r_plot/Rt, v_rad * conversion_sol_kms, label = f'{labels_parts[k]}' if i == 2 else None, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                axM.plot(r_plot/Rt, Mdot/Medd_sol, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                axratio.plot(r_plot/Rt, ratio_un, label = f'{lab}' if which_part == 'wind' else None, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                axT.plot(r_plot/Rt, t, label = f'{lab}' if which_part == 'wind' else None, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                axLadv.plot(r_plot/Rt, L_adv/Ledd_sol, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                axLkin.plot(r_plot/Rt, L_kin/Ledd_sol, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)

                if which_part == 'wind':
                    handles_color.append(line)
                    labels_color.append(lab)
                    
                if which_part == 'wind':
                    axd.scatter(r_plot[idx_rph]/Rt, d[idx_rph] * prel.den_converter, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axM.scatter(r_plot[idx_rph]/Rt, Mdot[idx_rph]/Medd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axV.scatter(r_plot[idx_rph]/Rt, v_rad[idx_rph] * conversion_sol_kms, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axLadv.scatter(r_plot[idx_rph]/Rt, L_adv[idx_rph]/Ledd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axT.scatter(r_plot[idx_rph]/Rt, t[idx_rph], marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axLkin.scatter(r_plot[idx_rph]/Rt, L_kin[idx_rph]/Ledd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)

                    axd.scatter(r_plot[idx_rtr]/Rt, d[idx_rtr] * prel.den_converter, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axM.scatter(r_plot[idx_rtr]/Rt, Mdot[idx_rtr]/Medd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axV.scatter(r_plot[idx_rtr]/Rt, v_rad[idx_rtr] * conversion_sol_kms, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axLadv.scatter(r_plot[idx_rtr]/Rt, L_adv[idx_rtr]/Ledd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axT.scatter(r_plot[idx_rtr]/Rt, t[idx_rtr], marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)
                    axLkin.scatter(r_plot[idx_rtr]/Rt, L_kin[idx_rtr]/Ledd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k], linewidth = 2)


        axd.set_ylim(2e-13, 1e-5)
        axV.set_ylim(1.5e3, 1.5e4)
        axT.set_ylim(2e4, 1e6)
        axd.plot(x_test, y_test2, c = 'gray', ls = 'dotted', label = r'$\rho \propto r^{-2}$')
        axd.text(75, 2e-11, r'$\rho \propto r^{-2}$', fontsize = 18, color = 'gray', rotation = -20)
        axV.axhline(v_esc_kms, c = 'k', ls = 'dotted')# 
        # axV.text(35, 1.1*0.2*v_esc_kms, r'0.2v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
        axT.plot(x_test, y_test23, c = 'k', ls = 'dotted', label = r'$T \propto r^{-2/3}$')
        # axT.text(1.2, 2.4e5, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -24)
        axLadv.plot(x_test, 1e-5*y_test23, c = 'k', ls = 'dotted', label = r'$L \propto r^{-2/3}$')
        # axLadv.text(1.2, 5.6e1, r'$L \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -18)
        # axd.legend(fontsize = 18, loc = 'upper right')
        
        # Legend 1: colored observer lines (three colors)
        legend1 = axd.legend(handles=handles_color,
                            labels=labels_color,
                            fontsize=16,
                            loc='upper right')
        axd.add_artist(legend1)

        # Legend 2: line-style explanation (solid vs dashed)
        solid_proxy = mlines.Line2D([0], [0], color='cornflowerblue', ls='-', linewidth=2,
                                    label='Unbound outflow (wind)')
        dashed_proxy = mlines.Line2D([0], [0], color='cornflowerblue', ls='--', linewidth=2,
                                    label='Unbound + bound outflow')

        legend2 = axd.legend(handles=[solid_proxy, dashed_proxy],
                            fontsize=16,
                            loc='lower left')

        axM.set_ylim(1e2, 1e7)
        axLadv.set_ylim(5e-2, 1e2)
        axLkin.set_ylim(1e-1, 5e2) 
        axratio.set_ylim(5e-2, 1.1)
        axT.legend(fontsize = 18)
        axd.set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 28)
        axV.set_ylabel(r'v$_{\rm r}$ (km/s)', fontsize = 28)
        axT.set_ylabel(r'$T_{\rm rad}$ (K)', fontsize = 28)
        axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 28)
        axLadv.set_ylabel(r'$L_{\rm adv} (L_{\rm Edd})$', fontsize = 28)
        axLkin.set_ylabel(r'$L_{\rm kin} (L_{\rm Edd})$', fontsize = 28)
        axratio.set_ylabel(r'f$_{\rm unb}$', fontsize = 28)
        axratio.legend(fontsize = 18)

        for ax in all_axes: 
            ax.tick_params(axis='both', which='minor', length = 8, width = 1)
            ax.tick_params(axis='both', which='major', length = 15, width = 1.5)
            ax.loglog()
            ax.set_xlim(1.5, 1.4e2)
            ax.grid()
            ax.axvline(apo/Rt, color = 'k', ls = 'dotted')
               
        axd.text(0.8*apo/Rt, 0.2*axd.get_ylim()[1], r'$r_{\rm a}$', fontsize = 20, color = 'k', rotation = 90)
        axLkin.set_xlabel(r'$r /r_{\rm t}$', fontsize = 28)
        # fig.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 30)
        figM.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 30)
        fig.tight_layout()
        figM.tight_layout()
        fig.savefig(f'{abspath}/Figs/2.paperWind/den_prof_{snap}.pdf', bbox_inches = 'tight')
        figM.savefig(f'{abspath}/Figs/2.paperWind/LT_{snap}.pdf', bbox_inches = 'tight')

# %%
