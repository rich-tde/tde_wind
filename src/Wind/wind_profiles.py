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
def profiles(loadpath, snap, ray_params, which_obs, which_part = '', what_varies = 'r'):
    if what_varies == 'r':
        rmin, rmax, Nray = ray_params
        ray_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
    elif what_varies == 'theta':
        r_fixed, Nray = ray_params
        ray_array = np.linspace(0, np.pi, Nray)
        delta_theta = ray_array[1] - ray_array[0]
    data = op.make_tree(loadpath, snap)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
    dim_cell = Vol**(1/3)
    indices_all = np.arange(len(X))

    # split in sections the wind cells
    if what_varies == 'r':
        Rsph_all = Rsph.copy()
    elif what_varies == 'theta':
        _, lat_all, _ = op.to_spherical_coordinate(X, Y, Z, r_frame = 'us') 

    dim_cell_all = dim_cell.copy()
    sections = op.choose_sections(X, Y, Z, which_obs)
    cond_sec_all = []
    for key in sections.keys():
        cond_sec_all.append(sections[key]['cond'])

    cut_wind, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
    if which_part == 'wind':
        cut = cut_wind
    elif which_part == 'outflow':
        cut = V_r >= 0 
    elif which_part == 'all':
        cut = Den > 1e-19
    if what_varies == 'theta':
        cut = np.logical_and(cut, np.abs(Rsph - r_fixed) < dim_cell)
    X, Y, Z, Rsph, Vol, Den, Mass, V_r, T, Press, IE_den, Rad_den, dim_cell = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, V_r, T, Press, IE_den, Rad_den, dim_cell], cut)       
    indices = np.arange(len(X))

    if what_varies == 'theta':
        _, lat, _ = op.to_spherical_coordinate(X, Y, Z, r_frame = 'us') #lat in [0, pi] with North pole at 0, orbital plane at pi/2, long counterclockwise in [0, 2pi] with direction of positive x at 0 

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
    for i, r in enumerate(ray_array): 
        # find cells at r
        if what_varies == 'r':
            ind_r = np.abs(Rsph-r) < dim_cell
            ind_r_all = np.abs(Rsph_all-r) < dim_cell_all 
        elif what_varies == 'theta':
            ind_r = np.abs(lat-r) < delta_theta 
            ind_r_all = np.abs(lat_all-r) < delta_theta
        shell_indices.append(indices[ind_r])
        shell_all_indices.append(indices_all[ind_r_all])

    # Convert to arrays for faster later indexing
    shell_indices = [np.asarray(s, dtype=int) for s in shell_indices]
    shell_all_indices = [np.asarray(s, dtype=int) for s in shell_all_indices]

    all_outflows = {}
    figtest, (axtest, axtest_sec) = plt.subplots(1,2,figsize=(16,8))
    coltest = plt.cm.get_cmap('rainbow', Nray)
    axtest.set_xlabel(r'X/$r_{\rm t}$')
    axtest.set_ylabel(r'Z/$r_{\rm t}$')
    for j, cond in enumerate(cond_sec):
        if what_varies == 'theta': # should be 2\pi r*l where l is the arch in the theta dir, which is l = rdtheta. \pi goes away dividing, so 2r^2dtheta
            const_C = 2 * r_fixed**2 * delta_theta / len(cond_sec)
            min_lat = np.min(lat[cond_sec[j]])
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
        
        for i, r in enumerate(ray_array): 
            if what_varies == 'r':
                const_C = 4*r**2/len(cond_sec)
            if np.logical_and(what_varies == 'theta', r < min_lat):
                continue
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
            img = axtest.scatter(X[idx]/Rt, Z[idx]/Rt, s = 10, label = f'{r:.2f}' if j not in [1] else None, c = coltest(i))
            axtest_sec.scatter(X[idx]/Rt, Z[idx]/Rt, s = 10, label = f'{r:.2f}' if j not in [1] else None, c = colors_obs[j])
            
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
            Mdot_prof[i] = const_C / np.sum(ray_dim**2) * np.pi * np.sum(ray_dim**2 * ray_d * ray_V_r)
            L_kin_prof[i] = const_C / np.sum(ray_dim**2)* 0.5 * np.pi *np.sum(ray_dim**2 * ray_d * ray_V_r**3)
            L_adv_prof[i] = const_C / np.sum(ray_dim**2) * np.pi * np.sum(ray_dim**2 * L_adv)
            ratio_un[i] = len(ray_d) / len_all if len_all > 0 else 0

            L_advmean_prof[i] =  const_C * np.pi * np.mean(L_adv)
            Mdotmean_prof[i] = const_C * np.pi * np.mean(ray_d * ray_V_r) if ray_V_r.size > 0 else 0 
            L_kinmean_prof[i] = const_C * np.pi * np.mean(ray_d * ray_V_r**3) if ray_V_r.size > 0 else 0
       
        outflow = {
            'r': ray_array,
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

    # axtest.legend()
    return all_outflows


#   
## MAIN
#
compute = False
which_part = 'wind' # 'outflow' or 'all' or 'wind' to have the wind
what_varies = 'theta' # 'r' or 'theta', only for radial profiles
which_obs = 'left_right_z' # 'left_right_z', 'all' or 'in_out_z'

if compute:
    snaps = [76, 109, 151]
    for snap in snaps: 
        path = f'{pre}/{snap}'
        if what_varies == 'r':
            ray_params = [Rt, 1e3*Rt, 300]
            r_chosen_name = ''
        elif what_varies == 'theta':
            r_chosen = amin
            r_chosen_name = 'amin'
            ray_params = [r_chosen, 100]

        all_outflows = profiles(path, snap, ray_params, which_obs, which_part, what_varies)
        out_path = f"{abspath}/data/{folder}/wind/{what_varies}_profile/{what_varies}{r_chosen_name}_profSec{snap}_{which_obs}_{which_part}.npy"
        np.save(out_path, all_outflows, allow_pickle=True)

else:
    which_plot = 'single_time' # 'time_compare' or 'single_time' or 
    # arrange for plotting
    observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
    x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    indices_obs, label_obs, colors_obs, _ = op.choose_observers(observers_xyz, which_obs)
    fig, (axd, axV, axM, axLkin) = plt.subplots(4, 1, figsize=(8, 22)) 
    figM, (axT, axLadv) = plt.subplots(1, 2, figsize=(15, 7))
    figr, axratio = plt.subplots(1, 1, figsize=(12, 10))
    all_axes = [axd, axV, axT, axM, axLadv, axLkin, axratio]
    
    if what_varies == 'r':
        r_chosen_name = ''
        norm = 1/Rt
        x_test = np.arange(1., 300)
        y_testplus1 = op.draw_line(x_test, [3.5, 1], 'powerlaw')
        y_test1 = op.draw_line(x_test, [9e4, -1], 'powerlaw')
        y_test23 = op.draw_line(x_test, [3.5e5, -2/3], 'powerlaw')
        y_test2 = op.draw_line(x_test, [2e-7, -2], 'powerlaw')
        axd.plot(x_test, y_test2, c = 'gray', ls = '-.', label = r'$\rho \propto r^{-2}$')
        axd.text(75, 2e-11, r'$\rho \propto r^{-2}$', fontsize = 18, color = 'gray', rotation = -20)
        axd.set_ylim(2e-13, 1e-5)
        axV.set_ylim(1.5e3, 1.5e4)
        axT.set_ylim(2e4, 1e6)
        axM.set_ylim(1e2, 1e7)
        axLadv.set_ylim(5e-2, 1e2)
        axLkin.set_ylim(1e-1, 5e2) 
        axratio.set_ylim(5e-2, 1.1)
    elif what_varies == 'theta':
        from Utilities.basic_units import radians
        r_chosen_name = 'amin'
        norm = radians 
        axd.set_title(f'r = {r_chosen_name}', fontsize = 18)
        axd.set_ylim(5e-14, 1e-8)
        axV.set_ylim(1.5e3, 1.5e4)
        axM.set_ylim(1e1, 1e6)
        axLkin.set_ylim(1e-4, 5) 

    if which_plot == 'time_compare':
        snaps = [76, 109, 151]
        which_parts = ['wind'] #['outflow', 'wind']
        labels_parts = []
        line_styles_parts = ['-.', '--', '-']
        
    else:
        snaps = [151]
        if what_varies == 'theta':
            which_parts = ['wind']
            labels_parts = ['05amin', 'amin', 'apo']
            line_styles_parts = ['-']
        else:
            which_parts = ['outflow', 'wind']
            labels_parts =  ['(unbound + bound) Outflow', 'Unbound outflow (wind)']
            line_styles_parts = ['--', '-']

    handles_color = []
    labels_color = []
    for s, snap in enumerate(snaps):
        # Load data Rph and Rtr
        path = f'{pre}/{snap}'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
        if which_plot == 'time_compare':
                labels_parts.append(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$')
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
    
        # Load profiles
        for k, which_part in enumerate(which_parts):
            profiles = np.load(f'{abspath}/data/{folder}/wind/{what_varies}_profile/{what_varies}{r_chosen_name}_profSec{snap}_{which_obs}_{which_part}.npy', allow_pickle=True).item()
            for i, lab in enumerate(profiles.keys()):
                if i > 2: #label_obs[i] == 'South pole':
                    continue 
                if which_plot == 'single_time':
                    lab_plot = lab
                else:
                    lab_plot = f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$'
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
                if what_varies == 'r':
                    r_plot, d, v_rad, t, Mdot, L_adv, L_kin, ratio_un = make_slices([r_plot, d, v_rad, t, Mdot, L_adv, L_kin, ratio_un], not_zero)
                    idx_rtr = np.argmin(np.abs(r_plot - rtr_nonzero_medians[i]))
                    idx_rph = np.argmin(np.abs(r_plot - rph_nonzero_medians[i]))
                    if label_obs[i] == 'Stream side': # just to cut the initially unbound material
                        idx_stop_d = np.where(np.logical_and(d > d[0], r_plot > apo))[0][0] #np.argmin(np.abs(r_plot - idx_stop_d_unb[k]*Rt)) 
                        d[idx_stop_d:] = 1e-20
                        Mdot[idx_stop_d:] = 1e-20
                        L_kin[idx_stop_d:] = 1e-20
                        # ratio_un[np.argmin(np.abs(r_plot - idx_stop_d_unb[0]*Rt)):] = 1e-20

                line = axd.plot(r_plot*norm, d * prel.den_converter, label = f'{lab_plot}' if which_part == 'wind' else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)[0]
                # elif which_plot == 'time_compare':
                #     line = axd.plot(r_plot*norm, d * prel.den_converter, label = f'{lab_plot}' if i ==0 else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)[0]
                axV.plot(r_plot*norm, v_rad * conversion_sol_kms, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axM.plot(r_plot*norm, Mdot/Medd_sol, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axratio.plot(r_plot*norm, ratio_un, label = f'{lab_plot}' if np.logical_and(which_part == 'wind', which_plot == 'single_time') else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axT.plot(r_plot*norm, t, label = f'{lab_plot}' if np.logical_and(which_part == 'wind', which_plot == 'single_time') else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axLadv.plot(r_plot*norm, L_adv/Ledd_sol, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axLkin.plot(r_plot*norm, L_kin/Ledd_sol, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)

                if np.logical_and(which_part == 'wind', s == 0): 
                    handles_color.append(line)
                    labels_color.append(lab)
                    
                if np.logical_and(what_varies =='r' , which_part == 'wind'):
                    axd.scatter(r_plot[idx_rph]*norm, d[idx_rph] * prel.den_converter, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axM.scatter(r_plot[idx_rph]*norm, Mdot[idx_rph]/Medd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axV.scatter(r_plot[idx_rph]*norm, v_rad[idx_rph] * conversion_sol_kms, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axLadv.scatter(r_plot[idx_rph]*norm, L_adv[idx_rph]/Ledd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axT.scatter(r_plot[idx_rph]*norm, t[idx_rph], marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axLkin.scatter(r_plot[idx_rph]*norm, L_kin[idx_rph]/Ledd_sol, marker = 'o', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)

                    axd.scatter(r_plot[idx_rtr]*norm, d[idx_rtr] * prel.den_converter, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axM.scatter(r_plot[idx_rtr]*norm, Mdot[idx_rtr]/Medd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axV.scatter(r_plot[idx_rtr]*norm, v_rad[idx_rtr] * conversion_sol_kms, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axLadv.scatter(r_plot[idx_rtr]*norm, L_adv[idx_rtr]/Ledd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axT.scatter(r_plot[idx_rtr]*norm, t[idx_rtr], marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axLkin.scatter(r_plot[idx_rtr]*norm, L_kin[idx_rtr]/Ledd_sol, marker = 'd', s = 100, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)

    axV.axhline(v_esc_kms, c = 'k', ls = 'dotted')# 
    # axV.text(35, 1.1*0.2*v_esc_kms, r'0.2v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    # axd.legend(fontsize = 18, loc = 'upper right')
    
    # Legend 1: colored observer lines (three colors)
    legend1 = axd.legend(handles=handles_color,
                        labels=labels_color,
                        fontsize=16,
                        loc='upper right')
    axd.add_artist(legend1)

    # Legend 2: line-style explanation (solid vs dashed)
    proxy_lines = []
    proxy_lines = []
    for l, line in enumerate(line_styles_parts):
        proxy_lines.append(
            mlines.Line2D([0], [0], color='cornflowerblue', ls=line, linewidth=2,
                        label=labels_parts[l])
        )

    legend2 = axd.legend(handles=proxy_lines, fontsize=16, 
                            loc='lower left' if which_plot == 'single_time' else 'upper left')

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
        if what_varies == 'r':
            ax.loglog()
            ax.axvline(apo*norm, color = 'k', ls = 'dotted')
            axd.text(0.8*apo*norm, 0.2*axd.get_ylim()[1], r'$r_{\rm a}$', fontsize = 20, color = 'k', rotation = 90)
            ax.set_xlim(1.5, 1.4e2)
        elif what_varies == 'theta':
            ax.set_yscale('log')
            ax.set_xlim(np.pi/3, 2*np.pi/3)
            ax.set_xticks([np.pi/3, 4*np.pi/9, np.pi/2, 5*np.pi/9, 2*np.pi/3])
            ax.set_xticklabels([r'$\pi/3$', r'$4\pi/9$', r'$\pi/2$', r'$5\pi/9$', r'$2\pi/3$'])
            # ax.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
            # ax.set_xticklabels([r'$0$', r'$\pi/6$', r'$\pi/3$', r'$\pi/2$', r'$2\pi/3$', r'$5\pi/6$', r'$\pi$'])
        ax.grid()
        if what_varies == 'theta':
            if ax != axLkin:
                ax.set_xlabel('')
            
    axLkin.set_xlabel(r'$r /r_{\rm t}$' if what_varies == 'r' else r'$\theta$', fontsize = 28)
    fig.tight_layout()
    figM.tight_layout()
    # if which_plot == 'time_compare':
    #     fig.savefig(f'{abspath}/Figs/{folder}/wind/den_prof_{what_varies}{r_chosen_name}_evol_zoom.png', bbox_inches = 'tight')
    # else:
    #     axd.set_title(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 30)
    #     figM.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 30)
    #     if what_varies == 'r':
    #         fig.savefig(f'{abspath}/Figs/2.paperWind/den_prof_{snap}.pdf', bbox_inches = 'tight')
    #         figM.savefig(f'{abspath}/Figs/2.paperWind/LT_{snap}.pdf', bbox_inches = 'tight')
    #     else: 
    #         fig.savefig(f'{abspath}/Figs/{folder}/wind/den_prof_{what_varies}{r_chosen_name}{snap}.png', bbox_inches = 'tight')
    