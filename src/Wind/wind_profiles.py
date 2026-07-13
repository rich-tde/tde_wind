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
from src.Wind.Rtrapp_tdiff import load_and_adjust_rtrap

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
tfallback = things['t_fb_days']
tfallback_code_units = tfallback * 24 * 3600 / prel.tsol_cgs
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
        X_all = X.copy()
        Y_all = Y.copy()
        Z_all = Z.copy()
    elif what_varies == 'theta':
        _, lat_all, _ = op.to_spherical_coordinate(X, Y, Z, r_frame = 'us') 

    Mass_all = Mass.copy()
    dim_cell_all = dim_cell.copy()
    sections = op.choose_sections(X, Y, Z, which_obs)
    cond_sec_all = []
    for key in sections.keys():
        cond_sec_all.append(sections[key]['cond'])

    cut_wind, bern, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
    if which_part == 'wind':
        cut = cut_wind
    elif which_part == 'outflow':
        cut = V_r >= 0 
    elif which_part == 'all':
        cut = Den > 1e-19
    elif which_part == 'acc':
        cut = np.logical_and(V_r < 0, bern < 0)
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
    # figtest, (ax1, ax2) = plt.subplots(1,2,figsize=(16,8))
    # coltest = plt.cm.get_cmap('rainbow', Nray)
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
        lens_tot = np.zeros(Nray)
        ratio_un = np.zeros(Nray)
        Mass_tot = np.zeros(Nray)
        Mass_wind = np.zeros(Nray)
        # ratio_Mass = np.zeros(Nray)
        
        for i, r in enumerate(ray_array): 
            if what_varies == 'r':
                const_C = 4*r**2/len(cond_sec)
            if what_varies == 'theta':
                if r < min_lat:
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
            idx_all = shell_all[mask_all]
            len_all = len(idx_all)

            # if what_varies == 'theta':
            #     img = ax1.scatter(X[idx]/Rt, Z[idx]/Rt, s = 10, label = f'{r:.2f}' if j not in [1] else None, c = coltest(i))
            #     ax2.scatter(X[idx]/Rt, Z[idx]/Rt, s = 10, label = f'{r:.2f}' if j not in [1] else None, c = colors_obs[j])
            # else: # bern is not cut, so it's of all cells, not wind
            #     X_toplot = X_all[idx_all]
            #     Y_toplot = Y_all[idx_all]
            #     Z_toplot = Z_all[idx_all]
            #     bern_toplot = bern[idx_all]
            #     dim_toplot = dim_cell_all[idx_all]
            #     img = ax1.scatter(X_toplot[np.abs(Z_toplot)<dim_toplot]/Rt, Y_toplot[np.abs(Z_toplot)<dim_toplot]/Rt, s = 2, c = bern_toplot[np.abs(Z_toplot)<dim_toplot], vmin = -10, vmax = 10, cmap = 'coolwarm')
            #     ax2.scatter(X_toplot[np.abs(Y_toplot)<dim_toplot]/Rt, Z_toplot[np.abs(Y_toplot)<dim_toplot]/Rt, s = 2, c = bern_toplot[np.abs(Y_toplot)<dim_toplot],  vmin = -10, vmax = 10, cmap = 'coolwarm')

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
            lens_tot[i] = len_all
            Mass_tot[i] = np.sum(Mass_all[idx_all])
            Mass_wind[i] = np.sum(ray_m)
            # ratio_Mass[i] = np.sum(ray_m) / Mass_tot[i] if Mass_tot[i] > 0 else 0

            L_advmean_prof[i] =  const_C * np.pi * np.mean(L_adv)
            Mdotmean_prof[i] = const_C * np.pi * np.mean(ray_d * ray_V_r) if ray_V_r.size > 0 else 0 
            L_kinmean_prof[i] = const_C * np.pi * np.mean(ray_d * ray_V_r**3) if ray_V_r.size > 0 else 0

        outflow = {
            'r': ray_array,
            't_prof': t_prof,
            'v_rad_prof': v_rad_prof,
            'm_prof': ray_m,
            'd_prof': d_prof,
            'Mdot_prof': Mdot_prof,
            'L_advmean_prof': L_advmean_prof,
            'Mdotmean_prof': Mdotmean_prof,
            'L_adv_prof': L_adv_prof,
            'L_kin_prof': L_kin_prof,
            'L_kinmean_prof': L_kinmean_prof,
            'Ntot_cells': lens_tot,
            'ratio_un': ratio_un,
            'Mass_tot': Mass_tot,
            # 'ratio_Mass': ratio_Mass,
            'Mass_wind': Mass_wind,
            'colors_obs': colors_obs[j],
            'lines_obs': lines_obs[j]
        }

        key = f"{label_obs[j]}"
        all_outflows[key] = outflow

    # if what_varies == 'theta':
    #     ax1.legend()
    #     ax1.set_ylabel(r'Z/$r_{\rm t}$')
    #     plt.colorbar(img, label = 'Bernoulli')
    # if what_varies == 'r':
    #     ax1.set_ylabel(r'Y/$r_{\rm t}$')
    #     ax2.set_ylabel(r'Z/$r_{\rm t}$')
    #     for ax in [ax1, ax2]:
    #         ax.set_xlim(-200, 200)
    #         ax.set_ylim(-200, 200)
    # ax1.set_xlabel(r'X/$r_{\rm t}$')
    # ax2.set_xlabel(r'X/$r_{\rm t}$')
    # plt.tight_layout()
    # if what_varies == 'r':
    #     figtest.savefig(f'{abspath}/Figs/2.paperWind/deeperanalysis/rad_prof_selection{snap}.png')
    
    return all_outflows

#   
## MAIN
#
compute = False
which_part = 'acc' # 'outflow' or 'all' or 'wind' to have the wind
what_varies = 'r' # 'r' or 'theta', only for radial profiles
which_obs = 'left_right_z' # 'left_right_z', 'all' or 'in_out_z'
if what_varies == 'r':
    r_chosen_name = '' 
elif what_varies == 'theta':
    r_chosen = apo
    r_chosen_name = 'apo'

if compute:
    snaps = [151]
    for snap in snaps: 
        path = f'{pre}/{snap}'
        if what_varies == 'r':
            ray_params = [Rt, 1e3*Rt, 300] 
        elif what_varies == 'theta':
            ray_params = [r_chosen, 100]

        all_outflows = profiles(path, snap, ray_params, which_obs, which_part, what_varies)
        out_path = f"{abspath}/data/{folder}/wind/{what_varies}_profile/{what_varies}{r_chosen_name}_profSec{snap}_{which_obs}_{which_part}.npy"
        np.save(out_path, all_outflows, allow_pickle=True)

else:
    which_plot = 'single_time' # 'time_compare' or 'single_time' 
    # arrange for plotting
    observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
    x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    indices_obs, label_obs, colors_obs, _ = op.choose_observers(observers_xyz, which_obs)
    fig, (axd, axV, axM, axLkin) = plt.subplots(4, 1, figsize=(8, 22)) 
    figM, (axT, axLadv) = plt.subplots(1, 2, figsize=(15, 8))
    fiC, axC = plt.subplots(1, 1, figsize=(8, 8))
    figr, ((axNcell, axNmass), (axratio, axratioM)) = plt.subplots(2, 2, figsize=(18, 18))
    # if which_plot == 'time_compare':
    #     figr, (axratio, axratioM) = plt.subplots(1, 2, figsize=(24, 10))
        # all_axes = [axd, axV, axT, axM, axLadv, axLkin, axratio, axratioM, axC]
    # elif which_plot == 'single_time':
    all_axes = [axd, axV, axT, axM, axLadv, axLkin, axratio, axratioM, axNcell, axNmass, axC]

    if what_varies == 'r':
        norm = 1/Rt
        x_test = np.arange(1., 300)
        y_testplus1 = op.draw_line(x_test, [3.5, 1], 'powerlaw')
        y_test1 = op.draw_line(x_test, [9e4, -1], 'powerlaw')
        y_test23 = op.draw_line(x_test, [3.5e5, -2/3], 'powerlaw')
        y_test2 = op.draw_line(x_test, [2e-7, -2], 'powerlaw')
        axd.set_ylim(2e-13, 1e-5)
        axV.set_ylim(1.5e3, 1.5e4)
        axT.set_ylim(2e4, 1e6)
        axM.set_ylim(1e2, 1e7)
        axC.set_ylim(1e2, 1e7)
        axLadv.set_ylim(1e-2, 1e2)
        axLkin.set_ylim(1e-1, 5e2) 
        axratio.set_ylim(1e-2, 1.1)
        axratioM.set_ylim(1e-2, 1.1)
        axd.text(0.8*apo*norm, 0.2*axd.get_ylim()[1], r'$r_{\rm a}$', fontsize = 20, color = 'gray', rotation = 90)   
        axd.plot(x_test, y_test2, c = 'gray', ls = '-.', label = r'$\rho \propto r^{-2}$')
        axd.text(75, 2e-11, r'$\rho \propto r^{-2}$', fontsize = 18, color = 'gray', rotation = -20)            
    elif what_varies == 'theta':
        from Utilities.basic_units import radians
        norm = radians 
        axd.set_title(f'r = {r_chosen_name}', fontsize = 18)
        axd.set_ylim(1e-12, 2e-6)
        axV.set_ylim(1.5e3, 1.5e4)
        axM.set_ylim(1e1, 1e6)
        axLkin.set_ylim(1e-3, 40) 

    if which_plot == 'time_compare':
        snaps = [76, 109, 151]
        which_parts = ['wind'] #['outflow', 'wind']
        labels_parts = []
        line_styles_parts = ['-.', '--', '-']
        
    else:
        snaps = [151] 
        which_parts = ['outflow', 'wind']#, 'acc'] 
        labels_parts =  ['(unbound + bound) Outflow', 'Unbound outflow (wind)']#, 'Accretion'] #['Unbound outflow (wind)']#
        line_styles_parts = ['--', '-']#, '-.'] 

    handles_color = []
    labels_color = []
    v_snap_peakM = np.zeros(len(snaps))
    R_peakM = np.zeros(len(snaps))
    tfbs = np.zeros(len(snaps))
    for s, snap in enumerate(snaps):
        # Load data Rph and Rtr
        path = f'{pre}/{snap}'
        tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
        tfbs[s] = tfb
        if which_plot == 'time_compare':
                labels_parts.append(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$')
        if check == 'HiResNewAMR':
            photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
            xph, yph, zph = photo['x'], photo['y'], photo['z']
        else:
            photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
            xph, yph, zph = photo[0], photo[1], photo[2]
        rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
        pathtrap = f"{abspath}/data/{folder}/trap"
        dataRtr = load_and_adjust_rtrap(pathtrap, check, snap)
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
        # Load profiles
        for k, which_part in enumerate(which_parts):
            profiles = np.load(f'{abspath}/data/{folder}/wind/{what_varies}_profile/{what_varies}{r_chosen_name}_profSec{snap}_{which_obs}_{which_part}.npy', allow_pickle=True).item()
            for i, lab in enumerate(profiles.keys()):
                if label_obs[i] == 'South pole':
                    continue 
                if which_plot == 'single_time':
                    lab_plot = lab
                else:
                    lab_plot = f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$'
                r_arr = profiles[lab]['r'] 
                d = profiles[lab]['d_prof']
                v_rad = np.abs(profiles[lab]['v_rad_prof'])
                t = profiles[lab]['t_prof']
                Mdot = np.abs(profiles[lab]['Mdot_prof']) #Mdotmean_prof
                L_adv = np.abs(profiles[lab]['L_adv_prof']) #L_advmean_prof
                L_kin = np.abs(profiles[lab]['L_kin_prof']) #L_kinmean_prof
                Mdotmean = profiles[lab]['Mdotmean_prof'] 
                L_advmean = profiles[lab]['L_advmean_prof'] 
                L_kinmean = profiles[lab]['L_kinmean_prof'] 
                ratio_un = profiles[lab]['ratio_un']
                Ntot_cells = profiles[lab]['Ntot_cells']
                Nwind_cells = ratio_un * Ntot_cells
                Mass_wind = profiles[lab]['Mass_wind']
                Mass_tot = profiles[lab]['Mass_tot']
                ratio_Mass = Mass_wind/Mass_tot
                colors_sec = colors_obs[i] #profiles[lab]['colors_obs']
                # Mdot = d * r_plot**2 * v_rad
                not_zero = np.where(np.logical_and(d != 0, r_arr > 0))
                if what_varies == 'r':
                    r_plot, d, v_rad, t, Mdot, L_adv, L_kin, ratio_un, ratio_Mass, Mdotmean = \
                        make_slices([r_arr, d, v_rad, t, Mdot, L_adv, L_kin, ratio_un, ratio_Mass, Mdotmean], not_zero)
                    if which_part == 'wind':
                        Nwind_cells, Ntot_cells, Mass_wind, Mass_tot = \
                            make_slices([Nwind_cells, Ntot_cells, Mass_wind, Mass_tot], not_zero)
                    idx_rtr = np.argmin(np.abs(r_plot - rtr_nonzero_medians[i]))
                    idx_rph = np.argmin(np.abs(r_plot - rph_nonzero_medians[i]))
                    if label_obs[i] == 'Stream side': # just to cut the initially unbound material
                        if which_part != 'acc':
                            idx_stop_d = np.where(np.logical_and(d > d[0], r_plot > apo))[0][0] #np.argmin(np.abs(r_plot - idx_stop_d_unb[k]*Rt)) 
                            d[idx_stop_d:] = 1e-20
                            Mdot[idx_stop_d:] = 1e-20
                            L_kin[idx_stop_d:] = 1e-20
                            ratio_un[idx_stop_d:] = 1e-20
                            ratio_Mass[idx_stop_d:] = 1e-20
                        if which_part == 'wind':
                            Nwind_cells[idx_stop_d-2:] = 0 # -2 to avoid weird spikes
                            # Mass_wind[idx_stop_d-2:] = 0
                            Ntot_cells[idx_stop_d-2:] = 0
                            # Mass_tot[idx_stop_d-2:] = 0
                            idx_maxM = np.argmax(Mass_wind)
                            v_snap_peakM[s] = v_rad[idx_maxM]
                            R_peakM[s] = r_plot[idx_maxM]
                else:
                    r_plot = r_arr

                
                line = axd.plot(r_plot*norm, d * prel.den_converter, label = f'{lab_plot}' if which_part == 'wind' else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)[0]
                # elif which_plot == 'time_compare':
                #     line = axd.plot(r_plot*norm, d * prel.den_converter, label = f'{lab_plot}' if i ==0 else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)[0]
                axV.plot(r_plot*norm, v_rad * conversion_sol_kms, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axM.plot(r_plot*norm, Mdot/Medd_sol, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axT.plot(r_plot*norm, t, label = f'{lab_plot}' if np.logical_and(which_part == 'wind', which_plot == 'single_time') else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axLadv.plot(r_plot*norm, L_adv/Ledd_sol, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                axLkin.plot(r_plot*norm, L_kin/Ledd_sol, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)

                if np.logical_and(which_part == 'wind', s == 0): 
                    handles_color.append(line)
                    labels_color.append(lab)
                
                if which_part == 'wind':
                    deltaR = np.diff(np.concatenate([r_plot, [r_plot[-1] + np.diff(r_plot)[-1]]]))
                    tocheck = Mass_wind * v_rad/deltaR
                    axNcell.plot(r_plot*norm, Nwind_cells, color = colors_sec, linewidth = 2, label = f'{lab_plot}', ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s])
                    # axNcell.plot(r*norm, Ntot_cells, color = colors_sec, ls = ':', linewidth = 2)
                    axNmass.plot(r_plot*norm, Mass_wind, color = colors_sec, linewidth = 2, label = 'Wind' if i == 0 else None, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s])
                    # axNmass.plot(r_plot*norm, Mass_tot, color = colors_sec, ls = ':', linewidth = 2, label = 'Total' if i == 0 else None)
                    axratio.plot(r_plot*norm, ratio_un, label = f'{lab_plot}' if np.logical_and(which_part == 'wind', which_plot == 'single_time') else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axratioM.plot(r_plot*norm, ratio_Mass, label = f'{lab_plot}' if np.logical_and(which_part == 'wind', which_plot == 'single_time') else None, color = colors_sec, ls = line_styles_parts[k] if which_plot == 'single_time' else line_styles_parts[s], linewidth = 2)
                    axC.plot(r_plot*norm, Mdot/Medd_sol, color = colors_sec, label = 'corrected sum' if np.logical_and(i == 0, s == 0) else None)
                    axC.plot(r_plot*norm, Mdotmean/Medd_sol, color = colors_sec, ls = '--', label = 'uniform mean' if np.logical_and(i == 0, s == 0) else None)
                    axC.plot(r_plot*norm, tocheck/Medd_sol, color = colors_sec, label = r'M$v_r/\Delta r$'  if np.logical_and(i == 0, s == 0) else None, ls = ':')
                    
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

    axV.axhline(v_esc_kms, c = 'k', ls = 'dotted')
    for i in range(len(snaps)-1):
        print('Effective delta r: ', R_peakM[i+1]-R_peakM[i], 't_peak/v_r: ' , (tfbs[i+1]-tfbs[i])*tfallback_code_units/v_snap_peakM[i])

    # Legend 1: colored observer lines (three colors)
    legend1 = axd.legend(handles=handles_color,
                        labels=labels_color,
                        fontsize=17,
                        loc='upper right' if what_varies == 'r' else 'upper left')
    axd.add_artist(legend1)

    # Legend 2: line-style explanation (solid vs dashed)
    proxy_lines = []
    proxy_lines = []
    for l, line in enumerate(line_styles_parts):
        proxy_lines.append(
            mlines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                        label=labels_parts[l])
        )

    axT.legend(fontsize = 17)
    axC.legend(fontsize = 17)
    axd.set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 28)
    axV.set_ylabel(r'v$_{\rm r}$ (km/s)', fontsize = 28)
    axT.set_ylabel(r'$T_{\rm rad}$ (K)', fontsize = 28)
    axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 28)
    axLadv.set_ylabel(r'$L_{\rm adv} (L_{\rm Edd})$', fontsize = 28)
    axLkin.set_ylabel(r'$L_{\rm kin} (L_{\rm Edd})$', fontsize = 28)
    axratio.set_ylabel(r'f$_{\rm unb}$', fontsize = 28)
    axratioM.set_ylabel(r'M$_{\rm wind}/M_{\rm sec}$', fontsize = 28)
    figM.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 22)
    axNcell.set_ylabel(r'N$_{\rm cells}$', fontsize = 28)
    axNmass.set_ylabel(r'M$_{\rm cells} [M_\odot]$', fontsize = 28)
    axNcell.legend(fontsize = 22)
    axNmass.legend(fontsize = 22)
    for ax in all_axes: 
        if ax in [axd, axT]:
            ax.legend(handles=proxy_lines, fontsize=17, 
                    loc='lower left' if what_varies == 'r' else 'upper left',          # anchor corner of the second legend
                    bbox_to_anchor=(0, 0) if what_varies == 'r' else (0, .75))
        ax.tick_params(axis='both', which='minor', length = 8, width = 1)
        ax.tick_params(axis='both', which='major', length = 15, width = 1.5)
        if what_varies == 'r':
            ax.loglog()
            ax.axvline(apo*norm, color = 'gray', ls = '--')
            ax.set_xlim(1.5, 1.4e2)
            if ax in [axLadv, axLkin, axT, axratio, axratioM]:
                ax.set_xlabel(r'$r /r_{\rm t}$' if what_varies == 'r' else r'$\theta$', fontsize = 28)
        elif what_varies == 'theta':
            ax.set_yscale('log')
            ax.set_xlim(0, 2*np.pi/3)
            # ax.set_xlim(np.pi/3, 2*np.pi/3)
            # ax.set_xticks([np.pi/3, 4*np.pi/9, np.pi/2, 5*np.pi/9, 2*np.pi/3])
            # ax.set_xticklabels([r'$\pi/3$', r'$4\pi/9$', r'$\pi/2$', r'$5\pi/9$', r'$2\pi/3$'])
            # ax.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
            # ax.set_xticklabels([r'$0$', r'$\pi/6$', r'$\pi/3$', r'$\pi/2$', r'$2\pi/3$', r'$5\pi/6$', r'$\pi$'])
            if ax != axLkin:
                ax.set_xlabel('')
        ax.grid()
    fig.tight_layout()
    figM.tight_layout()
    figr.tight_layout()
    figr.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 22)
    figr.tight_layout()
    if which_plot == 'time_compare':
        fig.savefig(f'{abspath}/Figs/{folder}/wind/den_prof_{what_varies}{r_chosen_name}_evol.png', bbox_inches = 'tight')
        figr.savefig(f'{abspath}/Figs/{folder}/wind/ratio_un_{what_varies}{r_chosen_name}_evol.png', bbox_inches = 'tight')
    else:
        axd.set_title(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 22)
        if np.logical_and(what_varies == 'r', which_obs == 'left_right_z'):
            fig.savefig(f'{abspath}/Figs/2.paperWind/den_prof{r_chosen_name}_{snap}.pdf', bbox_inches = 'tight')
            figM.savefig(f'{abspath}/Figs/2.paperWind/deeperanalysis/LT_{snap}.pdf', bbox_inches = 'tight')
            figr.savefig(f'{abspath}/Figs/2.paperWind/deeperanalysis/ratio_{snap}.pdf', bbox_inches = 'tight')
        else: 
            fig.savefig(f'{abspath}/Figs/{folder}/wind/{what_varies}_profile/den_prof_{what_varies}{r_chosen_name}{snap}_{which_obs}.png', bbox_inches = 'tight')
    
# %%