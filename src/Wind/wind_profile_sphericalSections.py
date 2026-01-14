""" Find density and (radial velocity) profiles for different lines of sight."""
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
    import matplotlib.cm as cm
    compute = False

import numpy as np
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import to_spherical_components, make_tree, choose_sections

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
which_obs = 'dark_bright_z_in_out'

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb']
t_fb_days = things['t_fb_days']
t_fb_days_cgs = t_fb_days * 24 * 3600 # in seconds
t_fb_sol = t_fb_days_cgs/prel.tsol_cgs
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs

# FUNCTIONS

def radial_profiles(loadpath, snap, ray_params):
    global rph_means
    data = make_tree(loadpath, snap, energy = True)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)  
    dim_cell = Vol**(1/3)
    # Trad = (Rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4)
    V_r, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    bern = orb.bern_coeff(Rsph, vel, Den, Mass, Press, IE_den, Rad_den, params)

    # cut_mid = np.abs(Z) > dim_cell #np.logical_and(X<-10*Rt, np.abs(Z) > 50) #dim_cell)
    cut = np.logical_and(bern > 0, V_r > 0)
    X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den, dim_cell, bern = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den, dim_cell, bern], cut)       
    
    # cond_sec = choose_sections(X, Y, Z)
    # cond_dark = cond_sec['dark']['cond']
    # colors_dark = cond_sec['dark']['color']
    # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,7))
    # mid_dark = np.logical_and(cond_dark, np.abs(Z) < dim_cell)
    # ax1.scatter(X[mid_dark]/Rp, Y[mid_dark]/Rp, c = colors_dark, s = 2, label = 'Dark side')
    # ax1.set_xlim(-120, 50) 
    # ax1.set_ylim(-50, 50)
    # ax1.legend(fontsize = 18)
    # ax1.set_xlabel(r'X [$r_{\rm p}$]')
    # ax1.set_ylabel(r'Y [$r_{\rm p}$]')
    # img = ax2.scatter(np.sqrt(X[cond_dark]**2 + Y[cond_dark]**2)/Rp, Z[cond_dark]/Rp, c = Den[cond_dark]*prel.den_converter, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-10, vmax = 1e-8), s = 2, label = 'Dark side')
    # cbar = fig.colorbar(img)
    # cbar.set_label(r'Density [g cm$^{-3}$]', fontsize=16)
    # ax2.set_xlim(0, 30) 
    # ax2.set_ylim(0, 30)
    # ax2.set_xlabel(r'R_{\rm cyl} [$r_{\rm p}$]')
    # ax2.set_ylabel(r'Z [$r_{\rm p}$]')
    # plt.suptitle(r'Wind material, t/t$_{\rm fb}$ = ' + f'{np.round(tfb,2)}' , fontsize = 20)
    # plt.tight_layout()
    # plt.savefig(f'{abspath}/Figs/next_meeting/section{snap}.png', bbox_inches = 'tight')

    rmin, rmax, Nray = ray_params
    r_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
    len_arr = len(rph_means)

    t_mean = np.zeros((Nray, len_arr))
    v_mean = np.zeros((Nray, len_arr))
    v_rad_mean = np.zeros((Nray, len_arr))
    d_mean = np.zeros((Nray, len_arr))
    L_adv_mean = np.zeros((Nray, len_arr))
    L_adv_median = np.zeros((Nray, len_arr))
    L_adv_weighmean = np.zeros((Nray, len_arr))
    Mdot_mean_dimcell = np.zeros((Nray, len_arr))
    for i, r in enumerate(r_array): 
        # find cells at r
        idx = np.abs(Rsph-r) < dim_cell
        ray_V = vel[idx]
        ray_V_r = V_r[idx] 
        ray_d = Den[idx] 
        ray_m = Mass[idx]
        ray_rad_den = Rad_den[idx]
        ray_vol = Vol[idx]
        ray_dim = dim_cell[idx]
        ray_t = (ray_rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4) 
        ray_Mdot = np.abs(ray_V_r) * ray_d 
        L_adv =  np.abs(ray_V_r) * ray_rad_den
        sections = choose_sections(X[idx],Y[idx],Z[idx], choice = which_obs)
        cond_sec = []
        colors_obs = []
        label_obs = []
        lines_obs = []
        for key in sections.keys():
            cond_sec.append(sections[key]['cond'])
            colors_obs.append(sections[key]['color'])
            label_obs.append(sections[key]['label'])
            lines_obs.append(sections[key]['line'])

        for j, cond in enumerate(cond_sec):
            # if i % 20 == 0:
            #     print(i, j, len(ray_t[cond]), flush=True)
            # print(np.where(L_adv[cond]==0))
            t_mean[i][j] = np.sum(ray_t[cond]*ray_vol[cond])/np.sum(ray_vol[cond])
            v_mean[i][j] = np.sum(ray_V[cond]*ray_m[cond])/np.sum(ray_m[cond])
            v_rad_mean[i][j] = np.sum(ray_V_r[cond]*ray_m[cond])/np.sum(ray_m[cond])
            d_mean[i][j] = np.sum(ray_d[cond]*ray_m[cond])/np.sum(ray_m[cond])
            L_adv_mean[i][j] = 4 * np.pi * r**2 * np.mean(L_adv[cond])
            L_adv_weighmean[i][j] = 4 * np.pi * r**2 * np.sum(L_adv[cond]*ray_m[cond])/np.sum(ray_m[cond])
            # Mdot_mean[i][j] = 4 * np.pi * r**2 * np.mean(ray_Mdot[cond]) if ray_Mdot[cond].size > 0 else 0 
            Mdot_mean_dimcell[i][j] = 4 * r**2 /np.sum(ray_dim[cond]**2) * np.pi * np.sum(ray_dim[cond]**2 * ray_Mdot[cond])

    outflow = {
        'r': r_array,
        't_mean': t_mean,
        'v_mean': v_mean,
        'v_rad_mean': v_rad_mean,
        'd_mean': d_mean,
        'L_adv_mean': L_adv_mean,
        'L_adv_weighmean': L_adv_weighmean,
        # 'Mdot_mean': Mdot_mean,
        'Mdot_mean_dimcell': Mdot_mean_dimcell,
        'colors_obs': colors_obs,
        'label_obs': label_obs,
        'lines_obs': lines_obs
    }
    return outflow

#
## MAIN
#
snap = 109

if plot:
    x_test = np.arange(1., 300)
    y_testplus1 = 3.5e3* (x_test)
    y_test1 = 9e4*(x_test)**(-1)
    y_test23 = 3.5e5*(x_test)**(-2/3)
    y_test2 = 5e-11* (x_test)**(-2) 
    figV, axV_tot = plt.subplots(1, 1, figsize=(9, 7))
    fig, (axV, axd, axT) = plt.subplots(1, 3, figsize=(24, 6)) 
    figM, (axMdot_dim, axL) = plt.subplots(1, 2, figsize=(18, 6))

path = f'{pre}/{snap}'
# Load data
tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 

ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
xph, yph, zph, Raddenph = ph_data[0], ph_data[1], ph_data[2], ph_data[6]
rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
rph = np.median(rph_all)
sections_ph = choose_sections(xph, yph, zph, which_obs)
cond_ph = []
rph_means = []
for key in sections_ph.keys(): 
    cond_single = sections_ph[key]['cond']
    cond_ph.append(cond_single)
    rph_means.append(np.mean(rph_all[cond_single]))
dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
x_tr, y_tr, z_tr, den_tr, vol_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['vol_tr'] 
r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
sec_tr = choose_sections(x_tr, y_tr, z_tr, which_obs)
cond_tr = []
rtr_medians = []
for key in sec_tr.keys():
    cond = sec_tr[key]['cond']
    rtr_medians.append(np.median(r_tr_all[np.logical_and(cond, r_tr_all!=0)]))

mass_tr = den_tr * vol_tr
nonzero = r_tr_all != 0
r_tr = np.median(r_tr_all[nonzero]) if nonzero.any() else 0 #then at early times you have rtr > rph 

if compute:
    all_outflows = radial_profiles(path, snap, [Rt, 3*rph, 200])
    out_path = f"{abspath}/data/{folder}/wind/den_prof{snap}SectionsForLoop{which_obs}.npy"
    np.save(out_path, all_outflows, allow_pickle=True)

if plot:
    profiles = np.load(f'{abspath}/data/{folder}/wind/den_prof{snap}SectionsForLoop{which_obs}.npy', allow_pickle=True).item()
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 

    # v_rad_tr = profiles['v_rad_mean'][np.argmin(np.abs(r_plot-r_tr))]
    # Mdot = profiles['Mdot_mean'][np.argmin(np.abs(r_plot-r_tr))]
    # t_dyn = (r_tr/v_rad_tr)*prel.tsol_cgs/t_fb_days_cgs # you want it in t_fb
    # tfb_adjusted = tfb - t_dyn
    # find_time = np.argmin(np.abs(tfb_fall-tfb_adjusted))
    # mfall_t = mfall[find_time]
    # eta = np.abs(Mdot/mfall_t) 

    # Radial profiles
    # R_edge = v_esc / prel.tsol_cgs * tfb * t_fb_days_cgs

    r_plot = profiles['r'] 
    d_all = profiles['d_mean']
    v_tot_all = profiles['v_mean']
    v_rad_all = profiles['v_rad_mean'] 
    t_all = profiles['t_mean']
    L_adv_all = profiles['L_adv_mean']
    L_adv_weighmean_all = profiles['L_adv_weighmean']
    # Mdot_all = profiles['Mdot_mean']
    Mdot_dimcell_all = profiles['Mdot_mean_dimcell']
    colors_all = profiles['colors_obs']
    label_obs_all = profiles['label_obs']
    idx_rtr = np.argmin(np.abs(r_plot - r_tr))

    v_term = v_esc_kms * np.sqrt(1-(r_plot/Rp)**(-2/3))

    for j, rph_cond in enumerate(rph_means):
        # if j in [5]:
        #     continue
        idx_rph = np.argmin(np.abs(r_plot - rph_cond))
        d = d_all[:idx_rph,j]
        v_tot = v_tot_all[:idx_rph,j]
        v_rad = v_rad_all[:idx_rph,j]
        t = t_all[:idx_rph,j]
        L_adv = L_adv_all[:idx_rph,j]
        Mdot_dimcell = Mdot_dimcell_all[:idx_rph,j]
        colors_obs = colors_all[j]
        label_obs = label_obs_all[j]
        t_dyn = r_plot[:idx_rph] / v_rad 
        axV_tot.plot(r_plot[:idx_rph]/Rt, v_tot * conversion_sol_kms,  color = colors_obs, label = f'{label_obs}')
        axV.plot(r_plot[:idx_rph]/Rt, v_rad * conversion_sol_kms,  color = colors_obs, label = f'{label_obs}')
        # axV.plot(r_plot/Rt, v_term, c = 'k', ls = 'dashed') #, label = r'v$_{\rm term}$')
        axd.plot(r_plot[:idx_rph]/Rt, d*prel.den_converter,  color = colors_obs, label = f'{label_obs}')
        axT.plot(r_plot[:idx_rph]/Rt, t,  color = colors_obs, label = f'{label_obs}')
        # axMdot.plot(r_plot/Rt, np.abs(Mdot/Medd_sol),  color = colors_obs, label = f'{label_obs}')
        # axMdot.scatter(r_tr/Rt, np.abs(Mdot[idx_rtr]/Medd_sol), color = colors_obs, s = 100, marker = 'd')
        axMdot_dim.plot(r_plot[:idx_rph]/Rt, np.abs(Mdot_dimcell/Medd_sol),  color = colors_obs, label = f'{label_obs}')
        axL.plot(r_plot[:idx_rph]/Rt, L_adv/Ledd_sol,  color = colors_obs, label = f'{label_obs}')
        # axL.plot(r_plot[:idx_rph]/Rt, L_adv_weighmean_all[:idx_rph,j]/Ledd_sol,  color = colors_obs)
        # axL.plot(r_plot[:idx_rph]/Rt, L_adv_median_all[:idx_rph,j]/Ledd_sol,  color = colors_obs, ls = 'dotted' , label = f'median' if j == 0 else "")
        # if snap != 76:
        #     axV.scatter(r_tr/Rt, v_rad[idx_rtr] * conversion_sol_kms, color = colors_obs, s = 100, marker = 'd')
        #     axd.scatter(r_tr/Rt, d[idx_rtr]*prel.den_converter, color = colors_obs, s = 100, marker = 'd')
        #     axT.scatter(r_tr/Rt, t[idx_rtr], color = colors_obs, s = 100, marker = 'd')
        #     axMdot_dim.scatter(r_tr/Rt, np.abs(Mdot_dimcell[idx_rtr]/Medd_sol), color = colors_obs, s = 100, marker = 'd')
        #     axL.scatter(r_tr/Rt, L_adv[idx_rtr]/Ledd_sol, color = colors_obs, s = 100, marker = 'd')

    axd.plot(x_test, y_test2, c = 'k', ls = 'dashed', label = r'$\rho \propto r^{-2}$')
    # axd.text(35, 1.1e-11, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
    axV.axhline(v_esc_kms, c = 'k', ls = 'dashed')# 
    # axV.text(35, 1.1*0.2*v_esc_kms, r'0.2v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    axT.plot(x_test, y_test23, c = 'k', ls = 'dashed', label = r'$T \propto r^{-2/3}$')
    # axT.text(1.2, 2.4e5, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -24)
    axL.plot(x_test, 1.5e-5*y_test23, c = 'k', ls = 'dashed', label = r'$L \propto r^{-2/3}$')
    # axL.text(1.2, 5.6e1, r'$L \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -18)

    for ax in [axV_tot, axd, axV, axMdot_dim, axT, axL]:
        ax.tick_params(axis='both', which='minor', length=6, width=1)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.loglog()
        ax.set_xlim(1, 1e2)
        ax.set_xlabel(r'$r [r_{\rm t}]$', fontsize = 28)
        ax.grid()
        if ax == axMdot_dim or ax == axd:
            ax.legend(fontsize = 18)
        # for j, rtr_cond in enumerate(rtr_medians):
            # if j == 3 or j ==2:
            #     continue
            # ax.axvline(rtr_cond/Rt, color = colors_all[j], ls = 'dotted')
            # ax.axvline(rph_means[j]/Rt, color = colors_all[j], ls = 'dashed')
            # ax.axvspan(np.min(rph_all[cond_ph[j]]/Rt), np.max(rph_all[cond_ph[j]]/Rt), color = colors_all[j], ls = 'dotted', alpha = 0.2)
            

    # axMdot.set_ylim(7e2, 1e7)
    # axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28) 
    # axMdot.set_title(r'Mean over spherical shells: $4\pi r^2\langle \rho v_r \rangle_{\rm wind \,cells}$', fontsize = 20)
    # axMdot_dim.set_title(r'Mean over spherical shells weighted by cell dimension: $\big(4 r^2/\sum s^2\big) \sum \pi s^2 \rho v_r$', fontsize = 16)
    axMdot_dim.set_ylim(7e2, 5e5)
    axd.set_ylim(1e-13, 1e-8)
    axV.set_ylim(2e3, 3e4)
    axV_tot.set_ylim(2e3, 3e4)
    axT.set_ylim(2e4, 5e5)
    axL.set_ylim(1e-1, 50)

    axMdot_dim.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28)
    axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
    axV_tot.set_ylabel(r'$|v|$ [km/s]', fontsize = 28)
    axV.set_ylabel(r'v$_{\rm r}$ [km/s]', fontsize = 28)
    axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
    axL.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
    fig.tight_layout()
    # fig.savefig(f'{abspath}/Figs/paper/den_profShell{which_part}_{suffix_saveing}.pdf', bbox_inches = 'tight')
    # figM_dim.savefig(f'{abspath}/Figs/paper/MwShell{which_part}.pdf', bbox_inches = 'tight')
    # figL.savefig(f'{abspath}/Figs/paper/LShell{which_part}.pdf', bbox_inches = 'tight')
    plt.show()

