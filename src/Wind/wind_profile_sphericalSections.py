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
snap = 76
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
which_obs = 'dark_bright_z'

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

def radial_profiles(loadpath, snap, ray_params, which_obs):
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
    cut = np.logical_and(bern > 0, V_r > 0)
    X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den, dim_cell, bern = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den, dim_cell, bern], cut)       
    indices_all = np.arange(len(X))
    
    # split in sections
    sections = choose_sections(X, Y, Z, choice = which_obs)
    ind_sec = []
    colors_obs = []
    label_obs = []
    lines_obs = []
    for key in sections.keys():
        cond_sec = sections[key]['cond']
        ind_sec.append(indices_all[cond_sec])
        colors_obs.append(sections[key]['color'])
        label_obs.append(sections[key]['label'])
        lines_obs.append(sections[key]['line'])

    # cut_mid = np.abs(Z) > dim_cell #np.logical_and(X<-10*Rt, np.abs(Z) > 50) #dim_cell)
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


    all_outflows = {}
    for j, ind in enumerate(ind_sec):
        t_prof = np.zeros(Nray)
        v_prof = np.zeros(Nray)
        v_rad_prof = np.zeros(Nray)
        d_prof = np.zeros(Nray)
        Mdot_prof = np.zeros(Nray)
        Mdotmean_prof = np.zeros(Nray)
        L_kin_prof = np.zeros(Nray)
        L_adv_prof = np.zeros(Nray)
        for i, r in enumerate(r_array): 
            # find cells at r
            cond_r = np.abs(Rsph-r) < dim_cell
            const_C = 4/len(ind_sec)
            ind_r = indices_all[cond_r]
            ind_sec_r = np.intersect1d(ind_r, ind)
            if len(ind_sec_r) == 0:
                continue
            ray_V = vel[ind_sec_r]
            ray_V_r = V_r[ind_sec_r] 
            ray_d = Den[ind_sec_r] 
            ray_m = Mass[ind_sec_r]
            ray_rad_den = Rad_den[ind_sec_r]
            ray_vol = Vol[ind_sec_r]
            ray_dim = dim_cell[ind_sec_r]
            ray_t = (ray_rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4) 
            L_adv =  ray_V_r * ray_rad_den
            t_prof[i] = np.sum(ray_t*ray_vol)/np.sum(ray_vol)
            v_prof[i] = np.sum(ray_V*ray_m)/np.sum(ray_m)
            v_rad_prof[i] = np.sum(ray_V_r*ray_m)/np.sum(ray_m)
            d_prof[i] = np.sum(ray_d*ray_m)/np.sum(ray_m)
            Mdot_prof[i] = const_C * r**2 /np.sum(ray_dim**2) * np.pi * np.sum(ray_dim**2 * ray_d * ray_V_r)
            Mdotmean_prof[i] = 4 * np.pi * r**2 * np.mean(ray_d * ray_V_r) if ray_V_r.size > 0 else 0 
            L_kin_prof[i] = const_C * r**2 /np.sum(ray_dim**2)* 0.5 * np.pi *np.sum(ray_dim**2 * ray_d * ray_V_r**3)
            L_adv_prof[i] = const_C * np.pi * r**2 * np.mean(L_adv)

        outflow = {
            'r': r_array,
            't_prof': t_prof,
            'v_prof': v_prof,
            'v_rad_prof': v_rad_prof,
            'd_prof': d_prof,
            'Mdot_prof': Mdot_prof,
            'Mdotmean_prof': Mdotmean_prof,
            'L_adv_prof': L_adv_prof,
            'L_kin_prof': L_kin_prof,
            'colors_obs': colors_obs[j],
            'lines_obs': lines_obs[j]
        }

        key = f"{label_obs[j]}"
        all_outflows[key] = outflow
    
    return all_outflows

#
## MAIN
#

if compute:
    path = f'{pre}/{snap}'
    all_outflows = radial_profiles(path, snap, [Rt, 8*apo, 100], which_obs)
    out_path = f"{abspath}/data/{folder}/wind/rad_profSec{snap}_{which_obs}.npy"
    np.save(out_path, all_outflows, allow_pickle=True)

if plot:
    # Load data Rph and Rtr
    path = f'{pre}/{snap}'
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
    ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    xph, yph, zph, Raddenph = ph_data[0], ph_data[1], ph_data[2], ph_data[6]
    rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
    sections_ph = choose_sections(xph, yph, zph, which_obs)
    rph_medians = []
    color_sec = []
    for key in sections_ph.keys(): 
        color_sec.append(sections_ph[key]['color'])
        cond_single = sections_ph[key]['cond']
        rph_medians.append(np.mean(rph_all[cond_single]))
    dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
    x_tr, y_tr, z_tr, den_tr, vol_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['vol_tr'] 
    r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    sec_tr = choose_sections(x_tr, y_tr, z_tr, which_obs)
    rtr_medians = []
    for key in sec_tr.keys():
        cond_single = sec_tr[key]['cond']
        rtr_medians.append(np.median(r_tr_all[np.logical_and(cond_single, r_tr_all!=0)]))
    
    x_test = np.arange(1., 300)
    y_testplus1 = 3.5e3* (x_test)
    y_test1 = 9e4*(x_test)**(-1)
    y_test23 = 3.5e5*(x_test)**(-2/3)
    y_test2 = 1e-8* (x_test)**(-2) 
    fig, (axd, axV, axT) = plt.subplots(1, 3, figsize=(26, 7)) 
    figM, (axMdot, axLadv, axLkin) = plt.subplots(1, 3, figsize=(26, 7))
    # Load profiles
    profiles = np.load(f'{abspath}/data/{folder}/wind/rad_profSec{snap}_{which_obs}.npy', allow_pickle=True).item()
    for i, lab in enumerate(profiles.keys()):
        if i == 3:
            continue
        r_plot = profiles[lab]['r'] 
        d = profiles[lab]['d_prof']
        v_tot = profiles[lab]['v_prof']
        v_rad = profiles[lab]['v_rad_prof'] 
        t = profiles[lab]['t_prof']
        Mdot = profiles[lab]['Mdot_prof']
        L_adv = profiles[lab]['L_adv_prof']
        L_kin = profiles[lab]['L_kin_prof']
        # Mdot_mean = profiles[lab]['Mdotmean_prof']
        colors_sec = profiles[lab]['colors_obs']
        idx_rtr = np.argmin(np.abs(r_plot - rtr_medians[i]))
        idx_rph = np.argmin(np.abs(r_plot - rph_medians[i]))

        # axV_tot.plot(r_plot/Rt, v_tot * conversion_sol_kms,  color = colors_sec, label = f'{lab}')
        
        axd.plot(r_plot/Rt, d*prel.den_converter,  color = colors_sec, label = f'{lab}')
        axV.plot(r_plot/Rt, v_rad * conversion_sol_kms,  color = colors_sec, label = f'{lab}')
        axT.plot(r_plot/Rt, t,  color = colors_sec, label = f'{lab}')

        axMdot.plot(r_plot/Rt, Mdot/Medd_sol,  color = colors_sec, label = f'{lab}')
        # axMdot.plot(r_plot/Rt, Mdot_mean/Medd_sol,  color = colors_sec, label = f'{lab}')
        axLadv.plot(r_plot/Rt, L_adv/Ledd_sol,  color = colors_sec, label = f'{lab}')
        axLkin.plot(r_plot/Rt, L_kin/Ledd_sol,  color = colors_sec, label = f'{lab}')

    axd.plot(x_test, y_test2, c = 'k', ls = 'dashed') #, label = r'$\rho \propto r^{-2}$')
    # axd.text(35, 1.1e-11, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
    axV.axhline(v_esc_kms, c = 'k', ls = 'dashed')# 
    # axV.text(35, 1.1*0.2*v_esc_kms, r'0.2v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    axT.plot(x_test, y_test23, c = 'k', ls = 'dashed', label = r'$T \propto r^{-2/3}$')
    # axT.text(1.2, 2.4e5, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -24)
    axLadv.plot(x_test, 5e-5*y_test23, c = 'k', ls = 'dashed', label = r'$L \propto r^{-2/3}$')
    # axLadv.text(1.2, 5.6e1, r'$L \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -18)

    for ax in [axd, axV, axT, axMdot, axLadv, axLkin]:
        ax.tick_params(axis='both', which='minor', length = 6, width = 1)
        ax.tick_params(axis='both', which='major', length = 10, width = 1.5)
        ax.loglog()
        ax.set_xlim(1, 2e2)
        ax.set_xlabel(r'$r [r_{\rm t}]$', fontsize = 28)
        ax.grid()
        if ax == axMdot or ax == axd:
            ax.legend(fontsize = 18)
        # for j, rtr_cond in enumerate(rtr_medians):
            # if j == 3 or j ==2:
            #     continue
            # ax.axvline(rtr_cond/Rt, color = colors_all[j], ls = 'dotted')
            # ax.axvline(rph_profs[j]/Rt, color = colors_all[j], ls = 'dashed')
            # ax.axvspan(np.min(rph_all[cond_ph[j]]/Rt), np.max(rph_all[cond_ph[j]]/Rt), color = colors_all[j], ls = 'dotted', alpha = 0.2)
            

    # axMdot.set_ylim(7e2, 1e7)
    # axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28) 
    # axMdot.set_title(r'Mean over spherical shells: $4\pi r^2\langle \rho v_r \rangle_{\rm wind \,cells}$', fontsize = 20)
    # axMdot.set_title(r'Mean over spherical shells weighted by cell dimension: $\big(4 r^2/\sum s^2\big) \sum \pi s^2 \rho v_r$', fontsize = 16)
    axMdot.set_ylim(1e3, 1e6)
    axd.set_ylim(1e-13, 1e-8)
    axV.set_ylim(2e3, 3e4)
    axT.set_ylim(2e4, 5e5)
    axLadv.set_ylim(1e-1, 1e2)
    axLkin.set_ylim(1e-1, 1e2)

    axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28)
    axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
    axV.set_ylabel(r'v$_{\rm r}$ [km/s]', fontsize = 28)
    axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
    axLadv.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
    axLkin.set_ylabel(r'$L_{\rm kin} [L_{\rm Edd}]$', fontsize = 28)
    fig.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 20)
    figM.suptitle(f't = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 20)
    fig.tight_layout()
    figM.tight_layout()
    # fig.savefig(f'{abspath}/Figs/paper/den_profShell{which_part}_{suffix_saveing}.pdf', bbox_inches = 'tight')
    # figM_dim.savefig(f'{abspath}/Figs/paper/MwShell{which_part}.pdf', bbox_inches = 'tight')
    # figL.savefig(f'{abspath}/Figs/paper/LShell{which_part}.pdf', bbox_inches = 'tight')
    plt.show()

