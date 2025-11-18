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
from Utilities.operators import to_spherical_components, make_tree

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
which_part = 'outflow'
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
t_fb_days = things['t_fb_days']
t_fb_days_cgs = t_fb_days * 24 * 3600 # in seconds
t_fb_sol = t_fb_days_cgs/prel.tsol_cgs
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs

#
# FUNCTIONS
#
def radial_profiles(loadpath, snap, which_part, ray_params):
    data = make_tree(loadpath, snap, energy = True)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)  
    V_r, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    bern = orb.bern_coeff(Rsph, vel, Den, Mass, Press, IE_den, Rad_den, params)
    orb_en = orb.orbital_energy(Rsph, vel, Mass, params, prel.G)

    if which_part == 'outflow':
        cut = np.logical_and(Den > 1e-19, np.logical_and(bern > 0, V_r >= 0)) 
        # cut = np.logical_and(Den > 1e-19, np.logical_and(orb_en > 0, V_r >= 0))
    elif which_part == 'inflow':
        cut = np.logical_and(Den > 1e-19, np.logical_and(bern < 0, V_r < 0)) 
    X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den], cut)       

    rmin, rmax, Nray = ray_params
    r_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
   
    t_mean = np.zeros(Nray)
    v_mean = np.zeros(Nray)
    v_rad_mean = np.zeros(Nray)
    d_mean = np.zeros(Nray)
    L_adv_mean = np.zeros(Nray)
    Mdot_mean = np.zeros(Nray)
    Mdot_mean_dimcell = np.zeros(Nray)
    for i, r in enumerate(r_array): 
        if i % 100 == 0:
            print(f'{i}', flush=True)
        # find cells at r
        idx = np.abs(Rsph-r) < Vol**(1/3)
        ray_rad_den = Rad_den[idx]
        ray_V = vel[idx]
        ray_V_r = V_r[idx] 
        ray_d = Den[idx] 
        ray_m = Mass[idx]
        ray_vol = Vol[idx]
        ray_dim = ray_vol**(1/3)
        ray_t = (ray_rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4) 
        ray_Mdot = np.abs(ray_V_r) * ray_d 
        L_adv =  np.abs(ray_V_r) * ray_rad_den

        # cond = ray_m >= 0 
        # t_mean[i] = np.mean(nonzero) if nonzero.size > 0 else 0 
        t_mean[i] = np.sum(ray_t*ray_vol)/np.sum(ray_vol)
        v_mean[i] = np.sum(ray_V*ray_m)/np.sum(ray_m)
        v_rad_mean[i] = np.sum(ray_V_r*ray_m)/np.sum(ray_m)
        d_mean[i] = np.sum(ray_d*ray_m)/np.sum(ray_m)
        L_adv_mean[i] = 4 * np.pi * r**2 * np.sum(L_adv*ray_m)/np.sum(ray_m)

        Mdot_mean[i] = 4 * np.pi * r**2 * np.mean(ray_Mdot) if ray_Mdot.size > 0 else 0 
        Mdot_mean_dimcell[i] = 4 * r**2 /np.sum(ray_dim**2) * np.pi * np.sum(ray_dim**2 * ray_Mdot)

    outflow = {
        'r': r_array,
        't_mean': t_mean,
        'v_mean': v_mean,
        'v_rad_mean': v_rad_mean,
        'd_mean': d_mean,
        'L_adv_mean': L_adv_mean,
        'Mdot_mean': Mdot_mean,
        'Mdot_mean_dimcell': Mdot_mean_dimcell
    }
    return outflow

#
## MAIN
#
snaps = [76, 109, 151]
colors_snaps = ['plum', 'magenta', 'darkviolet']

if plot:
    x_test = np.arange(1., 300)
    y_testplus1 = 3.5e3* (x_test)
    y_test1 = 9e4*(x_test)**(-1)
    y_test23 = 5e5*(x_test)**(-2/3)
    y_test2 = 3e-8* (x_test)**(-2) 
    figV, axV_tot = plt.subplots(1, 1, figsize=(9, 7))
    fig, (axMdot_dim, axV, axd, axT) = plt.subplots(1, 4, figsize=(32, 6)) 
    # figM_dim, axMdot_dim = plt.subplots(1, 1, figsize=(9, 7))
    figL, axL = plt.subplots(1, 1, figsize=(9, 7))

for i, snap in enumerate(snaps):
    path = f'{pre}/{snap}'
    # Load data
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
    print(tfb)

    ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    xph, yph, zph = ph_data[0], ph_data[1], ph_data[2]
    rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
    rph = np.median(rph_all)

    dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
    x_tr, y_tr, z_tr, den_tr, vol_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['vol_tr'] 
    r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    mass_tr = den_tr * vol_tr
    nonzero = r_tr_all != 0
    r_tr = np.median(r_tr_all[nonzero]) if nonzero.any() else 0 #then at early times you have rtr > rph 

    if compute:
        all_outflows = radial_profiles(path, snap, which_part, [Rt, rph, 1000])
        out_path = f"{abspath}/data/{folder}/wind/den_prof{snap}Shell{which_part}_weighmean.npy"
        np.save(out_path, all_outflows, allow_pickle=True)

    if plot:
        profiles = np.load(f'{abspath}/data/{folder}/wind/den_prof{snap}Shell{which_part}_weighmean.npy', allow_pickle=True).item()
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
        d = profiles['d_mean']
        v_tot = profiles['v_mean']
        v_rad = profiles['v_rad_mean'] 
        t = profiles['t_mean']
        L_adv = profiles['L_adv_mean']
        Mdot = profiles['Mdot_mean'] 
        Mdot_dimcell = profiles['Mdot_mean_dimcell']
        idx_rtr = np.argmin(np.abs(r_plot - r_tr))

        axV_tot.plot(r_plot/Rt, v_tot * conversion_sol_kms,  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        axV.plot(r_plot/Rt, v_rad * conversion_sol_kms,  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        axd.plot(r_plot/Rt, d*prel.den_converter,  color = colors_snaps[i])
        axT.plot(r_plot/Rt, t,  color = colors_snaps[i])
        # axMdot.plot(r_plot/Rt, np.abs(Mdot/Medd_sol),  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        # axMdot.scatter(r_tr/Rt, np.abs(Mdot[idx_rtr]/Medd_sol), color = colors_snaps[i], s = 100, marker = 'd')
        axMdot_dim.plot(r_plot/Rt, np.abs(Mdot_dimcell/Medd_sol),  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        axL.plot(r_plot/Rt, L_adv/Ledd_sol,  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        if snap != 76:
            axV.scatter(r_tr/Rt, v_rad[idx_rtr] * conversion_sol_kms, color = colors_snaps[i], s = 100, marker = 'd')
            axd.scatter(r_tr/Rt, d[idx_rtr]*prel.den_converter, color = colors_snaps[i], s = 100, marker = 'd')
            axT.scatter(r_tr/Rt, t[idx_rtr], color = colors_snaps[i], s = 100, marker = 'd')
            axMdot_dim.scatter(r_tr/Rt, np.abs(Mdot_dimcell[idx_rtr]/Medd_sol), color = colors_snaps[i], s = 100, marker = 'd')
            axL.scatter(r_tr/Rt, L_adv[idx_rtr]/Ledd_sol, color = colors_snaps[i], s = 100, marker = 'd')

if plot:
    axd.plot(x_test, y_test2, c = 'k', ls = 'dashed') #, label = r'$\rho \propto r^{-2}$')
    axd.text(35, 1.1e-11, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
    axV_tot.axhline(v_esc_kms, c = 'k', ls = 'dashed')#
    axV_tot.text(35, 1.1*v_esc_kms, r'v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    axT.plot(x_test, y_test23, c = 'k', ls = 'dashed') #, label = r'$T \propto r^{-2/3}$')
    axT.text(1.2, 2.8e5, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -38)
    axL.plot(x_test, 5e-5*y_test23, c = 'k', ls = 'dashed') #, label = r'$L \propto r^{-2/3}$')

    for ax in [axV_tot, axd, axV, axMdot_dim, axT, axL]:
        ax.tick_params(axis='both', which='minor', length=6, width=1)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.loglog()
        ax.set_xlim(1, 4*apo/Rt)
        ax.set_xlabel(r'$r [r_{\rm t}]$', fontsize = 28)
        ax.grid()
        if ax not in [axd, axT, axV]:
            ax.legend(fontsize = 20)

    # axMdot.set_ylim(7e2, 1e7)
    # axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28) 
    # axMdot.set_title(r'Mean over spherical shells: $4\pi r^2\langle \rho v_r \rangle_{\rm wind \,cells}$', fontsize = 20)
    axMdot_dim.set_ylim(7e2, 1e7)
    axMdot_dim.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28)
    # axMdot_dim.set_title(r'Mean over spherical shells weighted by cell dimension: $\big(4 r^2/\sum s^2\big) \sum \pi s^2 \rho v_r$', fontsize = 16)
    axd.set_ylim(1e-11, 4e-8)
    axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
    axV_tot.set_ylim(2e3, 3e4)
    axV_tot.set_ylabel(r'$|v|$ [km/s]', fontsize = 28)
    axV.set_ylim(2e3, 3e4)
    axV.set_ylabel(r'v$_{\rm r}$ [km/s]', fontsize = 28)
    axT.set_ylim(4e4, 1e6)
    axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
    axL.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
    axL.set_ylim(1, 5e2)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/paper/den_profShell{which_part}.pdf', bbox_inches = 'tight')
    # figM_dim.savefig(f'{abspath}/Figs/paper/MwShell{which_part}.pdf', bbox_inches = 'tight')
    figL.savefig(f'{abspath}/Figs/paper/LShell{which_part}.pdf', bbox_inches = 'tight')
    plt.show()


    # %%
