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
suffix_saveing = ''

# print('v_esc (r_p) = ', v_esc_kms, 'km/s')
#%%
# FUNCTIONS
#
def radial_profiles(loadpath, snap, which_part, ray_params):
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
    orb_en = orb.orbital_energy(Rsph, vel, Mass, params, prel.G)

    if which_part == 'outflow': 
        # cut_mid = np.abs(Z) > dim_cell #np.logical_and(X<-10*Rt, np.abs(Z) > 50) #dim_cell)
        cut_wind = np.logical_and(bern > 0, V_r > 0)
        cut = cut_wind #np.logical_and(cut_wind, ~cut_mid)
        # cut = np.logical_and(orb_en > 0, V_r >= 0)
    elif which_part == 'inflow':
        cut = np.logical_and(bern < 0, V_r < 0)
    X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den, dim_cell, orb_en, bern = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, vel, V_r, T, Press, IE_den, Rad_den, dim_cell, orb_en, bern], cut)       
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,7))
    img = ax1.scatter(X[np.abs(Z)<dim_cell]/Rp, Y[np.abs(Z)<dim_cell]/Rp, c = Mass[np.abs(Z)<dim_cell], s = 2, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-7))
    plt.colorbar(img, label = r'Mass $[M_\odot]$')
    img = ax2.scatter(X[np.abs(Z)<dim_cell]/Rp, Y[np.abs(Z)<dim_cell]/Rp, c = V_r[np.abs(Z)<dim_cell]/v_esc, s = 2, norm = colors.LogNorm(vmin=1e-2, vmax = 1), cmap = 'rainbow')
    cbar = plt.colorbar(img, label = r'v$_{\rm r} [v_{\rm esc}]$ ')
    cbar.ax.tick_params(axis = 'both', length = 9, width = 1.2)
    cbar.ax.tick_params(axis = 'both', which = 'minor', length = 7, width = .8)
    for ax in (ax1, ax2):
        ax.set_xlim(-120, 50) 
        ax.set_ylim(-50, 50)
        ax.set_xlabel(r'X [$r_p$]')
    ax1.set_ylabel(r'Y [$r_p$]')
    plt.suptitle(r'Wind material, t/t$_{fb}$ = ' + f'{np.round(tfb,2)}' , fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/next_meeting/section{snap}_{suffix_saveing}.png', bbox_inches = 'tight')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (25,8))
    img = ax1.scatter(Rsph[::50]/Rp, np.abs(Z)[::50]/Rp, c = Mass[::50], s = 1, norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-7), cmap = 'rainbow')
    cbar = plt.colorbar(img)
    cbar.set_label(r'Mass [$M_\odot$]')
    cbar.ax.tick_params(axis = 'both', length = 9, width = 1.2)
    img = ax2.scatter(Rsph[::50]/Rp, np.abs(Z)[::50]/Rp, c = V_r[::50]/v_esc, s = 1, norm = colors.LogNorm(vmin = 5e-2, vmax = 1), cmap = 'rainbow')
    cbar = plt.colorbar(img)
    cbar.set_label(r'$v_{\rm r} [v_{\rm esc}]$')
    cbar.ax.tick_params(axis = 'both', length = 9, width = 1.2)
    img = ax3.scatter(Rsph[::50]/Rp, np.abs(Z)[::50]/Rp, c = V_r[::50]*Mass[::50], s = 1, norm = colors.LogNorm(vmin = 1e-11, vmax = 1e-7), cmap = 'rainbow')
    cbar = plt.colorbar(img)
    cbar.set_label(r'$v_{\rm r}\cdot M $')
    cbar.ax.tick_params(axis = 'both', length = 9, width = 1.2)
    for ax in (ax1, ax2, ax3):
        ax.loglog()
        ax.set_xlim(1, 200)
        ax.set_ylim(1e-2, 200)
        ax.set_xlabel(r'$R [R_{\rm p}]$')
        ax.tick_params(axis = 'both', length = 9, width = 1.2)
        ax.tick_params(axis = 'both', which = 'minor', length = 7, width = .8)
    ax1.set_ylabel(r'$|Z| [R_p]$')
    plt.suptitle(f'Outflow material at t = {np.round(tfb, 2)}' + r'$t_{\rm fb}$', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/next_meeting/cellsDist{snap}_{suffix_saveing}.png', bbox_inches = 'tight')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))
    img = ax1.scatter(Rsph[::50]/Rp, np.abs(Z)[::50]/Rp, c = bern[::50], s = 1, vmin = -80, vmax = 80, cmap = 'coolwarm')
    cbar = plt.colorbar(img)
    cbar.set_label(r'B')
    cbar.ax.tick_params(axis = 'both', length = 9, width = 1.2)
    img = ax2.scatter(Rsph[::50]/Rp, np.abs(Z)[::50]/Rp, c = orb_en[::50]/Mass[::50], s = 1, vmin = -80, vmax = 80, cmap = 'coolwarm')
    cbar = plt.colorbar(img)
    cbar.set_label(r'specific oe')
    cbar.ax.tick_params(axis = 'both', length = 9, width = 1.2)
    for ax in (ax1, ax2):
        ax.loglog()
        ax.set_xlim(1, 200)
        ax.set_ylim(1e-2, 200)
        ax.set_xlabel(r'$R [R_{\rm p}]$')
        ax.tick_params(axis = 'both', length = 9, width = 1.2)
        ax.tick_params(axis = 'both', which = 'minor', length = 7, width = .8)
    ax1.set_ylabel(r'$|Z| [R_p]$')
    plt.suptitle(f'Outflow material at t = {np.round(tfb, 2)}' + r'$t_{\rm fb}$', fontsize = 20)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/next_meeting/cellsDist{snap}_{suffix_saveing}OeB.png', bbox_inches = 'tight')

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
        # find cells at r
        idx = np.abs(Rsph-r) < Vol**(1/3)
        ray_rad_den = Rad_den[idx]
        # if i % 10 == 0:
        #     print(f'{i}, number cells found {len(ray_rad_den)}', flush=True)
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
    fig, (axMdot_dim, axV, axd, axT, axL) = plt.subplots(5, 1, figsize=(8, 21)) 
    # figM_dim, axMdot_dim = plt.subplots(1, 1, figsize=(9, 7))
    # figL, axL = plt.subplots(1, 1, figsize=(9, 7))

for i, snap in enumerate(snaps):
    path = f'{pre}/{snap}'
    # Load data
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
    print(tfb)

    ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
    xph, yph, zph, Raddenph = ph_data[0], ph_data[1], ph_data[2], ph_data[6]
    rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
    rph = np.median(rph_all)

    dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
    x_tr, y_tr, z_tr, den_tr, vol_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr'], dataRtr['den_tr'], dataRtr['vol_tr'] 
    r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    mass_tr = den_tr * vol_tr
    nonzero = r_tr_all != 0
    r_tr = np.median(r_tr_all[nonzero]) if nonzero.any() else 0 #then at early times you have rtr > rph 

    if compute:
        all_outflows = radial_profiles(path, snap, which_part, [Rt, rph, 100])
        out_path = f"{abspath}/data/{folder}/wind/den_prof{snap}Shell{which_part}_{suffix_saveing}.npy"
        np.save(out_path, all_outflows, allow_pickle=True)

    if plot:
        profiles = np.load(f'{abspath}/data/{folder}/wind/den_prof{snap}Shell{which_part}_{suffix_saveing}.npy', allow_pickle=True).item()
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
        t_dyn = r_plot / v_rad 
        t = profiles['t_mean']
        L_adv = profiles['L_adv_mean']
        Mdot = profiles['Mdot_mean'] 
        Mdot_dimcell = profiles['Mdot_mean_dimcell']
        idx_rtr = np.argmin(np.abs(r_plot - r_tr))

        v_term = v_esc_kms * np.sqrt(1-(r_plot/Rp)**(-2/3))

        axV_tot.plot(r_plot/Rt, v_tot * conversion_sol_kms,  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        axV.plot(r_plot/Rt, v_rad * conversion_sol_kms,  color = colors_snaps[i], label = f't = {tfb:.1f} ' + r'$t_{\rm fb}$')
        # axV.plot(r_plot/Rt, v_term, c = 'k', ls = 'dashed') #, label = r'v$_{\rm term}$')
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

            print('at time ', np.round(tfb, 2))
            print('(t_ph/T_tr)', (t[-1]/t[idx_rtr]))
            print('(t_ph/T_tr)^4', (t[-1]/t[idx_rtr])**4)
            print('(r_ph/r_tr)^2', (rph/r_tr)**2)
            print('v(rtr) km/s', v_rad[idx_rtr] * conversion_sol_kms)
            print('L(rph)/L(rtr) from data= ', L_adv[-1]/L_adv[idx_rtr])
            print('estimated with escape velocity', 2 * 2/3 * (rph/r_tr)**2 * (t[-1]/t[idx_rtr])**4 * prel.csol_cgs/v_esc)#v_rad[idx_rtr])
            # from eq. 17 in the paper
            v_radiat2 = 20 * (prel.c_cgs*1e-5)**2 * Medd_sol/Mdot_dimcell[idx_rtr] #in km/s
            print(v_radiat2)
            print('v_esc-radiation contribution ', np.sqrt(v_esc_kms**2-v_radiat2))

if plot:
    axd.plot(x_test, y_test2, c = 'k', ls = 'dashed') #, label = r'$\rho \propto r^{-2}$')
    axd.text(35, 1.1e-11, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
    # axV.axhline(0.2*v_esc_kms, c = 'k', ls = 'dashed')# 
    # axV.text(35, 1.1*0.2*v_esc_kms, r'0.2v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    axT.plot(x_test, y_test23, c = 'k', ls = 'dashed') #, label = r'$T \propto r^{-2/3}$')
    axT.text(1.2, 2.4e5, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -24)
    axL.plot(x_test, 2e-4*y_test23, c = 'k', ls = 'dashed') #, label = r'$L \propto r^{-2/3}$')
    axL.text(1.2, 5.6e1, r'$L \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -18)

    for ax in [axV_tot, axd, axV, axMdot_dim, axT, axL]:
        ax.tick_params(axis='both', which='minor', length=6, width=1)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.loglog()
        ax.set_xlim(1, 4*apo/Rt)
        ax.grid()
        if ax == axMdot_dim:
            ax.legend(fontsize = 18)

    axL.set_xlabel(r'$r [r_{\rm t}]$', fontsize = 28)
    # axMdot.set_ylim(7e2, 1e7)
    # axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28) 
    # axMdot.set_title(r'Mean over spherical shells: $4\pi r^2\langle \rho v_r \rangle_{\rm wind \,cells}$', fontsize = 20)
    axMdot_dim.set_ylim(7e2, 5e6)
    axMdot_dim.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28)
    # axMdot_dim.set_title(r'Mean over spherical shells weighted by cell dimension: $\big(4 r^2/\sum s^2\big) \sum \pi s^2 \rho v_r$', fontsize = 16)
    axd.set_ylim(1e-11, 4e-8)
    axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
    axV_tot.set_ylim(2e3, 3e4)
    axV_tot.set_ylabel(r'$|v|$ [km/s]', fontsize = 28)
    axV.set_ylim(2e3, 3e4)
    axV.set_ylabel(r'v$_{\rm r}$ [km/s]', fontsize = 28)
    axT.set_ylim(1e4, 1e6)
    axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
    axL.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
    axL.set_ylim(1, 5e2)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/paper/den_profShell{which_part}_{suffix_saveing}.pdf', bbox_inches = 'tight')
    # figM_dim.savefig(f'{abspath}/Figs/paper/MwShell{which_part}.pdf', bbox_inches = 'tight')
    # figL.savefig(f'{abspath}/Figs/paper/LShell{which_part}.pdf', bbox_inches = 'tight')
    plt.show()


# %% compute constant wind
A_w_cgs = np.sqrt(prel.G_cgs) * (np.sqrt(2)/(3*np.pi))**(1/3) * (prel.Msol_cgs*1e4)**(1/18) * (prel.Msol_cgs)**(7/9) * (prel.Rsol_cgs)**(-5/6)
print('constant A wind (without wind efficiency)/Ledd^1/3:', A_w_cgs/Ledd_cgs**(1/3)) 
print('Ledd^1/3, :', 1e-13*Ledd_cgs**(1/3), '1e13')
A_w_cgs_eff = (np.sqrt(2*prel.G_cgs**5)/(3*np.pi* prel.c_cgs**2))**(1/3) * (prel.Msol_cgs )**(8/9) * (prel.Msol_cgs*1e4)**(5/18)  / (prel.Rsol_cgs)**(7/6) 
print('constant A wind (with wind efficiency rg/rp)/Ledd^1/3:', A_w_cgs_eff/Ledd_cgs**(1/3))
# %%
