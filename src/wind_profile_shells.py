""" Find density and (radial velocity) profiles for different lines of sight."""
# from Mdot_Rfixed import Medd_code
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
snap = 109
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

    if which_part == 'outflow':
        cut = np.logical_and(Den > 1e-19, np.logical_and(bern > 0, V_r >= 0)) 
    elif which_part == 'inflow':
        cut = np.logical_and(Den > 1e-19, np.logical_and(bern < 0, V_r < 0)) 
    X, Y, Z, Rsph, Vol, Den, Mass, VX, VY, VZ, V_r, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Rsph, Vol, Den, Mass, VX, VY, VZ, V_r, T, Press, IE_den, Rad_den], cut)       

    rmin, rmax, Nray = ray_params
    r_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
   
    t_mean = np.zeros(Nray)
    v_rad_mean = np.zeros(Nray)
    d_mean = np.zeros(Nray)
    L_adv_mean = np.zeros(Nray)
    Mdot_mean = np.zeros(Nray)
    for i, r in enumerate(r_array): 
        if i % 100 == 0:
            print(f'{i}', flush=True)
        # find cells at r
        idx = np.abs(Rsph-r) < Vol**(1/3)
        ray_rad_den = Rad_den[idx]
        ray_V_r = V_r[idx] 
        ray_d = Den[idx] 
        ray_m = Mass[idx]
        ray_t = (ray_rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4) 
        ray_Mdot = 4 * np.pi * r**2 * np.abs(ray_V_r) * ray_d 
        L_adv = 4 * np.pi * r**2 * np.abs(ray_V_r) * ray_rad_den

        cond = ray_m >= 0 
        nonzero = ray_t[cond]
        t_mean[i] = np.mean(nonzero) if nonzero.size > 0 else 0
        # t_mean[i] = np.sum(ray_t*ray_m)/np.sum(ray_m)
        nonzero = ray_V_r[cond]
        v_rad_mean[i] = np.mean(nonzero) if nonzero.size > 0 else 0
        # v_rad_mean[i] = np.sum(ray_V_r*ray_m)/np.sum(ray_m)
        nonzero = ray_d[cond]
        d_mean[i] = np.mean(nonzero) if nonzero.size > 0 else 0
        # d_mean[i] = np.sum(ray_d*ray_m)/np.sum(ray_m)
        nonzero = L_adv[cond]
        L_adv_mean[i] = np.mean(nonzero) if nonzero.size > 0 else 0
        # L_adv_mean[i] = np.sum(L_adv*ray_m)/np.sum(ray_m)
 
        Mdot_mean[i] = np.mean(ray_Mdot) if ray_Mdot.size > 0 else 0
        # Mdot_mean[i] = np.sum(ray_Mdot*ray_m)/np.sum(ray_m)

    outflow = {
        'r': r_array,
        't_mean': t_mean,
        'v_rad_mean': v_rad_mean,
        'd_mean': d_mean,
        'L_adv_mean': L_adv_mean,
        'Mdot_mean': Mdot_mean
    }
    return outflow

#
## MAIN
#
path = f'{pre}/{snap}'
# Load data
tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
_, tfb_fall, mfall, mwind_dimCell, mwind_R, mwind_R_nonzero, Vwind, Vwind_nonzero = \
    np.loadtxt(f'{abspath}/data/{folder}/wind/Mdot_{check}05aminmean.csv', 
                delimiter = ',', 
                skiprows=1, 
                unpack=True) 

ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
xph, yph, zph = ph_data[0], ph_data[1], ph_data[2]
denph = ph_data[4]
alphaph, Lum_ph = ph_data[-4], ph_data[-2]
kappaph = alphaph/denph
rph_all = np.sqrt(xph**2 + yph**2 + zph**2)
rph = np.median(rph_all)

dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
x_tr, y_tr, z_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr']
r_tr_all = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
nonzero = r_tr_all != 0
r_tr = np.median(r_tr_all[nonzero]) if nonzero.any() else 0

if compute:
    all_outflows = radial_profiles(path, snap, which_part, [Rt, rph, 1000])
    out_path = f"{abspath}/data/{folder}/wind/den_prof{snap}Shell{which_part}_mean.npy"
    np.save(out_path, all_outflows, allow_pickle=True)

if plot:
    profiles = np.load(f'{abspath}/data/{folder}/wind/den_prof{snap}Shell{which_part}_mean.npy', allow_pickle=True).item()

    r_plot = profiles['r']
    d = profiles['d_mean']
    v_rad_tr = profiles['v_rad_mean'][np.argmin(np.abs(r_plot-r_tr))]
    Mdot = profiles['Mdot_mean'][np.argmin(np.abs(r_plot-r_tr))]
        
    t_dyn = (r_tr/v_rad_tr)*prel.tsol_cgs/t_fb_days_cgs # you want it in t_fb
    tfb_adjusted = tfb - t_dyn
    find_time = np.argmin(np.abs(tfb_fall-tfb_adjusted))
    mfall_t = mfall[find_time]
    eta = np.abs(Mdot/mfall_t) 
    print(eta)

    # Radial profiles
    # R_edge = v_esc / prel.tsol_cgs * tfb * t_fb_days_cgs
    x_test = np.arange(1., 300)
    y_testplus1 = 3.5e3* (x_test)
    y_test1 = 9e4*(x_test)**(-1)
    y_test23 = 5e5*(x_test)**(-2/3)
    y_test2 = 1e-8* (x_test)**(-2) 

    fig, (axV, axd, axT) = plt.subplots(1, 3, figsize=(24, 6)) 
    figM, axMdot = plt.subplots(1, 1, figsize=(10, 6))
    figL, axL = plt.subplots(1, 1, figsize=(10, 6))

    r_plot = profiles['r'] 
    d = profiles['d_mean']
    v_rad = profiles['v_rad_mean'] 
    t = profiles['t_mean']
    L_adv = profiles['L_adv_mean']
    Mdot = profiles['Mdot_mean'] 

    axd.plot(r_plot/Rt, d*prel.den_converter,  color = 'darkviolet') 
    axV.plot(r_plot/Rt, v_rad * conversion_sol_kms,  color = 'darkviolet')
    axMdot.plot(r_plot/Rt, np.abs(Mdot/Medd_sol),  color = 'darkviolet')
    axT.plot(r_plot/Rt, t,  color = 'darkviolet')
    axL.plot(r_plot/Rt, L_adv/Ledd_sol,  color = 'darkviolet')

    axd.plot(x_test, y_test2, c = 'k', ls = 'dashed', label = r'$\rho \propto r^{-2}$')
    # axd.text(35, 2e-11, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
    axV.axhline(v_esc_kms, c = 'k', ls = 'dashed')#
    axV.text(35, 1.1*v_esc_kms, r'v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    axT.plot(x_test, y_test23, c = 'k', ls = 'dashed', label = r'$T \propto r^{-2/3}$')
    # axT.text(27, 4.2e4, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -33)
    axL.plot(x_test, 5e-5*y_test23, c = 'k', ls = 'dashed', label = r'$L \propto r^{-2/3}$')

    for ax in [axd, axV, axMdot, axT, axL]:
        ax.axvline(r_tr/Rt, c = 'darkviolet', ls = ':')
        ax.tick_params(axis='both', which='minor', length=6, width=1)
        ax.tick_params(axis='both', which='major', length=10, width=1.5)
        ax.loglog()
        ax.set_xlim(1, 3*apo/Rp)
        ax.set_xlabel(r'$r [r_{\rm t}]$', fontsize = 28)
        ax.grid()

    axMdot.set_ylim(1e4, 1e7)
    axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28) 
    axd.set_ylim(1e-11, 4e-8)
    axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
    axV.set_ylim(2e3, 3e4)
    axV.set_ylabel(r'v$_{\rm r}$ [km/s]', fontsize = 28)
    axT.set_ylim(7e4, 1e6)
    axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
    axL.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
    axL.set_ylim(1, 1e2)
    # fig.suptitle(f'{check}, t = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
    # figT.suptitle(f'{check}, t = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/paper/den_prof{snap}Shell{which_part}.pdf', bbox_inches = 'tight')
    figM.savefig(f'{abspath}/Figs/paper/Mw{snap}Shell{which_part}.pdf', bbox_inches = 'tight')
    figL.savefig(f'{abspath}/Figs/paper/L{snap}Shell{which_part}.pdf', bbox_inches = 'tight')
    plt.show()


# %%
