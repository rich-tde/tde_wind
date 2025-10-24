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
from sklearn.neighbors import KDTree
import healpy as hp
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix, select_snap
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import to_spherical_components, make_tree, choose_observers

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
which_obs = '' #'dark_bright_z' # 'arch', 'quadrants', 'axis'
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

rmin_array = Rt

#
# FUNCTIONS
#
def CouBegel(r, theta, q, gamma=4/3):
    # I'm missing the normalization
    alpha = (1-q*(gamma-1))/(gamma-1)
    rho = r**(-q) * np.sin(theta)**(2*alpha)
    return rho

def Metzger(r, q, Mbh, mstar, Rstar, k = 0.9, Me = 1, G = prel.G):
    # I'm missing the correct noramlization
    Rv = 2 * Rstar/(5*k) * (Mbh/mstar)**(2/3) * Me/mstar
    norm = Me*(3-q)/(4*np.pi*Rv**3 * (7-2*q))
    if norm == 0.0:
        print('norm is 0')
    if r < Rv:
        rho = (r/Rv)**(-q)
    else:
        rho = np.exp(-(r-Rv)/Rv)
    return (norm * rho)

def radial_profiles(loadpath, snap, which_part, indices_sorted):
    data = make_tree(loadpath, snap, energy = True)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    R = np.sqrt(X**2 + Y**2 + Z**2)  
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)  
    V_r, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    bern = orb.bern_coeff(R, vel, Den, Mass, Press, IE_den, Rad_den, params)

    if which_part == 'outflow':
        cut = np.logical_and(Den > 1e-19, np.logical_and(bern > 0, V_r >= 0)) 
    elif which_part == 'inflow':
        cut = np.logical_and(Den > 1e-19, np.logical_and(bern < 0, V_r < 0)) 
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, V_r, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, V_r, T, Press, IE_den, Rad_den], cut)       

    xyz = np.array([X, Y, Z]).T
    tree = KDTree(xyz, leaf_size = 50) 
    N_ray = 5_000

    all_outflows = {}
    for j, idx_list in enumerate(indices_sorted):
        print(f'Region: {j}', flush= True)
        t_all = []
        v_rad_all = []
        d_all = []
        m_all = []
        L_adv_all = []
        Mdot_all = []
        r = np.logspace(np.log10(rmin_array), np.log10(rph_median[j]), N_ray)
        for i in idx_list: # i in [0, 192]: pick the line of sight that you'll use for the mean of the chosen direction
            print(f'Obs {i}', flush= True)
            mu_x = observers_xyz[i][0]
            mu_y = observers_xyz[i][1]
            mu_z = observers_xyz[i][2]

            x = r*mu_x
            y = r*mu_y
            z = r*mu_z
            xyz2 = np.array([x, y, z]).T
            # radii2 = np.sqrt(x**2 + y**2 + z**2)
            del x, y, z

            # ray tracing along observer i 
            dist, idx = tree.query(xyz2, k=1) #you can do k=4, comment the line afterwards and then do d = np.mean(d, axis=1) and the same for all the quantity. But you doesn't really change since you are averaging already on line of sights
            dist = np.concatenate(dist)
            idx = np.array([ int(idx[i][0]) for i in range(len(idx))])
            
            # Quantity corresponding to the ray
            ray_rad_den = Rad_den[idx]
            ray_V_r = V_r[idx] 
            d = Den[idx] 
            ray_m = Mass[idx]
            
            # pick them just if near enough, otherwise set to 0 (easier for saving, insted of discard them)
            r_sim = np.sqrt(X[idx]**2 + Y[idx]**2 + Z[idx]**2)
            check_dist = np.logical_and(dist <= Vol[idx]**(1/3), r_sim >= rmin_array)
            ray_V_r[~check_dist] = 0
            d[~check_dist] = 0 
            ray_rad_den[~check_dist] = 0
            ray_t = (ray_rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4) 
            ray_m[~check_dist] = 0
            ray_Mdot = 4 * np.pi * r_sim**2 * np.abs(ray_V_r) * d 
            L_adv = 4 * np.pi * r_sim**2 * np.abs(ray_V_r) * ray_rad_den

            # store
            t_all.append(ray_t)
            v_rad_all.append(ray_V_r)
            d_all.append(d) 
            m_all.append(ray_m)
            L_adv_all.append(L_adv)
            Mdot_all.append(ray_Mdot)

        t_median = []
        v_rad_median = []
        d_mean = []
        L_adv_median = []
        Mdot_mean = []
        for i in range(len(np.transpose(t_all))):
            t_col = np.transpose(t_all)[i]
            v_rad_col = np.transpose(v_rad_all)[i]
            d_col = np.transpose(d_all)[i]
            L_adv_col = np.transpose(L_adv_all)[i]
            Mdot_col = np.transpose(Mdot_all)[i]
            ray_m_col = np.transpose(m_all)[i]
            cond = np.logical_and(t_col != 0, ray_m_col != 0)
            nonzero = t_col[cond]
            # t_median.append(np.median(nonzero) if nonzero.size > 0 else 0)
            t_median.append(np.sum(nonzero*ray_m_col[cond])/np.sum(ray_m_col[cond]))
            cond = np.logical_and(v_rad_col != 0, ray_m_col != 0)
            nonzero = v_rad_col[cond]
            # v_rad_median.append(np.median(nonzero) if nonzero.size > 0 else 0)
            v_rad_median.append(np.sum(nonzero*ray_m_col[cond])/np.sum(ray_m_col[cond]))
            cond = np.logical_and(d_col != 0, ray_m_col != 0)
            nonzero = d_col[cond]
            # d_mean.append(np.median(nonzero) if nonzero.size > 0 else 0) 
            d_mean.append(np.sum(nonzero*ray_m_col[cond])/np.sum(ray_m_col[cond]))
            cond = np.logical_and(L_adv_col != 0, ray_m_col != 0)
            nonzero = L_adv_col[cond]
            # L_adv_median.append(np.median(nonzero) if nonzero.size > 0 else 0)
            L_adv_median.append(np.sum(nonzero*ray_m_col[cond])/np.sum(ray_m_col[cond]))
 
            Mdot_mean.append(np.mean(Mdot_col) if Mdot_col.size > 0 else 0)
        t_median = np.array(t_median)
        v_rad_median = np.array(v_rad_median)
        d_mean = np.array(d_mean)
        L_adv_median = np.array(L_adv_median)
        Mdot_mean = np.array(Mdot_mean)

        outflow = {
            'r': r,
            't_median': t_median,
            'v_rad_median': v_rad_median,
            'd_mean': d_mean,
            'L_adv_median': L_adv_median,
            'Mdot_mean': Mdot_mean
        }
        key = f"{label_obs[j]}"
        all_outflows[key] = outflow   # dict of dicts

    return all_outflows

#
## MAIN
#

# Observers
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, which_obs)
observers_xyz = np.transpose(observers_xyz) #shape: Nx3

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
else:
    snaps = [109]

for snap in snaps:
    print(snap, flush=True)
    if alice:
        path = f'{pre}/snap_{snap}'
    else:
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
    rph = np.sqrt(xph**2 + yph**2 + zph**2)
    rph_median = np.zeros(len(indices_sorted))
    Lum_ph_mean = np.zeros(len(indices_sorted)) 

    dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}.npz")
    x_tr, y_tr, z_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr']
    r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
    r_tr_median = np.zeros(len(indices_sorted))
    for i in range(len(indices_sorted)):  
        nonzero = r_tr[indices_sorted[i]] != 0
        if nonzero.any():
            print(f'{label_obs[i]}: no Rtr in {np.sum(~nonzero)/len(r_tr[indices_sorted[i]])*100:.2f}%')
            r_tr_median[i] = np.median(r_tr[indices_sorted[i]][nonzero])
        rph_median[i] = np.median(rph[indices_sorted[i]])
        Lum_ph_mean[i] = np.mean(Lum_ph[indices_sorted[i]]) # in this case we do the mean since it's what we do for the FLD

    if plot: 
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for i, idx_list in enumerate(indices_sorted):
            nonzero = r_tr[indices_sorted[i]] != 0
            ax1.scatter(x_obs[indices_sorted[i]][nonzero], y_obs[indices_sorted[i]][nonzero], color = colors_obs[i], linewidths = 1)
            ax1.scatter(x_obs[idx_list], y_obs[idx_list], facecolor = 'none', edgecolors = 'k', linewidths = 1)
            ax2.scatter(x_obs[indices_sorted[i]][nonzero], z_obs[indices_sorted[i]][nonzero], color = colors_obs[i], linewidths = 1, label = r'r$_{\rm tr}\neq0$' if i == 0 else '')
            ax2.scatter(x_obs[idx_list], z_obs[idx_list], facecolor = 'none', edgecolors = 'k', linewidths = 1)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f'{abspath}/Figs/next_meeting/{check}/observers_{snap}.png') 

        # make a 3d scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d') 
        # for i, idx_list in enumerate(indices_sorted):
        #     ax.scatter(x_obs, y_obs, z_obs, color = colors_obs[i], linewidths = 1)
        # plt.show()

    if compute:
        all_outflows = radial_profiles(path, snap, which_part, indices_sorted)
        out_path = f"{abspath}/data/{folder}/wind/den_prof{snap}{which_obs}{which_part}_weighmeanNOZERO.npy"
        np.save(out_path, all_outflows, allow_pickle=True)
 
    if plot:
        x_as_thetas = True
        thetas_tick = [-np.pi, 0, np.pi]
        thetas_tick_labels = [r'$-\pi$', r'$0$', r'$\pi$']
        profiles = np.load(f'{abspath}/data/{folder}/wind/den_prof{snap}{which_obs}{which_part}_medianNOZERO.npy', allow_pickle=True).item()
        # find eta = mfall(t_fb-t_dyn)/Mwind(tfb)
        figeta, axeta = plt.subplots(1, 1, figsize=(9, 6))
        fig, (axEdd, axL) = plt.subplots(2, 1, figsize=(9, 12))
        mfall_mean = np.zeros(len(profiles))
        for i, lab in enumerate(profiles.keys()):
            if lab in [r'$-\hat{\textbf{z}}$']:
                continue
            r_tr = r_tr_median[i]
            rph_single = rph_median[i]
            Lum_ph_single = Lum_ph_mean[i]
            r_plot = profiles[lab]['r']
            d = profiles[lab]['d_mean']
            v_rad_tr = profiles[lab]['v_rad_median'][np.argmin(np.abs(r_plot-r_tr))]
            Mdot = profiles[lab]['Mdot_mean'][np.argmin(np.abs(r_plot-r_tr))]
            
            t_dyn = (r_tr/v_rad_tr)*prel.tsol_cgs/t_fb_days_cgs # you want it in t_fb
            print(f'{lab}: t_dyn/t_fb', np.round(t_dyn, 3))
            tfb_adjusted = tfb - t_dyn
            find_time = np.argmin(np.abs(tfb_fall-tfb_adjusted))
            mfall_mean[i] = mfall[find_time]
            eta = np.abs(Mdot/mfall_mean[i])

            if x_as_thetas:
                axeta.scatter(thetas_tick[i], eta, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
                axEdd.scatter(thetas_tick[i], Mdot/Medd_sol, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
                axL.scatter(thetas_tick[i], Lum_ph_single/Ledd_cgs, c = colors_obs[i], label = f'{label_obs[i]}', ls = lines_obs[i], s = 100)
            else:
                axeta.scatter(r_tr/Rt, eta, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
                axEdd.scatter(r_tr/Rt, Mdot/Medd_sol, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
                axL.scatter(rph_single/Rt, Lum_ph_single/Ledd_cgs, c = colors_obs[i], label = f'{label_obs[i]}', ls = lines_obs[i], s = 100)

            print((Mdot/prel.tsol_cgs)*3600*24*365)

        for axis in [axeta, axEdd, axL]: 
            if x_as_thetas:
                axis.set_xlabel(r'$\theta$')
                axis.set_xticks(thetas_tick) 
                axis.set_xticklabels(thetas_tick_labels)
                axis.set_yscale('log')  
                axEdd.legend(fontsize = 14)
            else: 
                if axis != axL:
                    axis.set_xlabel(r'$r_{\rm tr} [r_{\rm t}]$')
                    axis.legend(fontsize = 14)
                else:
                    axis.set_xlabel(r'$r_{\rm ph} [r_{\rm t}]$')
                axis.set_xlim(1, 150)
                axis.set_ylim(1e-5, 1e-1)
                axis.loglog()
            axis.tick_params(axis='both', which='minor', length=5, width=1)
            axis.tick_params(axis='both', which='major', length=8, width=1.2)
            axis.grid()
        axeta.set_ylabel(r'$\eta = |\dot{M}_{\rm w}/\dot{M}_{\rm fb}|$')
        axeta.set_ylim(1e-5, 1e-1)
        axL.set_ylabel(r'$L_{\rm ph} [L_{\rm Edd}]$')
        axL.set_ylim(2e-1, 10)
        axEdd.set_ylabel(r'$|\dot{M}_{\rm w}| [\dot{M}_{\rm Edd}]$')
        axEdd.set_ylim(3, 6e2)
        fig.suptitle(f't = {np.round(tfb, 2)}' + r' t$_{\rm fb}$', fontsize = 20)
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/next_meeting/{check}/MdotL{snap}{which_part}.png', bbox_inches = 'tight')

        #%% Radial profiles
        # R_edge = v_esc / prel.tsol_cgs * tfb * t_fb_days_cgs
        x_test = np.arange(1., 300)
        y_testplus1 = 3.5e3* (x_test)
        y_test1 = 9e4*(x_test)**(-1)
        y_test12 = 0.75*(x_test)**(-0.5)
        y_test02 = 5e3* (x_test)**(-0.2)
        y_test08 = 4.2e-12* (x_test)**(-0.8)
        y_test23 = 3e5*(x_test)**(-2/3)
        y_test2 = 3e-11* (x_test)**(-2) 
        y_testplus2 = 2.5e3* (x_test)**(2)
        y_test3 = 6.5e-13 * (x_test)**(-3)

        normalize_by = 'Rt'
        fig, (axV, axd, axT) = plt.subplots(1, 3, figsize=(24, 6)) 
        figM, axMdot = plt.subplots(1, 1, figsize=(10, 6))
        figL, axL = plt.subplots(1, 1, figsize=(10, 6))

        for i, lab in enumerate(profiles.keys()):
            if lab in [r'$-\hat{\textbf{z}}$']:
                continue

            r_normalizer = Rt if normalize_by == 'Rt' else r_tr_median[i]
            if normalize_by == 'Rt':
                for ax in [axMdot, axV, axd, axT, axL]:
                    ax.axvline(r_tr_median[i]/r_normalizer, c = colors_obs[i], ls = ':')
                    # ax.axvline(rph_median[i]/r_normalizer, c = colors_obs[i], ls = '--')
            print(f'{lab}: Rtr/Rph: {r_tr_median[i]/rph_median[i]}')

            r_plot = profiles[lab]['r'] 
            d = profiles[lab]['d_mean']
            v_rad = profiles[lab]['v_rad_median'] 
            t = profiles[lab]['t_median']
            L_adv = profiles[lab]['L_adv_median']
            Mdot = profiles[lab]['Mdot_mean'] 

            axd.plot(r_plot/r_normalizer, d*prel.den_converter,  color = colors_obs[i]) #, ls = lines_obs[i] , label = f'{lab}')# Observer {lab} ({indices_sorted[i]})')
            axV.plot(r_plot/r_normalizer, v_rad * conversion_sol_kms,  color = colors_obs[i], label = f'{lab}')
            axMdot.plot(r_plot/r_normalizer, np.abs(Mdot/mfall_mean[i]),  color = colors_obs[i], label = f'{lab}')# , ls = lines_obs[i])
            axT.plot(r_plot/r_normalizer, t,  color = colors_obs[i])#, label = f'{lab}')
            axL.plot(r_plot/r_normalizer, L_adv/Ledd_sol,  color = colors_obs[i]) #, ls = lines_obs[i]), label = f'{lab}')

            # for ax in [axd, axV, axMdot, axT, axL]:
        # axMdot.plot(x_test, y_testplus1, c = 'gray', ls = 'dotted', label = r'$\dot{M} \propto r$')
        # axd.plot(r/apo, rho_from_dM, c = 'gray', ls = '--', label = r'$\rho \propto R^{-2}$') #'From dM/dt')
        axd.plot(x_test, y_test2, c = 'k', ls = 'dashed', label = r'$\rho \propto r^{-2}$')
        axd.text(32, 1.2e-14, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
        # axd.plot(x_test, y_test3, c = 'gray', ls = 'dotted', label = r'$\rho \propto r^{-3}$') 
        # axd.plot(x_test, y_test08, c = 'gray', ls = 'dashed', label = r'$v_r \propto r^{-0.8}$')
        axV.axhline(v_esc_kms, c = 'k', ls = 'dashed')#
        axV.text(35, 1.1*v_esc_kms, r'v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
        # axV.plot(x_test, y_test02, c = 'gray', ls = 'dashed', label = r'$v_r \propto  r^{-0.2}$')
        # axV.plot(x_test, 2.9*y_testplus1, c = 'gray', ls = '-.', label = r'$v_r \propto r$')
        # axV.plot(x_test, 2.5*y_testplus2, c = 'gray', ls = '--', label = r'$v_r \propto r^2$')
        axT.plot(x_test, y_test23, c = 'k', ls = 'dashed', label = r'$T \propto r^{-2/3}$')
        axT.text(32, 1.2e4, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -33)
        axL.plot(x_test,3e-5*y_test23, c = 'k', ls = 'dashed', label = r'$L \propto r^{-2/3}$')

        for ax in [axd, axV, axMdot, axT, axL]:
            # put the legend if which_obs != 'all_rotate'. Lt it be outside
            # boxleg = ax.get_position()
            # ax.set_position([boxleg.x0, boxleg.y0, boxleg.width * 0.8, boxleg.height])
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
            ax.tick_params(axis='both', which='minor', length=6, width=1)
            ax.tick_params(axis='both', which='major', length=10, width=1.5)
            if normalize_by == 'Rt':
                ax.set_xlim(1, 3*apo/Rp)
            elif normalize_by == '':
                ax.set_xlim(Rt/apo, 2)
            ax.set_xlabel(r'$r [r_{\rm t}]$' if normalize_by == 'Rt' else r'$r [r_{\rm tr}]$', fontsize = 28)
            ax.loglog(), 
            # if normalize_by == 'Rt':
            #     ax.legend(fontsize = 19) 
            # if ax != axMdot:
            #     ax.set_title('Median over NON zero lines of sight', fontsize = 20)

            ax.grid()

        axMdot.set_ylim(1e-4, 1e1)
        axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm fb}]$', fontsize = 28) 
        axd.set_ylim(1e-14, 1e-10)
        axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
        axV.set_ylim(2e3, 3e4)
        axV.set_ylabel(r'v$_r$ [km/s]', fontsize = 28)
        axT.set_ylim(9e3, 5e5)
        axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
        axL.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
        axL.set_ylim(2e-1, 2e1)
        # fig.suptitle(f'{check}, t = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
        # figT.suptitle(f'{check}, t = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/paper/den_prof{snap}{which_part}_{normalize_by}.pdf', bbox_inches = 'tight')
        figM.savefig(f'{abspath}/Figs/paper/Mw{snap}{which_part}_{normalize_by}.pdf', bbox_inches = 'tight')
        figL.savefig(f'{abspath}/Figs/paper/L{snap}{which_part}_{normalize_by}.pdf', bbox_inches = 'tight')
        plt.show()


# %%
