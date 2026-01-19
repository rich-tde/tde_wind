""" Find angle averaged radial velocity profiles for different region of the space."""
# from Mdot_Rfixed import Medd_code
import sys
abspath = '/Users/paolamartire/shocks'
sys.path.append(abspath)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from sklearn.neighbors import KDTree
import healpy as hp
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import to_spherical_components, make_tree, choose_observers

compute = True
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
snap = 151
which_obs = 'dark_bright_z' # 'dark_bright_z' # 'arch', 'quadrants', 'axis'
norm = '' # '' or '_norm'
n_obs = '' # '' or '_npix8
if n_obs == '_npix8':
    NSIDE = 8  
else:
    NSIDE = prel.NSIDE
NPIX = hp.nside2npix(NSIDE)
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
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
conversion_sol_kms = prel.Rsol_cgs*1e-5/prel.tsol_cgs
v_esc_kms = v_esc * conversion_sol_kms
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs
# To plot lines
x_test = np.arange(1., 300)
y_test1 = 9e4*(x_test)**(-1)
y_test02 = 5e3* (x_test)**(-0.2)
y_test23 = 3.5e5*(x_test)**(-2/3)
y_test2 = 5e-11* (x_test)**(-2) 

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

def mean_nonzero(arr, axis=1):
    count = np.count_nonzero(arr, axis=axis)
    return np.divide(
        np.sum(arr, axis=axis),
        count,
        out=np.zeros_like(count, dtype=float),
        where=count != 0
    )

def radial_profiles(loadpath, snap, observers_xyz, indices_sorted, r_tr, ray_params = None):
    # create rays
    if ray_params == None:
        rmin, rmax, Nray = 1e-1, 50, 200
    else:
        rmin, rmax, Nray = ray_params
    r = np.logspace(np.log10(rmin), np.log10(rmax), Nray)
    
    # Load data and pick wind cells
    data = make_tree(loadpath, snap, energy = True)
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
    cut = Den > 1e-19
    R = np.sqrt(X**2 + Y**2 + Z**2)  
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)  
    V_r, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
    bern = orb.bern_coeff(R, vel, Den, Mass, Press, IE_den, Rad_den, params)
    cut = np.logical_and(Den > 1e-19, np.logical_and(bern > 0, V_r > 0)) 
    X, Y, Z, Vol, Den, Mass, VX, VY, VZ, V_r, T, Press, IE_den, Rad_den = \
        make_slices([X, Y, Z, Vol, Den, Mass, VX, VY, VZ, V_r, T, Press, IE_den, Rad_den], cut)       

    xyz = np.array([X, Y, Z]).T
    tree = KDTree(xyz, leaf_size = 50) 

    all_outflows = {}
    for j, idx_list in enumerate(indices_sorted):
        print(f'Region: {label_obs[j]}, considering {len(idx_list)} obs', flush= True)
        v_rad_all = [] # it will have shape (len(idx_list), N_ray)
        d_all = []
        m_all = []
        vol_all = []
        t_all = []
        L_adv_all = []
        Mdot_dimcell_all = []

        if len(idx_list) == 0:
            v_rad_all.append(np.zeros(Nray))
            d_all.append(np.zeros(Nray))
            m_all.append(np.zeros(Nray))
            vol_all.append(np.zeros(Nray))
            t_all.append(np.zeros(Nray))
            L_adv_all.append(np.zeros(Nray))
            Mdot_dimcell_all.append(np.zeros(Nray))

        else:
            for i in idx_list: # i in [0, Nobs]: pick the line of sight that you'll use for the mean of the chosen direction
                # print(f'Obs {i}', flush= True)
                if r_tr[i] == 0:
                    v_rad_all.append(np.zeros(Nray))
                    d_all.append(np.zeros(Nray))
                    m_all.append(np.zeros(Nray))
                    vol_all.append(np.zeros(Nray))
                    t_all.append(np.zeros(Nray))
                    L_adv_all.append(np.zeros(Nray))
                    Mdot_dimcell_all.append(np.zeros(Nray))
                    continue
                mu_x = observers_xyz[i][0]
                mu_y = observers_xyz[i][1]
                mu_z = observers_xyz[i][2]

                if norm == '_norm':
                    x = r*r_tr[i]*mu_x
                    y = r*r_tr[i]*mu_y
                    z = r*r_tr[i]*mu_z
                else:
                    x = r*mu_x
                    y = r*mu_y
                    z = r*mu_z
                xyz2 = np.array([x, y, z]).T
                del x, y, z

                dist, idx = tree.query(xyz2, k=10) 
                # dist = np.concatenate(dist)
                # idx = np.array([ int(idx[i][0]) for i in range(len(idx))])
                
                # Quantity corresponding to the ray
                ray_V_r = V_r[idx] 
                d = Den[idx] 
                ray_m = Mass[idx]
                ray_vol = Vol[idx]
                ray_rad_den = Rad_den[idx]
                r_sim = np.sqrt(X[idx]**2 + Y[idx]**2 + Z[idx]**2)
            
                # pick them just if near enough, otherwise set to 0 (easier for saving, insted of discard them)
                check_dist = dist <= ray_vol**(1/3) # np.logical_and(dist <= ray_vol**(1/3), r_sim >= rmin_array)
                ray_V_r[~check_dist] = 0 # shape is (N_ray, k)
                d[~check_dist] = 0 
                ray_m[~check_dist] = 0
                ray_vol[~check_dist] = 0
                ray_rad_den[~check_dist] = 0
                ray_t = (ray_rad_den*prel.en_den_converter/prel.alpha_cgs)**(1/4) 
                if norm == '_norm': # in that case we'll do the mean of Mdot since we can't fund the normalization... nor very happy of that
                    ray_Mdot = 4 * np.pi * r_sim**2 * ray_V_r * d
                else:
                    ray_Mdot = np.pi * ray_vol**(2/3) * ray_V_r * d 
                L_adv = 4 * np.pi * r_sim**2 * ray_V_r * ray_rad_den

                v_rad_all.append(mean_nonzero(ray_V_r))
                d_all.append(mean_nonzero(d))
                m_all.append(mean_nonzero(ray_m))
                vol_all.append(mean_nonzero(ray_vol))
                t_all.append(mean_nonzero(ray_t))
                L_adv_all.append(mean_nonzero(L_adv))
                Mdot_dimcell_all.append(mean_nonzero(ray_Mdot))
        
        # shape v_rad_all: (len(idx_list), Nray)
        t_mean = []
        v_rad_mean = []
        d_mean = [] 
        L_adv_mean = []
        Mdot_mean = []
        for i in range(Nray):
            # if len(v_rad_all)== 0:
            #     v_rad_mean.append(0)
            #     d_mean.append(0)
            #     t_mean.append(0)
            #     L_adv_mean.append(0)
            #     Mdot_mean.append(0)
            #     continue
            v_rad_col = np.transpose(v_rad_all)[i]
            d_col = np.transpose(d_all)[i]
            vol_col = np.transpose(vol_all)[i]
            t_col = np.transpose(t_all)[i]
            L_adv_col = np.transpose(L_adv_all)[i]
            Mdot_col = np.transpose(Mdot_dimcell_all)[i]
            ray_m_col = np.transpose(m_all)[i] 
            cond = ray_m_col != 0
            nonzero = v_rad_col[cond] 
            v_rad_mean.append(np.sum(nonzero*ray_m_col[cond])/np.sum(ray_m_col[cond]))
            nonzero = d_col[cond]
            d_mean.append(np.sum(nonzero*ray_m_col[cond])/np.sum(ray_m_col[cond]))
            nonzero = t_col[cond]
            t_mean.append(np.sum(nonzero*vol_col[cond])/np.sum(vol_col[cond]))
            nonzero = L_adv_col[cond]
            L_adv_mean.append(np.mean(nonzero))
            if norm == '_norm':
                Mdot_mean.append(np.mean(Mdot_col[cond]))
            else:
                Mdot_mean.append(1/np.sum((vol_col[cond])**(2/3))* np.sum(Mdot_col[cond]))

        v_rad_mean = np.array(v_rad_mean)
        d_mean = np.array(d_mean)
        t_mean = np.array(t_mean)
        L_adv_mean = np.array(L_adv_mean)
        Mdot_mean = np.array(Mdot_mean)
        Mdot_mean = np.array(Mdot_mean if norm == '_norm' else 4 * r**2 * Mdot_mean)

        outflow = {
            'r': r,
            'v_rad_mean': v_rad_mean,
            'd_mean': d_mean,
            't_mean': t_mean,
            'L_adv_mean': L_adv_mean,
            'Mdot_mean': Mdot_mean,
            'indices_obs': idx_list
        }
        key = f"{label_obs[j]}"
        all_outflows[key] = outflow   # dict of dicts

    return all_outflows

#
## MAIN
#

# Load data
ph_data = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}{n_obs}.txt')
x_ph, y_ph, z_ph, Lum_ph = ph_data[0], ph_data[1], ph_data[2], ph_data[-2]
rph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
dataRtr = np.load(f"{abspath}/data/{folder}/trap/{check}_Rtr{snap}{n_obs}.npz")
x_tr, y_tr, z_tr = dataRtr['x_tr'], dataRtr['y_tr'], dataRtr['z_tr']
r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)

# divide Healpix observers
observers_xyz = np.array(hp.pix2vec(NSIDE, range(NPIX))) # shape is 3,N
x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, which_obs)
observers_xyz = np.transpose(observers_xyz) #shape: Nx3

r_tr_means = np.zeros(len(indices_sorted))
rph_means = np.zeros(len(indices_sorted))
fig, (ax1, ax2) = plt.subplots(1, 2)
for i, idx_list in enumerate(indices_sorted):  
    ax1.scatter(x_obs[idx_list], y_obs[idx_list], facecolor = 'none', edgecolors = 'k', linewidths = 1)
    ax2.scatter(x_obs[idx_list], z_obs[idx_list], facecolor = 'none', edgecolors = 'k', linewidths = 1)
    rph_means[i] = np.mean(rph[idx_list])
    
    nonzero = r_tr[idx_list] != 0
    if nonzero.any():
        print(f'{label_obs[i]}: no Rtr in {np.sum(~nonzero)/len(r_tr[idx_list])*100:.2f}%')
        r_tr_means[i] = np.mean(r_tr[idx_list[nonzero]])
    idx_list = idx_list[nonzero]
    indices_sorted[i] = idx_list
    # Plot the observers with trapping radius non zero
    ax1.scatter(x_obs[idx_list], y_obs[idx_list], color = colors_obs[i], linewidths = 1)
    ax2.scatter(x_obs[idx_list], z_obs[idx_list], color = colors_obs[i], linewidths = 1, label = r'r$_{\rm tr}\neq0$' if i == 0 else '')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax1.legend()
plt.tight_layout()

if compute:
    path = f'{abspath}/TDE/{folder}/{snap}'
    if norm == '_norm':
        profiles = radial_profiles(path, snap, observers_xyz, indices_sorted, r_tr)
    else: 
        profiles = radial_profiles(path, snap, observers_xyz, indices_sorted, r_tr, [Rt, 3*np.mean(rph), 200])
    out_path = f"{abspath}/data/{folder}/paper2/den_prof{snap}{which_obs}_sections{n_obs}{norm}.npy"
    np.save(out_path, profiles, allow_pickle = True)
else:
    profiles = np.load(f'{abspath}/data/{folder}/paper2/den_prof{snap}{which_obs}_sections{n_obs}{norm}.npy', allow_pickle=True).item()


fig, (axV, axd, axT) = plt.subplots(1, 3, figsize=(24, 6)) 
figM, (axMdot, axL) = plt.subplots(1, 2, figsize=(18, 6))
for i, lab in enumerate(profiles.keys()):
    if i == 3:
        continue
    # for ax in [axMdot, axV, axd, axT, axL]:
    #     ax.axvline(r_tr_means[i]/Rt, c = colors_obs[i], ls = ':')
    #     ax.axvline(rph_means[i]/Rt, c = colors_obs[i], ls = '--')

    r_plot = profiles[lab]['r']
    idx_ph = np.argmin(np.abs(r_plot - rph_means[i]))
    idx_tr = np.argmin(np.abs(r_plot - r_tr_means[i]))
    v_rad = profiles[lab]['v_rad_mean']
    d = profiles[lab]['d_mean']
    t = profiles[lab]['t_mean']
    L_adv = profiles[lab]['L_adv_mean']
    Mdot= profiles[lab]['Mdot_mean']
    
    if norm == '_norm':
        axV.plot(r_plot, v_rad * conversion_sol_kms,  color = colors_obs[i], label = f'{lab}')
        axd.plot(r_plot, d*prel.den_converter,  color = colors_obs[i]) #, ls = lines_obs[i] , label = f'{lab}')# Observer {lab} ({indices_sorted[i]})')
        axMdot.plot(r_plot, np.abs(Mdot/Medd_sol),  color = colors_obs[i], label = f'{lab}')# , ls = lines_obs[i])
        axT.plot(r_plot, t,  color = colors_obs[i])#, label = f'{lab}')
        axL.plot(r_plot, L_adv/Ledd_sol,  color = colors_obs[i])
    
    else:
        axV.plot(r_plot/Rt, v_rad * conversion_sol_kms,  color = colors_obs[i], label = f'{lab}')
        axd.plot(r_plot/Rt, d*prel.den_converter,  color = colors_obs[i]) #, ls = lines_obs[i] , label = f'{lab}')# Observer {lab} ({indices_sorted[i]})')
        axMdot.plot(r_plot/Rt, np.abs(Mdot/Medd_sol),  color = colors_obs[i], label = f'{lab}')# , ls = lines_obs[i])
        axT.plot(r_plot/Rt, t,  color = colors_obs[i])#, label = f'{lab}')
        axL.plot(r_plot/Rt, L_adv/Ledd_sol,  color = colors_obs[i]) #, ls = lines_obs[i]), label = f'{lab}')]
        # add point where is the photosphere and trapping radius
        axV.scatter(r_plot[idx_ph]/Rt, v_rad[idx_ph] * conversion_sol_kms, color = colors_obs[i], s = 120)
        axd.scatter(r_plot[idx_ph]/Rt, d[idx_ph]*prel.den_converter, color = colors_obs[i], s = 120)
        axT.scatter(r_plot[idx_ph]/Rt, t[idx_ph], color = colors_obs[i], s = 120)
        axL.scatter(r_plot[idx_ph]/Rt, L_adv[idx_ph]/Ledd_sol, color = colors_obs[i], s = 120)
        axMdot.scatter(r_plot[idx_ph]/Rt, np.abs(Mdot[idx_ph]/Medd_sol), color = colors_obs[i], s = 120)
        axV.scatter(r_plot[idx_tr]/Rt, v_rad[idx_tr] * conversion_sol_kms, color = colors_obs[i], s = 120, marker = 's')
        axd.scatter(r_plot[idx_tr]/Rt, d[idx_tr]*prel.den_converter, color = colors_obs[i], s = 120, marker = 's')
        axT.scatter(r_plot[idx_tr]/Rt, t[idx_tr], color = colors_obs[i], s = 120, marker = 's')
        axL.scatter(r_plot[idx_tr]/Rt, L_adv[idx_tr]/Ledd_sol, color = colors_obs[i], s = 120, marker = 's')
        axMdot.scatter(r_plot[idx_tr]/Rt, np.abs(Mdot[idx_tr]/Medd_sol), color = colors_obs[i], s = 120, marker = 's')

if norm != '_norm':
    axd.plot(x_test, y_test2, c = 'k', ls = 'dashed', label = r'$\rho \propto r^{-2}$')
    # axd.text(32, 1.2e-14, r'$\rho \propto r^{-2}$', fontsize = 20, color = 'k', rotation = -42)
    axV.axhline(v_esc_kms, c = 'k', ls = 'dashed')#
    axV.text(35, 1.1*v_esc_kms, r'v$_{\rm esc} (r_{\rm p})$', fontsize = 20, color = 'k')
    axT.plot(x_test, y_test23, c = 'k', ls = 'dashed', label = r'$T \propto r^{-2/3}$')
    # axT.text(32, 1.2e4, r'$T_{\rm rad} \propto r^{-2/3}$', fontsize = 20, color = 'k', rotation = -33)
    axL.plot(x_test,1.5e-5*y_test23, c = 'k', ls = 'dashed', label = r'$L \propto r^{-2/3}$')

for ax in [axd, axV, axMdot, axT, axL]:
    ax.tick_params(axis='both', which='minor', length = 6, width=1)
    ax.tick_params(axis='both', which='major', length = 10, width=1.5)
    if norm == '_norm':
        ax.set_xlabel(r'$r [r_{\rm tr}]$', fontsize = 28)
        ax.set_xlim(1e-1, 5)
    else:
        ax.set_xlabel(r'$r [r_{\rm t}]$', fontsize = 28)
        ax.set_xlim(1, 1e2)
    ax.loglog()
    ax.grid()

axMdot.legend(fontsize = 19) 
axMdot.set_ylim(7e2, 5e5)
if norm == '_norm':
    axMdot.text(1.5, 1.1e4, r'$\dot{M}_{\rm w} = 4\pi \langle r^2 v_{\rm} \rho \rangle$', fontsize = 20, color = 'k')
axMdot.set_ylabel(r'$\dot{M}_{\rm w} [\dot{M}_{\rm Edd}]$', fontsize = 28) 
axd.set_ylim(1e-13, 1e-8)
axd.set_ylabel(r'$\rho$ [g/cm$^3]$', fontsize = 28)
axV.set_ylim(2e3, 3e4)
axV.set_ylabel(r'v$_r$ [km/s]', fontsize = 28)
axT.set_ylim(2e4, 5e5)
axT.set_ylabel(r'$T_{\rm rad}$ [K]', fontsize = 28)
axL.set_ylabel(r'$L [L_{\rm Edd}]$', fontsize = 28)
axL.set_ylim(1e-1, 50)
# fig.suptitle(f'{check}, t = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
# figT.suptitle(f'{check}, t = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20)
fig.tight_layout()
# fig.savefig(f'{abspath}/Figs/paper/den_prof{snap}{which_part}_{normalize_by}.pdf', bbox_inches = 'tight')
# figM.savefig(f'{abspath}/Figs/paper/Mw{snap}{which_part}_{normalize_by}.pdf', bbox_inches = 'tight')
# figL.savefig(f'{abspath}/Figs/paper/L{snap}{which_part}_{normalize_by}.pdf', bbox_inches = 'tight')
plt.show()


# %%
 
#     if plot:
#         x_as_thetas = True
#         thetas_tick = [-np.pi, 0, np.pi]
#         thetas_tick_labels = [r'$-\pi$', r'$0$', r'$\pi$']
#         profiles = np.load(f'{abspath}/data/{folder}/paper2/den_prof{snap}{which_obs}{which_part}_obsHealp.npy', allow_pickle=True).item()
#         # find eta = mfall(t_fb-t_dyn)/Mwind(tfb)
#         figeta, axeta = plt.subplots(1, 1, figsize=(9, 6))
#         fig, (axEdd, axL) = plt.subplots(2, 1, figsize=(9, 12))
#         mfall_mean = np.zeros(len(profiles))
#         for i, lab in enumerate(profiles.keys()):
#             if i == 3:
#                 continue
#             # r_tr = r_tr_means[i]
#             rph_single = rph_means[i]
#             Lum_ph_single = Lum_ph_means[i]
#             r_plot = profiles[lab]['r']
#             d = profiles[lab]['d_mean']
#             # v_rad_tr = profiles[lab]['v_rad_median'][np.argmin(np.abs(r_plot-r_tr))]
#             # Mdot = profiles[lab]['Mdot_mean'][np.argmin(np.abs(r_plot-r_tr))]
#             # Mdot = profiles[lab]['Mdot_mean'][np.argmin(np.abs(r_plot-r_tr))]
            
#             # t_dyn = (r_tr/v_rad_tr)*prel.tsol_cgs/t_fb_days_cgs # you want it in t_fb
#             # tfb_adjusted = tfb - t_dyn
#             # find_time = np.argmin(np.abs(tfb_fall-tfb_adjusted))
#             # mfall_mean[i] = mfall[find_time]
#             # eta = np.abs(Mdot/mfall_mean[i])

#             if x_as_thetas:
#                 # axeta.scatter(thetas_tic
#                 # k[i], eta, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
#                 axEdd.scatter(thetas_tick[i], Mdot/Medd_sol, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
#                 axL.scatter(thetas_tick[i], Lum_ph_single/Ledd_cgs, c = colors_obs[i], label = f'{label_obs[i]}', ls = lines_obs[i], s = 100)
#             else:
#             #     axeta.scatter(r_tr/Rt, eta, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
#             #     axEdd.scatter(r_tr/Rt, Mdot/Medd_sol, color = colors_obs[i], s = 80, label = f'{label_obs[i]}')
#                 axL.scatter(rph_single/Rt, Lum_ph_single/Ledd_cgs, c = colors_obs[i], label = f'{label_obs[i]}', ls = lines_obs[i], s = 100)

#         for axis in [axeta, axEdd, axL]: 
#             if x_as_thetas:
#                 axis.set_xlabel(r'$\theta$')
#                 axis.set_xticks(thetas_tick) 
#                 axis.set_xticklabels(thetas_tick_labels)
#                 axis.set_yscale('log')  
#                 axEdd.legend(fontsize = 14)
#             else: 
#                 if axis != axL:
#                     axis.set_xlabel(r'$r_{\rm tr} [r_{\rm t}]$')
#                     axis.legend(fontsize = 14)
#                 else:
#                     axis.set_xlabel(r'$r_{\rm ph} [r_{\rm t}]$')
#                 axis.set_xlim(1, 150)
#                 axis.set_ylim(1e-5, 1e-1)
#                 axis.loglog()
#             axis.tick_params(axis='both', which='minor', length=5, width=1)
#             axis.tick_params(axis='both', which='major', length=8, width=1.2)
#             axis.grid()
#         axeta.set_ylabel(r'$\eta = |\dot{M}_{\rm w}/\dot{M}_{\rm fb}|$')
#         axeta.set_ylim(1e-5, 1e-1)
#         axL.set_ylabel(r'$L_{\rm ph} [L_{\rm Edd}]$')
#         axL.set_ylim(2e-1, 10)
#         axEdd.set_ylabel(r'$|\dot{M}_{\rm w}| [\dot{M}_{\rm Edd}]$')
#         axEdd.set_ylim(3, 6e2)
#         fig.suptitle(f't = {np.round(tfb, 2)}' + r' t$_{\rm fb}$', fontsize = 20)
#         fig.tight_layout()
#         # fig.savefig(f'{abspath}/Figs/next_meeting/MdotL{snap}{which_part}.png', bbox_inches = 'tight')
