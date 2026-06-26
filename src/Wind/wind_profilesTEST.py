""" Find/plot radial profiles as weighted average on spherical sections. 
Find/plot polar profiles for fixed r and phi_array. 
Written to be run locally."""

import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import healpy as hp
from sklearn.neighbors import KDTree
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
import Utilities.operators as op

#
# PARAMS
#%%
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
which_obs = 'left_right_z' # 'left_right_z', 'all' or 'in_out_z'
snap = 151
rmin, rmax, Nray = Rt, 50*Rt, 50
origin = '0'
more_cuts = 'cutVphi'

path = f'{pre}/{snap}'
ray_array = np.logspace(np.log10(rmin), np.log10(rmax), Nray)

#%%
data = op.make_tree(path, snap)
X, Y, Z, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
    data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.VX, data.VY, data.VZ, data.Temp, data.Press, data.IE, data.Rad
Rsph = np.sqrt(X**2 + Y**2 + Z**2)  
cut = np.logical_and(Den > 1e-19, Rsph > rmin)
X, Y, Z, Rsph, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den = \
    make_slices([X, Y, Z, Rsph, Vol, Den, Mass, VX, VY, VZ, T, Press, IE_den, Rad_den], cut)
dim_cell = Vol**(1/3) 
indices_all = np.arange(len(X))
cut, _, V_r = orb.pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params, cond = 'bern')
_, V_theta, V_phi = op.to_spherical_components(VX, VY, VZ, X, Y, Z)
if more_cuts == 'cutVphi':
    cut = np.logical_and(cut, np.abs(V_phi)<V_r)
if origin == 'Rp':
    X  = X-Rp
elif origin == '0':
    X  = X
else:
    raise ValueError(f'Origin {origin} not recognized. Use "Rp" or "0".')
# split in sections the wind cells
Rsph_all = Rsph.copy()
Mass_all = Mass.copy()
dim_cell_all = dim_cell.copy()
sections = op.choose_sections(X, Y, Z, which_obs)
cond_sec_all = []
for key in sections.keys():
    cond_sec_all.append(sections[key]['cond'])

X, Y, Z, Rsph, Vol, Den, Mass, V_r, V_phi, T, Press, IE_den, Rad_den, dim_cell = \
    make_slices([X, Y, Z, Rsph, Vol, Den, Mass, V_r, V_phi, T, Press, IE_den, Rad_den, dim_cell], cut)       
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
for i, r in enumerate(ray_array): 
    # find cells at r
    ind_r = np.abs(Rsph-r) < dim_cell
    ind_r_all = np.abs(Rsph_all-r) < dim_cell_all 
    shell_indices.append(indices[ind_r])
    shell_all_indices.append(indices_all[ind_r_all])

# Convert to arrays for faster later indexing
shell_indices = [np.asarray(s, dtype=int) for s in shell_indices]
shell_all_indices = [np.asarray(s, dtype=int) for s in shell_all_indices]

all_outflows = {}
#%%
figtest, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
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
    lens_tot = np.zeros(Nray)
    ratio_un = np.zeros(Nray)
    Mass_tot = np.zeros(Nray)
    Mass_wind = np.zeros(Nray)
    d_profmean = np.zeros(Nray)
    
    for i, r in enumerate(ray_array): 
        const_C = 4*r**2/len(cond_sec)
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
        
        X_toplot = X[idx]
        Y_toplot = Y[idx]
        Z_toplot = Z[idx]
        dim_toplot = dim_cell[idx]
        Den_toplot = Den[idx]
        Vphi_toplot = V_phi[idx]
        Vr_toplot = V_r[idx]
        Vtheta_toplot = V_theta[idx]
        # make an istogram of the density where bins are X
        # binsX, edgesX = np.histogram(X_toplot, bins = 50, weights = Den_toplot)
        # binsY, edgesY = np.histogram(Y_toplot, bins = 50, weights = Den_toplot)
        # binsZ, edgesZ = np.histogram(Z_toplot, bins = 50, weights = Den_toplot)
        # ax1.plot(edgesX[:-1]/Rt, binsX * prel.den_converter, color = 'k')
        # ax2.plot(edgesY[:-1]/Rt, binsY * prel.den_converter,  color = 'r')
        # ax3.plot(edgesZ[:-1]/Rt, binsZ * prel.den_converter, color = 'b')
        # find the bin with the maximum 
        
        img1 = ax1.scatter(X_toplot[np.abs(Z_toplot)<Rt]/Rt, Y_toplot[np.abs(Z_toplot)<Rt]/Rt, s = 2, c = Den_toplot[np.abs(Z_toplot)<Rt]*prel.den_converter, norm = colors.LogNorm(vmin = 1e-16, vmax = 1e-10), cmap = 'rainbow')
        img2 = ax2.scatter(X_toplot[np.abs(Y_toplot)<Rt]/Rt, Z_toplot[np.abs(Y_toplot)<Rt]/Rt, s = 2, c = Den_toplot[np.abs(Y_toplot)<Rt]*prel.den_converter, norm = colors.LogNorm(vmin = 1e-16, vmax = 1e-10), cmap = 'rainbow')
        img3 = ax3.scatter(X_toplot[np.abs(Z_toplot)<Rt]/Rt, Y_toplot[np.abs(Z_toplot)<Rt]/Rt, s = 2, c = np.abs(Vphi_toplot[np.abs(Z_toplot)<Rt])/Vr_toplot[np.abs(Z_toplot)<Rt], norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1), cmap = 'coolwarm')
        img4 = ax4.scatter(X_toplot[np.abs(Y_toplot)<Rt]/Rt, Z_toplot[np.abs(Y_toplot)<Rt]/Rt, s = 2, c = np.abs(Vtheta_toplot[np.abs(Y_toplot)<Rt])/Vr_toplot[np.abs(Y_toplot)<Rt], norm = colors.LogNorm(vmin = 1e-1, vmax = 1e1), cmap = 'coolwarm')

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
        d_profmean[i] = np.mean(ray_d)
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
        'd_profmean': d_profmean,
        'Mdot_prof': Mdot_prof,
        'L_advmean_prof': L_advmean_prof,
        'Mdotmean_prof': Mdotmean_prof,
        'L_adv_prof': L_adv_prof,
        'L_kin_prof': L_kin_prof,
        'L_kinmean_prof': L_kinmean_prof,
        'Ntot_cells': lens_tot,
        'ratio_un': ratio_un,
        'Mass_tot': Mass_tot,
        'Mass_wind': Mass_wind,
        'colors_obs': colors_obs[j],
        'lines_obs': lines_obs[j]
    }

    key = f"{label_obs[j]}"
    all_outflows[key] = outflow

ax1.set_ylabel(r'Y/$r_{\rm t}$')
cbar = plt.colorbar(img1)
cbar.set_label(r'$\rho$ (g/cm$^3$)', fontsize = 20)

ax2.set_ylabel(r'Z/$r_{\rm t}$')
cbar = plt.colorbar(img2)
cbar.set_label(r'$\rho$ (g/cm$^3$)', fontsize = 20)

ax3.set_ylabel(r'Y/$r_{\rm t}$')
cbar = plt.colorbar(img3)
cbar.set_label(r'$|v_{\phi}|$/v$_r$')

ax4.set_ylabel(r'Z/$r_{\rm t}$')
cbar = plt.colorbar(img4)
cbar.set_label(r'$|v_{\theta}|$/v$_r$')
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r'X/$r_{\rm t}$')
plt.suptitle(f'All wind points in snap {snap} with origin = {origin}', fontsize = 22)
plt.tight_layout()
plt.savefig(f"{abspath}/Figs/Test/wind/rprof_scatt{snap}_origin{origin}{more_cuts}.png", dpi = 300)
out_path = f"{abspath}/data/{folder}/wind/r_profile/rTEST_profSec{snap}_{which_obs}_wind_origin{origin}{more_cuts}.npy"
np.save(out_path, all_outflows, allow_pickle=True)

#%%
plot = True
if plot:
    # arrange for plotting
    observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
    x_obs, y_obs, z_obs = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    indices_obs, label_obs, colors_obs, _ = op.choose_observers(observers_xyz, which_obs)
    fig, (axd, axV, axM, axN) = plt.subplots(4, 1, figsize=(8, 30)) 

    all_axes = [axd, axV, axM, axN]
    
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
    axM.set_ylim(1e1, 1e7)

    # Load data Rph and Rtr
    path = f'{pre}/{snap}'
    tfb = np.loadtxt(f'{path}/tfb_{snap}.txt') 
    
    profiles = np.load(f'{abspath}/data/{folder}/wind/r_profile/rTEST_profSec{snap}_{which_obs}_wind_origin{origin}{more_cuts}.npy', allow_pickle=True).item()
    for i, lab in enumerate(profiles.keys()):
        if i > 1: #label_obs[i] == 'South pole':
            continue 
        lab_plot = lab
        r_arr = profiles[lab]['r'] 
        d = profiles[lab]['d_prof']
        # d_mean = profiles[lab]['d_profmean']
        v_rad = profiles[lab]['v_rad_prof'] 
        t = profiles[lab]['t_prof']
        Mdot = profiles[lab]['Mdot_prof'] #Mdotmean_prof
        L_adv = profiles[lab]['L_adv_prof'] #L_advmean_prof
        L_kin = profiles[lab]['L_kin_prof'] #L_kinmean_prof
        Mdotmean = profiles[lab]['Mdotmean_prof'] 
        L_advmean = profiles[lab]['L_advmean_prof'] 
        L_kinmean = profiles[lab]['L_kinmean_prof'] 
        ratio_un = profiles[lab]['ratio_un']
        Ntot_cells = profiles[lab]['Ntot_cells']
        Nwind_cells = ratio_un * Ntot_cells
        Mass_wind = profiles[lab]['Mass_wind']
        Mass_tot = profiles[lab]['Mass_tot']
        ratio_Mass = Mass_wind/Mass_tot
        colors_sec = profiles[lab]['colors_obs']
        # Mdot = d * r_plot**2 * v_rad
        not_zero = np.where(np.logical_and(d != 0, r_arr > 0))
        r_plot, d, d_mean, v_rad, t, Mdot, L_adv, L_kin, ratio_un, ratio_Mass, Mdotmean = \
            make_slices([r_arr, d, d_mean, v_rad, t, Mdot, L_adv, L_kin, ratio_un, ratio_Mass, Mdotmean], not_zero)
        Nwind_cells, Ntot_cells, Mass_wind, Mass_tot = \
            make_slices([Nwind_cells, Ntot_cells, Mass_wind, Mass_tot], not_zero)

        axd.plot(r_plot*norm, d * prel.den_converter, label = f'{lab_plot}', color = colors_sec, linewidth = 2)
        # axd.plot(r_plot*norm, d_mean * prel.den_converter, label = f'mean' if i == 0 else None, color = colors_sec, linewidth = 2, ls = ':')
        axV.plot(r_plot*norm, v_rad * conversion_sol_kms, color = colors_sec, linewidth = 2)
        axM.plot(r_plot*norm, Mdot/Medd_sol, color = colors_sec, linewidth = 2)
        axN.plot(r_plot*norm, Nwind_cells, color = colors_sec, linewidth = 2)


    axV.axhline(v_esc_kms, c = 'k', ls = 'dotted')
    axd.set_ylabel(r'$\rho$ (g/cm$^3$)', fontsize = 28)
    axV.set_ylabel(r'v$_{\rm r}$ (km/s)', fontsize = 28)
    axM.set_ylabel(r'$\dot{M} (\dot{M}_{\rm Edd})$', fontsize = 28)
    axN.set_ylabel(r'N$_{\rm wind}$', fontsize = 28)
    for ax in all_axes: 
        ax.tick_params(axis='both', which='minor', length = 8, width = 1)
        ax.tick_params(axis='both', which='major', length = 15, width = 1.5)
        ax.loglog()
        ax.axvline(apo*norm, color = 'k', ls = 'dotted')
        axd.text(0.8*apo*norm, 0.2*axd.get_ylim()[1], r'$r_{\rm a}$', fontsize = 20, color = 'k', rotation = 90)
        ax.set_xlim(1.5, 1.4e2)
        ax.grid()
    axM.set_xlabel(r'$r /r_{\rm t}$', fontsize = 28)     
    fig.tight_layout()
    axd.set_title(f'Origin = {origin}, t = {np.round(tfb,2)} ' + r'$t_{\rm fb}$', fontsize = 22)
    plt.savefig(f"{abspath}/Figs/Test/wind/rprof_{snap}_origin{origin}{more_cuts}.png", dpi = 300)

    



# %%
