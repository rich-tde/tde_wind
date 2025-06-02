""" Compare different resolutions"""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'

import numpy as np
from Utilities.operators import make_tree
from Utilities.time_extractor import days_since_distruption
import Utilities.sections as sec
import src.orbits as orb
import Utilities.prelude as prel
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.Opacity.linextrapolator import first_rich_extrap, linear_rich
import matlab.engine
#%%
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
Rt = orb.tidal_radius(Rstar, mstar, Mbh) 
Rp = orb.pericentre(Rstar, mstar, Mbh, beta)
R0 = 0.6*Rp
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
compton = 'Compton'
change = 'AMR'
if change == 'Extr':
    snap = 267 # 164 or 267 
    eng = matlab.engine.start_matlab()
    checks = ['', 'QuadraticOpacity']
    check_name = ['Old','Old-']
    opac_path = f'{abspath}/src/Opacity'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    minT_tab, maxT_tab = np.min(np.exp(T_cool)), np.max(np.exp(T_cool))
    minRho_tab, maxRho_tab = np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool))
    print(f'minT_tab = {minT_tab}, maxT_tab = {maxT_tab}')
    print(f'minRho_tab = {minRho_tab}, maxRho_tab = {maxRho_tab}')
else:
    snap = 164 # 115, 164, 240 
    checks = ['QuadraticOpacity', 'QuadraticOpacityNewAMR']
    check_name = ['Old-','New']

Ncell = np.zeros(len(checks))
Ncell_mid = np.zeros(len(checks))
fig, ax = plt.subplots(2, 2, figsize = (18, 10))
if change == 'Extr':
    figK, axK = plt.subplots(1, 2, figsize = (10, 5))
for i,check in enumerate(checks):
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}' 
    path = f'{abspath}/TDE/{folder}/{snap}'
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    data = make_tree(path, snap, energy = True)
    cut = data.Den > 1e-19
    X, Y, Z, den, Vol, Diss_den, Temp = \
       sec. make_slices([data.X, data.Y, data.Z,data.Den, data.Vol, data.Diss, data.Temp], cut)
    Ncell[i] = len(data.X)
    print(f'check = {check}, Ncell = {Ncell[i]}')
    dim_cell = Vol**(1/3) 

    midplane = np.abs(Z) < dim_cell
    X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Diss_den_midplane, Temp_midplane = \
        sec.make_slices([X, Y, Z, dim_cell, den, Diss_den, Temp], midplane)
    Ncell_mid[i] = len(X_midplane)
    Den_midplane_cgs = Den_midplane * prel.Msol_cgs / prel.Rsol_cgs**3 # [g/cm^3]

    if change == 'Extr':
        extrapolated_den = np.logical_or(Den_midplane_cgs < minRho_tab, Den_midplane_cgs > maxRho_tab)
        extrapolated_temp = np.logical_or(Temp_midplane < minT_tab, Temp_midplane > maxT_tab)
        extrapoalted_all = np.logical_or(extrapolated_den, extrapolated_temp)
        if check in ['LowRes', '', 'HiRes']:
            T_cool2, Rho_cool2, rossland2 = first_rich_extrap(T_cool, Rho_cool, rossland, what = 'scattering_limit', slope_length = 5, highT_slope=-3.5)
        if check in ['QuadraticOpacity', 'QuadraticOpacityNewAMR']:
            T_cool2, Rho_cool2, rossland2 = linear_rich(T_cool, Rho_cool, rossland, what = 'scattering_limit', highT_slope = 0)
        sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(Temp_midplane), np.log(Den_midplane_cgs), 'linear', 0)
        sigma_rossland = np.array(sigma_rossland)[0]
        sigma_rossland_eval = np.exp(sigma_rossland) # [1/cm]
        sigma_rossland_eval[sigma_rossland == 0.0] = 1e-20
        kappa_mid = sigma_rossland_eval/Den_midplane_cgs # [cm^2/g]
        tau_mid = sigma_rossland_eval * dim_midplane * prel.Rsol_cgs 

        img = ax[0][i].scatter(X_midplane/apo, Y_midplane/apo, c = tau_mid, s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1, vmax = 1e5))
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\tau$', fontsize = 20)

        img = ax[1][i].scatter(X_midplane[extrapoalted_all]/apo, Y_midplane[extrapoalted_all]/apo, c = kappa_mid[extrapoalted_all], s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e2))
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\kappa$ [cm$^2$/g] extrapolated cells', fontsize = 20)
        ax[1][i].text(-1.45, 0.4, r'median $\rho$ = ' + f'{np.median(Den_midplane_cgs[extrapoalted_all]):.2e} g/cm$^3$, median T = {np.median(Temp_midplane[extrapoalted_all]):.2e} ', fontsize = 18)

        img = axK[i].scatter(Den_midplane_cgs[extrapoalted_all], Temp_midplane[extrapoalted_all], s = 1, c = kappa_mid[extrapoalted_all], cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-1, vmax = 1e2))
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\kappa$ [cm$^2$/g]', fontsize = 20)
        axK[i].axvline(minRho_tab, color = 'k', linestyle = '--')
        axK[i].axvline(maxRho_tab, color = 'k', linestyle = '--')
        axK[i].axhline(minT_tab, color = 'k', linestyle = '--')
        axK[i].axhline(maxT_tab, color = 'k', linestyle = '--')
        axK[i].set_xscale('log')
        axK[i].set_yscale('log')
        axK[i].set_xlabel(r'$\rho$ [g/cm$^3$]', fontsize = 18)
        axK[0].set_ylabel(r'T [K]', fontsize = 18)

    if change == 'AMR':
        img = ax[0][i].scatter(X_midplane/apo, Y_midplane/apo, c = Den_midplane_cgs, s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-11, vmax = 1e-5))
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\rho$ [g/cm$^3]$', fontsize = 20)

        img = ax[1][i].scatter(X_midplane/apo, Y_midplane/apo, c = np.abs(Diss_den_midplane)*prel.en_den_converter/prel.tsol_cgs, s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-5, vmax = 1e7))
        cbar = plt.colorbar(img)
        cbar.set_label(r'$|\dot{u_{\rm diss}}|$ [g/cm$^3s]$', fontsize = 20)
    
    ax[0][i].set_title(f'{check_name[i]} run', fontsize = 20)
    
for i in range(2):
    for j in range(2):  
        ax[i][j].set_xlim(-1.5,.1)
        ax[i][j].set_ylim(-.5, .5)
    ax[1][i].set_xlabel(r'X [$R_{\rm a}$]', fontsize = 18)
    ax[i][0].set_ylabel(r'Y [$R_{\rm a}$]', fontsize = 18)

fig.suptitle(f't = {np.round(tfb,2)}' + r't$_{\rm fb}, N_{\rm cell, left}/N_{\rm cell, right}=$' + f'{np.round(Ncell[0]/Ncell[1], 2)}, on midplane: {np.round(Ncell_mid[0]/Ncell_mid[1], 2)}', fontsize = 20)
fig.tight_layout()
# fig.savefig(f'{abspath}/Figs/Test/Test{change}{snap}.png', dpi = 100, bbox_inches = 'tight')
if change == 'Extr':
    figK.suptitle('Extrapolated cells', fontsize = 20)
    figK.tight_layout()
    # figK.savefig(f'{abspath}/Figs/Test/TestKextrap{snap}.png', dpi = 100, bbox_inches = 'tight')

# %%
