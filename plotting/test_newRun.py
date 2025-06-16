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
from src.Opacity.linextrapolator import opacity_extrap, opacity_linear
import matlab.engine

#
##
#
def L_Edd_k(k): # cgs
    L_e = 1.38*1e42 * (0.34/k)
    return L_e
#
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
choose = 'section' # section, distribution
change = 'AMR'
if change == 'Extr':
    snap = 248 # Fid: 164 or 267 // Low: 126, 248
    eng = matlab.engine.start_matlab()
    checks = ['LowRes', 'LowResOpacityNew'] #['', 'OpacityNew']
    check_name = ['Old','NewExtr+OldAMR']
    colorshist = ['C1', 'goldenrod'] #['yellowgreen', 'forestgreen']
    styles = ['solid', 'dashed']
    markers = ['o', 'x']
elif change == 'AMR':
    snap = 126 # 115, 164, 240 # 126, 301 for sink and no sink
    checks = ['LowResNewAMR', 'LowResNewAMRRemoveCenter'] #['OpacityNew', 'OpacityNewNewAMR']
    check_name = ['no sink', 'sink'] #['NewExtr+OldAMR','NewExtr+NewAMR']
    colorshist = ['plum', 'maroon'] ##['forestgreen', 'k']

opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rosseland = np.loadtxt(f'{opac_path}/ross.txt')
minT_tab, maxT_tab = np.min(np.exp(T_cool)), np.max(np.exp(T_cool))
minRho_tab, maxRho_tab = np.min(np.exp(Rho_cool)), np.max(np.exp(Rho_cool))
# print(f'minT_tab = {minT_tab}, maxT_tab = {maxT_tab}')
# print(f'minRho_tab = {minRho_tab}, maxRho_tab = {maxRho_tab}')

if choose == 'distribution':
    alphahist = [.8, 0.5]
    which_distrib = 'Cum' # 'Cum' or 'Histo
    where = 'Ph' # 'Ph' or 'Mid'
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (15, 15))
    figkappa, ((axkappa1, axkappa4) ,(axkappa2, axkappa3)) = plt.subplots(2,2, figsize = (15, 15))
    Lfld = np.zeros(len(checks))
    median_Rph = np.zeros(len(checks))
    Ncell_ph = np.zeros(len(checks))
    mediankappa = np.zeros(len(checks))
    kappaFlux = np.zeros(len(checks))
    L_kappaFlux = np.zeros(len(checks))
    meanLLe = np.zeros(len(checks))
    for i, check in enumerate(checks):
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}' 
        path = f'{abspath}/TDE/{folder}/{snap}'

        # Load data either for midplane or photosphere and convert density in CGS
        if where == 'Mid':
            tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
            data = make_tree(path, snap, energy = True)
            cut = data.Den > 1e-19
            X, Y, Z, den, Vol, Diss_den, Temp = \
                sec.make_slices([data.X, data.Y, data.Z,data.Den, data.Vol, data.Diss, data.Temp], cut)
            Ncell_ph[i] = len(X)
            dim_cell = Vol**(1/3) 
            midplane = np.abs(Z) < dim_cell
            X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Diss_den_midplane, Temp_midplane = \
                sec.make_slices([X, Y, Z, dim_cell, den, Diss_den, Temp], midplane)
            den_cgs = Den_midplane * prel.den_converter # [g/cm^3]
            Temp_cgs = Temp_midplane # [K]


        if where == 'Ph':
            # download fluxes and photosphere
            data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
            snap_Lum = data[:, 0]   
            Lum = data[:, 2] 
            Lfld[i] = Lum[np.where(snap_Lum == snap)[0][0]]
            idx_fluxes = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}_phidx_fluxes.txt')
            all_indices, all_fluxes = idx_fluxes[0::2], idx_fluxes[1::2]
            snap_flux, time_flux, fluxes = all_fluxes[:,0], all_fluxes[:,1], all_fluxes[:,2:]
            find_snap = np.where(snap_flux == snap)[0]
            find_snap = find_snap[-1]
            tfb, fluxes_ph = time_flux[find_snap], fluxes[find_snap]
            _, _, _, _, denph, Tempph, _, _, _, _, alpha, rph, Lph = \
                np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
            # denph is already CGS!!!!
            median_Rph[i] = np.median(rph)
            den_cgs = denph # [g/cm^3]
            Temp_cgs = Tempph # [K]
            print(f'Lfld in {check_name[i]} at time {np.round(tfb,2)} = {Lfld[i]} erg/s')

        scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
        _, _, scatter_ext = opacity_linear(T_cool, Rho_cool, scattering, slope_length = 7, highT_slope = 0)    
        if check in ['LowRes', '', 'HiRes']:
            T_cool2, Rho_cool2, rosseland2 = opacity_extrap(T_cool, Rho_cool, rosseland, scatter = scatter_ext, slope_length = 5, highT_slope = -3.5)
        if check in ['LowResNewAMR', 'LowResNewAMRRemoveCenter', 'NewAMRRemoveCenter']:
            T_cool2, Rho_cool2, rosseland2 = opacity_extrap(T_cool, Rho_cool, rosseland, scatter = scatter_ext, slope_length = 7, highT_slope = 0)
        if check in ['LowResOpacityNew', 'OpacityNew', 'OpacityNewNewAMR']:
            T_cool2, Rho_cool2, rosseland2 = opacity_linear(T_cool, Rho_cool, rosseland, scatter = scatter_ext, slope_length = 7, highT_slope = 0)
        
        # Compute Rosseland opacity
        ln_alpha = eng.interp2(T_cool2, Rho_cool2, rosseland2.T, np.log(Temp_cgs), np.log(den_cgs), 'linear', 0)
        ln_alpha = np.array(ln_alpha)[0]
        alpha = np.exp(ln_alpha) # [1/cm]
        alpha[ln_alpha == 0.0] = 1e-20
        kappa = alpha/den_cgs # [cm^2/g]

        if where == 'Ph':
            mediankappa[i] = np.median(kappa)
            fluxOverkappa = fluxes_ph/kappa
            kappaFlux[i] = np.sum(fluxes_ph)/np.sum(fluxOverkappa)
            LLe = Lph/L_Edd_k(kappa)
            L_kappaFlux[i] = Lfld[i] / L_Edd_k(kappaFlux[i])
            print(f'ratio L/Ledd in {check_name[i]}:\nwith mean (L_obs/Led(K_obs)) = {np.mean(LLe)} \nwith median (L_obs/Led(K_obs)) = {np.median(LLe)} \nwith flux-weighted inverse k = {L_kappaFlux[i]}\n-------')
        
        log_den_cgs = np.log10(den_cgs) # [g/cm^3]
        log_T = np.log10(Temp_cgs) # [K]
        log_kappa = np.log10(kappa) # [cm^2/g]
        log_alpha = np.log10(alpha)
        log_inverse_kappa = np.log10(1/kappa) 
        log_lum = np.log10(Lph) # [erg/s]
        kappa_flux = fluxes_ph/(np.mean(fluxes_ph)*kappa) # [erg/s/cm^2/g]
        log_inverse_kappaflux = np.log10(kappa_flux)

        if which_distrib == 'Histo':
            ax1.hist(den_cgs/np.mean(den_cgs), bins = 60, color = colorshist[i], alpha = alphahist[i], label = f'{check_name[i]}')
            ax2.hist(Temp_cgs/np.mean(Temp_cgs), bins = 60, color = colorshist[i], alpha = alphahist[i])
            ax3.hist(kappa/np.mean(kappa), bins = 60, color = colorshist[i], alpha = alphahist[i])
            ax4.hist(alpha/np.mean(alpha), bins = 60, color = colorshist[i], alpha = alphahist[i])
            # axkappa1.hist(kappa/np.mean(kappa), bins = 60, color = colorshist[i], alpha = alphahist[i], label = f'{check_name[i]}')
            # axkappa4.hist(Lph/np.mean(Lph), bins = 60, color = colorshist[i], alpha = alphahist[i])
            # axkappa2.hist((1/kappa)/(np.mean(1/kappa)), bins = 60, color = colorshist[i], alpha = alphahist[i])
            # axkappa3.hist(kappa_flux/np.mean(kappa_flux), bins = 60, color = colorshist[i], alpha = alphahist[i])

        elif which_distrib == 'Cum':
            log_den_cgs = np.sort(log_den_cgs)
            log_T = np.sort(log_T)
            log_kappa = np.sort(log_kappa)
            # kappa = np.sort(kappa)
            log_alpha = np.sort(log_alpha)
            cum_den = np.arange(len(log_den_cgs))/len(log_den_cgs)
            cum_T = np.arange(len(log_T))/len(log_T)
            # cum_kappa = np.arange(len(kappa))/len(kappa)
            cum_kappa = np.arange(len(log_kappa))/len(log_kappa)
            cum_alpha = np.arange(len(log_alpha))/len(log_alpha)
            ax1.plot(log_den_cgs, cum_den, color = colorshist[i], linewidth = 2, label = f'{check_name[i]}')
            ax2.plot(log_T, cum_T, color = colorshist[i], linewidth = 2)
            ax3.plot(log_kappa, cum_kappa, color = colorshist[i], linewidth = 2)
            ax4.plot(log_alpha, cum_alpha, color = colorshist[i], linewidth = 2)

        # Ticks and labels
        new_ticks_d = np.arange(-15, 1, 1) 
        mid_d = (new_ticks_d[:-1] + new_ticks_d[1:]) / 2
        all_ticks_d = np.sort(np.concatenate([new_ticks_d, mid_d]))
        mid_d = (all_ticks_d[:-1] + all_ticks_d[1:]) / 2
        all_ticks_d = np.concatenate([all_ticks_d, mid_d])
        labels_d = [f'{int(tick)}' if tick in new_ticks_d else '' for tick in all_ticks_d]
        ax1.set_xticks(all_ticks_d)
        ax1.set_xticklabels(labels_d)
        new_ticks_t = np.arange(3, 11, 1)  
        mid_t = (new_ticks_t[:-1] + new_ticks_t[1:]) / 2
        all_ticks_t = np.concatenate([new_ticks_t, mid_t])
        labels_t = [f'{tick:.1f}' if tick in new_ticks_t else '' for tick in all_ticks_t]
        ax2.set_xticklabels(labels_t) 
        new_ticks_k = np.arange(-2, 4, .5)
        mid_k = (new_ticks_k[:-1] + new_ticks_k[1:]) / 2
        all_ticks_k = np.concatenate([new_ticks_k, mid_k])
        labels_k = [f'{tick:.1f}' if tick in new_ticks_k else '' for tick in all_ticks_k]
        ax3.set_xticks(all_ticks_k)
        ax3.set_xticklabels(labels_k)
        new_ticks_alpha = np.arange(-15, 7, 1) 
        mid_alpha = (new_ticks_alpha[:-1] + new_ticks_alpha[1:]) / 2
        all_ticks_alpha = np.sort(np.concatenate([new_ticks_alpha, mid_alpha]))
        mid_alpha = (all_ticks_alpha[:-1] + all_ticks_alpha[1:]) / 2
        all_ticks_alpha = np.concatenate([all_ticks_alpha, mid_alpha])
        labels_alpha = [f'{int(tick)}' if tick in new_ticks_alpha else '' for tick in all_ticks_alpha]
        ax4.set_xticklabels(labels_alpha)
        ax4.set_xticks(all_ticks_alpha)

    # axkappa1.legend(fontsize = 16)
    # if which_distrib == 'Histo':
    #     axkappa1.set_xlabel(r'1/mean $\kappa$')
    #     axkappa4.set_xlabel(r'1/mean $L$')
    #     axkappa2.set_xlabel(r'1/mean $\frac{1}{\kappa}$')
    #     axkappa3.set_xlabel(r'1/mean $\frac{F/<F>}{\kappa}$')
    #     axkappa2.set_xlim(0, 5)
    #     axkappa3.set_xlim(0, 5)
    
    print(f'ratio {check_name[0]}/{check_name[1]}:\nfor Ltot: {Lfld[0]/Lfld[1]}\nfor (inverse of) mediankappa = {mediankappa[1]/mediankappa[0]}, \nfor (inverse of) kappaFlux = {kappaFlux[1]/kappaFlux[0]}')
    # ax1.axvline(np.log10(maxRho_tab), color = 'k', linestyle = 'dotted', label = r'max table')
    ax1.axvline(np.log10(minRho_tab), color = 'k', linestyle = '--', label = r'min table')
    ax2.axvline(np.log10(minT_tab), color = 'k', linestyle = '--')
    ax2.axvline(np.log10(maxT_tab), color = 'k', linestyle = '--')
    if which_distrib == 'Cum':
        ax1.set_xlabel(r'$\log_{10}\rho$ [g/cm$^3$]')
        ax2.set_xlabel(r'$\log_{10}$T [K]')
        ax3.set_xlabel(r'$\log_{10}\kappa$ [cm$^2$/g]')
        ax4.set_xlabel(r'$\log_{10}\alpha$ [1/cm]')        
    ax3.axvline(np.log10(0.34), color = 'k', linestyle = '-.')
    ax3.text(np.log10(0.36), .8, r'$\kappa_{\rm scatt}$', fontsize = 25, rotation = 90)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', which='major', width = 1.5, length = 9, color = 'k')  
        if ax in [ax1, ax3]:
            if which_distrib == 'Cum':
                ax.set_ylabel('CDF')
            else:
                ax.set_ylabel('N')
        if which_distrib == 'Cum':
            ax.grid()
            ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.legend(fontsize = 20, loc = 'lower right')
    if where == 'Ph':
        d_min, d_max = -14.5, -9.5
        k_min, k_max = -0.8, 2
        alpha_min, alpha_max = -14.1, -10.5
        fig.suptitle(f'Photospheric cells at t = {np.round(tfb,2)} ' + r't$_{\rm fb}$. Median $R_{ph}$:' + f'{check_name[0]} {int(median_Rph[0])} R$_\odot$, {check_name[1]} {int(median_Rph[1])} R$_\odot$', fontsize = 20)
    if where == 'Mid':
        if change == 'Extr':
            d_min, d_max = -11, -3
            alpha_min, alpha_max = -9, 2
        else:
            d_min, d_max = -13, -2
            alpha_min, alpha_max = -12, 1
        t_min, t_max = 3.5, 6
        k_min, k_max = 0, 6
        fig.suptitle(f'Midplane cells at t = {np.round(tfb,2)} ' + r't$_{\rm fb}$, N$_{\rm cell, 1}$/N$_{\rm cell, 2}=$' + f'{np.round(Ncell_ph[0]/Ncell_ph[1], 2)}', fontsize = 20)
    ax1.set_xlim(d_min, d_max)
    ax2.set_xlim(3.5, 8.5)
    ax3.set_xlim(k_min, k_max)
    # axkappa.set_ylim(k_min, k_max)
    ax4.set_xlim(alpha_min, alpha_max)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/Test/MazeOfRuns/{change}/{which_distrib}{where}runs{checks[0]}{snap}.png', bbox_inches = 'tight')
    figkappa.savefig(f'{abspath}/Figs/Test/MazeOfRuns/{change}/{which_distrib}Kappa{where}runs{checks[0]}{snap}.png', bbox_inches = 'tight')

if choose == 'section':
    radii_grid = [R0, Rt] #*apo 
    xcfr_grid, ycfr_grid, cfr_grid = [], [], []
    for i, radius_grid in enumerate(radii_grid):
        xcr, ycr, cr = orb.make_cfr(radius_grid)
        xcfr_grid.append(xcr)
        ycfr_grid.append(ycr)
        cfr_grid.append(cr)

    Ncell = np.zeros(len(checks))
    Ncell_mid = np.zeros(len(checks))
    fig, ax = plt.subplots(2, 2, figsize = (14, 10))
    if change == 'Extr':
        figK, axK = plt.subplots(1, 2, figsize = (10, 5))
    for i,check in enumerate(checks):
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}' 
        path = f'{abspath}/TDE/{folder}/{snap}'
        tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
        data = make_tree(path, snap, energy = True)
        cut = data.Den > 1e-19
        X, Y, Z, den, Vol, Mass, Diss_den, Temp = \
            sec.make_slices([data.X, data.Y, data.Z,data.Den, data.Vol, data.Mass, data.Diss, data.Temp], cut)
        Ncell[i] = len(data.X)
        print(f'check = {check}, Ncell = {Ncell[i]}')
        dim_cell = Vol**(1/3) 

        midplane = np.abs(Z) < dim_cell
        X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Mass_midplane, Diss_den_midplane, Temp_midplane = \
            sec.make_slices([X, Y, Z, dim_cell, den, Mass, Diss_den, Temp], midplane)
        Ncell_mid[i] = len(X_midplane)
        Den_midplane_cgs = Den_midplane * prel.Msol_cgs / prel.Rsol_cgs**3 # [g/cm^3]

        if change == 'Extr':
            extrapolated_den = np.logical_or(Den_midplane_cgs < minRho_tab, Den_midplane_cgs > maxRho_tab)
            extrapolated_temp = np.logical_or(Temp_midplane < minT_tab, Temp_midplane > maxT_tab)
            extrapoalted_all = np.logical_or(extrapolated_den, extrapolated_temp)
            if check in ['LowRes', '', 'HiRes']:
                T_cool2, Rho_cool2, rosseland2 = opacity_extrap(T_cool, Rho_cool, rosseland, what = 'scattering_limit', slope_length = 5, highT_slope=-3.5)
            if check in ['OpacityNew', 'OpacityNewNewAMR']:
                T_cool2, Rho_cool2, rosseland2 = opacity_linear(T_cool, Rho_cool, rosseland, what = 'scattering_limit', highT_slope = 0)
            sigma_rosseland = eng.interp2(T_cool2, Rho_cool2, rosseland2.T, np.log(Temp_midplane), np.log(Den_midplane_cgs), 'linear', 0)
            sigma_rosseland = np.array(sigma_rosseland)[0]
            sigma_rosseland_eval = np.exp(sigma_rosseland) # [1/cm]
            sigma_rosseland_eval[sigma_rosseland == 0.0] = 1e-20
            kappa_mid = sigma_rosseland_eval/Den_midplane_cgs # [cm^2/g]
            tau_mid = sigma_rosseland_eval * dim_midplane * prel.Rsol_cgs 

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
            img = ax[0][i].scatter(X_midplane/Rt, Y_midplane/Rt, c = Mass_midplane, s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-14, vmax = 5e-10))
            cbar = plt.colorbar(img)
            cbar.set_label(r'Mass [$M_\odot$]', fontsize = 20)

            img = ax[1][i].scatter(X_midplane/Rt, Y_midplane/Rt, c = np.abs(Diss_den_midplane)*prel.en_den_converter/prel.tsol_cgs, s = 1, cmap = 'turbo', norm = colors.LogNorm(vmin = 1e-5, vmax = 5e7))
            cbar = plt.colorbar(img)
            cbar.set_label(r'$|\dot{u_{\rm diss}}|$ [g/cm$^3s]$', fontsize = 20)

            big_ticks = np.array([-2, -1, 0, 1, 2]) # [R_t]
            mid_ticks = (big_ticks[:-1] + big_ticks[1:]) / 2
            all_ticks = np.sort(np.concatenate([big_ticks, mid_ticks]))
            for k in range(2):
                for j in range(2):  
                    # ax[k][j].contour(xcfr_grid[0]/Rt, ycfr_grid[0]/Rt, cfr_grid[0]/Rt, levels=[0], colors='grey')
                    # ax[k][j].contour(xcfr_grid[1]/Rt, ycfr_grid[1]/Rt, cfr_grid[1]/Rt, levels=[0], colors='grey')
                    ax[k][j].axhline(0.6, color = 'grey', linestyle = '--')
                    ax[k][j].axhline(-0.6, color = 'grey', linestyle = '--')
                    ax[k][j].axvline(0.6, color = 'grey', linestyle = '--')
                    ax[k][j].axvline(-0.6, color = 'grey', linestyle = '--')
                    ax[k][j].set_xlim(-2,2)
                    ax[k][j].set_ylim(-2,2)
                    ax[k][j].set_xticks(all_ticks)
                    ax[k][j].set_xticklabels([f'{tick}' if tick in big_ticks else '' for tick in all_ticks])
                    ax[k][j].set_yticks(all_ticks)
                    ax[k][j].set_yticklabels([f'{tick}' if tick in big_ticks else '' for tick in all_ticks])
                    ax[k][j].tick_params(axis='x', which='major', width = 1, length=7)
                    ax[k][j].tick_params(axis='y', which='major', width = 1, length=7)
                    ax[k][j].set_yticks(all_ticks)
                    ax[k][j].set_aspect('equal')
                ax[1][k].set_xlabel(r'X [$R_{\rm t}$]')
                ax[k][0].set_ylabel(r'Y [$R_{\rm t}$]')

        ax[0][i].set_title(f'{check_name[i]}', fontsize = 20)
        

    fig.suptitle(f't = {np.round(tfb,2)}' + r't$_{\rm fb}$', fontsize = 20) #, N_{\rm cell, left}/N_{\rm cell, right}=$' + f'{np.round(Ncell[0]/Ncell[1], 3)}, on midplane: {np.round(Ncell_mid[0]/Ncell_mid[1], 3)}', fontsize = 20)
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/Test/MazeOfRuns/sink/TestLowRes_sink{snap}.png', dpi = 100, bbox_inches = 'tight')
    if change == 'Extr':
        figK.suptitle('Extrapolated cells', fontsize = 20)
        figK.tight_layout()
        # figK.savefig(f'{abspath}/Figs/Test/MazeOfRuns/{change}/TestKextrap{snap}.png', dpi = 100, bbox_inches = 'tight')

    



# %%
