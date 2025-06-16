#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:06 2024

@author: konstantinos
"""
import numpy as np

def opacity_extrap(x, y, K, scatter = None, slope_length = 7, highT_slope = 0, extrarowsx = 101, 
                 extrarowsy = 100):
    ''' 
    Extra/Interpolation for opacity both in density and temperature.
    Look at:
    - https://gitlab.com/eladtan/RICH/-/blob/master/source/misc/utils.cpp 
    x: array of ln(T)
    y: array of ln(rho)
    K: array of ln(kappa) [1/cm]
    scatter: either None or interpoalted scattering table in ln (with the same shape of K).
             if != None, brings to opacity always above scattering. It has to be applied for rosseland.
    slope_length, int: position of the other point used for the slope.
    highT_slope, float: slope for high temperature extrapolation.
    extrarowsx/extrarowsy, int: number of rows/columns to extrapolate.
    
    '''
    # Extend x and y, adding data equally space (this suppose x,y as array equally spaced)
    # Low extrapolation
    deltaxn_low = x[1] - x[0]
    deltayn_low = y[1] - y[0] 
    x_extra_low = [x[0] - deltaxn_low * (i + 1) for i in range(extrarowsx)]
    y_extra_low = [y[0] - deltayn_low * (i + 1) for i in range(extrarowsy)]
    
    # High extrapolation
    deltaxn_high = x[-1] - x[-2]
    deltayn_high = y[-1] - y[-2]
    x_extra_high = [x[-1] + deltaxn_high * (i + 1) for i in range(extrarowsx)]
    y_extra_high = [y[-1] + deltayn_high * (i + 1) for i in range(extrarowsy)]
    
    # Stack, reverse low to stack properly
    xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    Kn = np.zeros((len(xn), len(yn)))
    T_scatter = []
    d_scatter = []
    alpha_scatter = []
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            if xsel < x[0]: # Too cold
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]: # Too rarefied
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = (K[0, slope_length - 1] - K[0, 0]) / deltay
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                
                    if np.logical_and(highT_slope != -3.5, scatter is not None):
                        # the condition on highT_slope is beacuse for the first runs of RICH (where highT_slope=-3.5) we didn't have the scatter limit
                        scatter_this_den = scatter[ix][iy]
                        if Kn[ix][iy] < scatter_this_den:
                            Kn[ix][iy] = scatter_this_den
                        T_scatter.append(xsel)
                        d_scatter.append(ysel)
                        alpha_scatter.append(Kn[ix][iy])

                elif ysel > y[-1]: # Too dense
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = (K[0, -1] - K[0, -slope_length]) / deltay
                    # Rho_ext.append(ysel)
                    # slope_rho.append(Kyslope)
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
            
            # Too hot
            elif xsel > x[-1]: 
                if ysel < y[0]: # Too rarefied
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = highT_slope #(K[-1, 0] - K[-slope_length, 0]) / deltax
                    Kyslope = (K[-1, slope_length - 1] - K[-1, 0]) / deltay
                    Kn[ix][iy] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: # Too dense
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = highT_slope #(K[-1, -1] - K[-slope_length, -1]) / deltax
                    Kyslope = (K[-1, -1] - K[-1, -slope_length]) / deltay
                    Kn[ix][iy] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = highT_slope #(K[-1, iy_inK] - K[-slope_length, iy_inK]) / deltax
                    Kn[ix][iy] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
                
                if scatter is not None:
                    scatter_this_den = scatter[ix][iy]
                    if Kn[ix][iy] < scatter_this_den:
                        Kn[ix][iy] = scatter_this_den
                    T_scatter.append(xsel)
                    d_scatter.append(ysel)
                    alpha_scatter.append(Kn[ix][iy])
            else: 
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]: # Too rarefied, Temperature is inside table
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[ix_inK, slope_length - 1] - K[ix_inK, 0]) / deltay
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])

                    if np.logical_and(highT_slope != -3.5, scatter is not None):
                        scatter_this_den =  scatter[ix][iy] # 1/cm
                        if Kn[ix][iy] < scatter_this_den:
                            Kn[ix][iy] = scatter_this_den
                        T_scatter.append(xsel)
                        d_scatter.append(ysel)
                        alpha_scatter.append(Kn[ix][iy])
                    
                elif ysel > y[-1]:  # Too dense, Temperature is inside table
                    deltay = y[-1] - y[-slope_length]
                    Kyslope = (K[ix_inK, -1] - K[ix_inK, -slope_length]) / deltay
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])

                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]

    # check how scatter works
    # if __name__ == '__main__':
    #     xn_real = np.exp(xn)
    #     yn_real = np.exp(yn)
    #     T_scatter_real = np.exp(T_scatter)
    #     d_scatter_real = np.exp(d_scatter)
    #     alpha_scatter_real = np.exp(alpha_scatter) 
    #     alpha_real = np.exp(Kn)
    #     kappa_scatter = alpha_scatter_real/d_scatter_real
    #     kappa_n = []
    #     for m in range((len(xn_real))):
    #         kappa_n.append(alpha_real[m, :] / yn_real)
    #     kappa_n = np.array(kappa_n)
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #     img = ax.pcolormesh(np.log10(xn_real), np.log10(yn_real), kappa_n.T, norm=LogNorm(vmin=1e-5, vmax=1e4), cmap='jet')
    #     plt.colorbar(img, label=r'$\kappa$ [cm$^2$/g]')
    #     plt.scatter(np.log10(T_scatter_real), np.log10(d_scatter_real), c = 'k',  s=4)
    #     ax.axvline(np.log10(min_T), color = 'grey', linestyle = '--', label = 'lim table')
    #     ax.axvline(np.log10(max_T), color = 'grey', linestyle = '--')
    #     ax.axhline(np.log10(min_Rho), color = 'grey', linestyle = '--')
    #     ax.axhline(np.log10(max_Rho), color = 'grey', linestyle = '--')
    #     ax.axhline(np.log10(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3), color = 'grey', linestyle = ':', label = 'simulation cut')
    #     # Get the existing ticks on the x-axis
    #     big_ticks = [-10, -5, 0, 5, 10, 15] #ax.get_xticks()
    #     # Calculate midpoints between each pair of ticks
    #     midpointsx = np.arange(big_ticks[0], big_ticks[-1])
    #     # Combine the original ticks and midpointsx
    #     new_ticksx = np.sort(np.concatenate((big_ticks, midpointsx)))
    #     labelsx = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticksx]   
    #     ax.set_xticks(new_ticksx)
    #     ax.set_xticklabels(labelsx)

    #     big_ticks = [-20, -15, -10, -5, 0, 5, 10, 15] #ax.get_yticks()
    #     # Calculate midpoints between each pair of ticks
    #     midpoints = np.arange(big_ticks[0], big_ticks[-1])
    #     # Combine the original ticks and midpoints
    #     new_ticks = np.sort(np.concatenate((big_ticks, midpoints)))
    #     labels = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticks]   
    #     ax.set_yticks(new_ticks)
    #     ax.set_yticklabels(labels)

    #     ax.tick_params(axis='x', which='major', width=1.2, length=7, color = 'k')
    #     ax.tick_params(axis='y', which='major', width=1.2, length=7, color = 'k')
    #     ax.set_xlabel(r'$\log_{10} T$ [K]')
    #     ax.set_xlim(0.8,11)
    #     ax.set_ylim(-19.5,11)

    return xn, yn, Kn

def opacity_linear(x, y, K, scatter = None, slope_length = 7, highT_slope = 0, 
                extrarowsx = 101, extrarowsy = 100):
    ''' 
    Extra/Interpolation for temperature, linear with slope = 1 for density. 
    It's used for scattering and in some runs for rosseland.
    Look at:
    - https://gitlab.com/eladtan/RICH/-/blob/master/source/misc/utils.cpp 
    - CalcDiffusionCoefficient, which gives you the inverse of Rosseland in https://gitlab.com/eladtan/RICH/-/blob/master/source/Radiation/STAgreyOpacity.cpp 
    x: array of ln(T)
    y: array of ln(rho)
    K: array of ln(kappa) [1/cm]
    scatter: either None or interpoalted scattering table in ln(with the same shape of K).
             if != None, brings to opacity always above scattering. It has to be applied for rosseland.
    slope_length, int: position of the other point used for the slope.
    highT_slope, float: slope for high temperature extrapolation.
    extrarowsx/extrarowsy, int: number of rows/columns to extrapolate.
    '''    
    # Extend x and y, adding data equally space (this suppose x,y as array equally spaced)
    # Low extrapolation
    deltaxn_low = x[1] - x[0]
    deltayn_low = y[1] - y[0] 
    x_extra_low = [x[0] - deltaxn_low * (i + 1) for i in range(extrarowsx)]
    y_extra_low = [y[0] - deltayn_low * (i + 1) for i in range(extrarowsy)]
    
    # High extrapolation
    deltaxn_high = x[-1] - x[-2]
    deltayn_high = y[-1] - y[-2]
    x_extra_high = [x[-1] + deltaxn_high * (i + 1) for i in range(extrarowsx)]
    y_extra_high = [y[-1] + deltayn_high * (i + 1) for i in range(extrarowsy)]
    
    # Stack, reverse low to stack properly
    xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    Kn = np.zeros((len(xn), len(yn)))
    Kyslope = 1
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            if xsel < x[0]: # Too cold
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]: # Too rarefied
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: # Too dense
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
            
            # Too hot
            elif xsel > x[-1]: 
                if ysel < y[0]: # Too rarefied
                    Kxslope = highT_slope #(K[-1, 0] - K[-slope_length, 0]) / deltax
                    Kn[ix][iy] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: # Too dense
                    Kxslope = highT_slope #(K[-1, -1] - K[-slope_length, -1]) / deltax
                    Kn[ix][iy] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = highT_slope #(K[-1, iy_inK] - K[-slope_length, iy_inK]) / deltax
                    Kn[ix][iy] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
                
                if scatter is not None:
                    scatter_this_den = scatter[ix][iy]
                    if Kn[ix][iy] < scatter_this_den:
                        Kn[ix][iy] = scatter_this_den

            else: 
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]: # Too rarefied, Temperature is inside table
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])
                    
                elif ysel > y[-1]:  # Too dense, Temperature is inside table
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])

                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]

    return xn, yn, Kn

if __name__ == '__main__':
    abspath = '/Users/paolamartire/shocks/'
    opac_path = f'{abspath}/src/Opacity'
    import sys
    sys.path.append(abspath)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import Utilities.prelude as prel
    save = True

    # Load data (they are the ln of the values)
    ln_T_tab = np.loadtxt(f'{opac_path}/T.txt') 
    ln_Rho_tab = np.loadtxt(f'{opac_path}/rho.txt') 
    ln_rossland_tab = np.loadtxt(f'{opac_path}/ross.txt') # Each row is a fixed T, column a fixed rho
    ln_planck_tab = np.loadtxt(f'{opac_path}/planck.txt') # Each row is a fixed T, column a fixed rho
    ln_scatt_tab = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm
    T__tab = np.exp(ln_T_tab)
    Rho__tab = np.exp(ln_Rho_tab)
    ross_tab = np.exp(ln_rossland_tab)
    pl_tab = np.exp(ln_planck_tab)
    min_T, max_T = np.min(T__tab), np.max(T__tab)
    min_Rho, max_Rho = np.min(Rho__tab), np.max(Rho__tab)
    kappa_ross_tab = []
    for i in range(len(T__tab)):
        kappa_ross_tab.append(ross_tab[i, :]/Rho__tab)
    kappa_ross_tab = np.array(kappa_ross_tab)
    # multiply column i of ross by Rho_tab[i] to get kappa
    rossrho_tab = []
    for i in range(len(T__tab)):
        rossrho_tab.append(ross_tab[i, :]/Rho__tab)
    rossrho_tab = np.array(rossrho_tab)
    thom_scatt = 0.2*(1+prel.X_nf) * Rho__tab #1/cm

    #%%
    # check slopes from table
    den_slopeross = np.zeros(len(ln_T_tab))
    den_slopeplanck = np.zeros(len(ln_T_tab))
    for i in range(len(ln_T_tab)):
        Rho_for_fit = ln_Rho_tab 
        Ross_for_fit = ln_rossland_tab[i,:]
        Planck_for_fit = ln_planck_tab[i,:] 
        den_slopeross[i] = (Ross_for_fit[9]-Ross_for_fit[0])/(Rho_for_fit[9]-Rho_for_fit[0]) # this is the slope of the line between the first and the 10th point in Elad's table
        den_slopeplanck[i] = (Planck_for_fit[9]-Planck_for_fit[0])/(Rho_for_fit[9]-Rho_for_fit[0]) # this is the slope of the line between the first and the 10th point in Elad's table
    # do the same for T
    T_slopeross = np.zeros(len(ln_Rho_tab))
    T_slopeplanck = np.zeros(len(ln_Rho_tab))
    for i in range(len(ln_Rho_tab)):
        T_for_fit = ln_T_tab 
        Ross_for_fit = ln_rossland_tab[:,i]
        Planck_for_fit = ln_planck_tab[:,i]
        T_slopeross[i] = (Ross_for_fit[9]-Ross_for_fit[0])/(T_for_fit[9]-T_for_fit[0])
        T_slopeplanck[i] = (Planck_for_fit[9]-Planck_for_fit[0])/(T_for_fit[9]-T_for_fit[0])
    # Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))
    ax1.plot(T__tab, den_slopeross,  c = 'darkviolet', label = 'Rosseland')
    ax1.plot(T__tab, den_slopeplanck, c = 'b', label = 'Planck')
    ax1.set_xlabel(r'$T$ [K]')
    ax1.set_ylabel(r'$\frac{\kappa_9-\kappa_0}{\rho_9-\rho_0}$')
    ax1.set_xscale('log')
    ax1.set_xlim(1e3, 1e8)
    ax1.set_ylim(0.4, 2.8)
    ax1.legend()
    ax1.set_title('Opacity slope for low density')
    ax2.plot(Rho__tab, T_slopeross,  c = 'darkviolet', label = 'Rosseland')
    ax2.plot(Rho__tab, T_slopeplanck, c = 'b', label = 'Planck')
    ax2.set_xlabel(r'$\rho$ [g/cm$^3$]')
    ax2.set_ylabel(r'$\frac{\kappa_9-\kappa_0}{T_9-T_0}$ ')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.set_title('Opacity slope for low temperature')
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/Test/extrap_kappa_slope.png')

    #%% Extrapolate alpha
    ln_T_extr, ln_Rho_extr, ln_scatter = opacity_linear(ln_T_tab, ln_Rho_tab, ln_scatt_tab, slope_length = 7, highT_slope = 0)
    T_ext = np.exp(ln_T_extr)
    Rho_ext = np.exp(ln_Rho_extr)
    scatter_extr = np.exp(ln_scatter) # 1/cm
    # Old extrapoaltion
    _, _, ln_rossland_Oldextr = \
        opacity_extrap(ln_T_tab, ln_Rho_tab, ln_rossland_tab, scatter = ln_scatter, slope_length = 5, highT_slope=-3.5)
    ross_Oldext = np.exp(ln_rossland_Oldextr)
    # lin A = q
    _, _, ln_ross_lin = \
        opacity_linear(ln_T_tab, ln_Rho_tab, ln_rossland_tab, scatter = ln_scatter, slope_length = 7, highT_slope=0)
    ross_lin = np.exp(ln_ross_lin)
    # Last extrapolation
    _, _, ln_rossland_final = \
        opacity_extrap(ln_T_tab, ln_Rho_tab, ln_rossland_tab, scatter = ln_scatter, slope_length = 7, highT_slope= 0)
    ross_final = np.exp(ln_rossland_final)
    
    # find kappa
    kappa_scatter = []
    kappa_ross_Oldext = []
    kappa_ross_lin = []
    kappa_ross_final = []
    for i in range(len(T_ext)):
        kappa_scatter.append(scatter_extr[i, :]/Rho_ext)
        kappa_ross_Oldext.append(ross_Oldext[i, :]/Rho_ext)
        kappa_ross_lin.append(ross_lin[i, :]/Rho_ext)
        kappa_ross_final.append(ross_final[i, :]/Rho_ext)
    kappa_scatter = np.array(kappa_scatter)
    kappa_ross_Oldext = np.array(kappa_ross_Oldext)
    kappa_ross_lin = np.array(kappa_ross_lin)
    kappa_ross_final = np.array(kappa_ross_final)

    # Plot at fixed T
    chosenTs = [1e4, 1e5, 1e7] #all inside the table
    fig, ax = plt.subplots(1,3, figsize = (15,5))
    for i,chosenT in enumerate(chosenTs):
        iT = np.argmin(np.abs(T__tab - chosenT))
        ax[i].plot(Rho__tab, kappa_ross_tab[iT, :], c = 'k', label = 'original')
        iT_ext = np.argmin(np.abs(T_ext - chosenT))
        ax[i].plot(Rho_ext, kappa_ross_Oldext[iT_ext, :], '--', c = 'yellowgreen', label = 'old extrapolation')
        # print the angular coefficien of the line above
        # idx_overcome = np.where(Rho_ext>np.max(Rho_tab))[0][0]
        # print(np.gradient(np.log(kappa_ross_Oldext[iT_4, idx_overcome:]), np.log(Rho_ext[idx_overcome:])))
        ax[i].plot(Rho_ext, kappa_ross_lin[iT_ext, :], ':', c = 'orchid', label = r'linear in $\rho$ A=1')
        # ax[i].set_ylim(.2, 1e4)
        # ax[i].set_xlim(5e-1, 1e-12)
        ax[i].plot(Rho_ext, kappa_scatter[iT_ext, :], c = 'coral', label = 'scattering')
        ax[i].axhline(0.2 * (1 + prel.X_nf),  color = 'maroon', linestyle = '--', label = 'Thomson scattering')
        ax[i].set_xlabel(r'$\rho$ [g/cm$^3$]')
        ax[i].axvline(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3, color = 'grey', linestyle = ':', label = 'simulation cut')
        ax[i].axvline(min_Rho, color = 'grey', linestyle = '--', label = 'lim table')
        ax[i].axvline(max_Rho, color = 'grey', linestyle = '--')
        ax[i].set_title(f'T = {chosenT:.0e} K')        
        ax[i].loglog()
    ax[0].legend()
    ax[0].set_ylabel(r'$\kappa$ [cm$^{2}$/g]')
    plt.tight_layout()
    #%% fixed rho
    chosenRhos = [1e-9, 1e-11] # you want 1e-6, 1e-11 kg/m^3 (too far from Elad's table, u want plot it)
    colors_plot = ['forestgreen', 'r']
    lines = ['solid', 'dashed']
    fig, ax = plt.subplots(1,2,figsize = (15,6))
    for i,chosenRho in enumerate(chosenRhos):
        if np.logical_and(chosenRho < max_Rho, chosenRho > min_Rho):
            irho = np.argmin(np.abs(Rho__tab - chosenRho))
            ax[i].plot(T__tab, kappa_ross_tab[:, irho], c = 'k', label = 'original')
        i_Rho = np.argmin(np.abs(Rho_ext - chosenRho))
        ax[i].plot(T_ext, kappa_ross_Oldext[:, i_Rho], c = 'yellowgreen', ls = '--', label = r'old extrapolation')
        ax[i].plot(T_ext, kappa_ross_lin[:, i_Rho], c = 'orchid', ls = ':', label = r'linear in $\rho$ A =1')
        ax[i].set_xlabel(r'T [K]')
        ax[i].set_xlim(1e1,2e8)
        ax[i].set_ylim(1e-1, 2e2) #the axis from 7e-4 to 2e1 m2/g
        ax[i].axvline(min_T, color = 'grey', linestyle = '--', label = 'lim table')
        ax[i].axvline(max_T, color = 'grey', linestyle = '--')
        ax[i].plot(T_ext, kappa_scatter[:, i_Rho], c = 'coral', label = 'scattering')
        ax[i].axhline(0.2 * (1 + prel.X_nf), color = 'maroon', linestyle = '--', label = 'Thomson scattering')
        ax[i].loglog()
        ax[i].grid()
        ax[i].set_title(f'Density: {chosenRho:.0e} g/cm$^3$', fontsize = 16)
    ax[1].set_ylabel(r'$\kappa$ [cm$^2$/g]')
    ax[1].legend(fontsize=15, loc='upper right')
    plt.tight_layout()

    #%% Mesh
    fig, (ax0, axfin, ax1,ax2) = plt.subplots(1,4, figsize = (25,7))
    img = ax0.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), kappa_scatter.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    ax0.set_title('Scattering')

    img = axfin.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), kappa_ross_final.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    axfin.set_title('Final extrap')

    img = ax1.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), kappa_ross_lin.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    ax1.set_title('Old extrapolation')
    ax0.set_ylabel(r'$\log_{10} \rho$ [g/cm$^3$]')
    img = ax2.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), kappa_ross_Oldext.T, norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\kappa$ [cm$^2$/g]')
    ax2.set_title('A=1 extrapolation')

    for ax in [ax0, ax1, ax2, axfin]:
        ax.axvline(np.log10(min_T), color = 'grey', linestyle = '--', label = 'lim table')
        ax.axvline(np.log10(max_T), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(min_Rho), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(max_Rho), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3), color = 'grey', linestyle = ':', label = 'simulation cut')
        # Get the existing ticks on the x-axis
        big_ticks = [-10, -5, 0, 5, 10, 15] #ax.get_xticks()
        # Calculate midpoints between each pair of ticks
        midpointsx = np.arange(big_ticks[0], big_ticks[-1])
        # Combine the original ticks and midpointsx
        new_ticksx = np.sort(np.concatenate((big_ticks, midpointsx)))
        labelsx = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticksx]   
        ax.set_xticks(new_ticksx)
        ax.set_xticklabels(labelsx)

        big_ticks = [-20, -15, -10, -5, 0, 5, 10, 15] #ax.get_yticks()
        # Calculate midpoints between each pair of ticks
        midpoints = np.arange(big_ticks[0], big_ticks[-1])
        # Combine the original ticks and midpoints
        new_ticks = np.sort(np.concatenate((big_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticks]   
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(labels)

        ax.tick_params(axis='x', which='major', width=1.2, length=7, color = 'k')
        ax.tick_params(axis='y', which='major', width=1.2, length=7, color = 'k')
        ax.set_xlabel(r'$\log_{10} T$ [K]')
        ax.set_xlim(0.8,11)
        ax.set_ylim(-19.5,11)
    ax0.legend(fontsize=12, loc='center right')
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/Test/extrap_kappa_mesh.png')
    
    #%% check difference with and without scatter
    from Utilities.operators import find_ratio
    _, _, ln_rossland_noscatt = \
        opacity_extrap(ln_T_tab, ln_Rho_tab, ln_rossland_tab, scatter = None, slope_length = 7, highT_slope= 0)
    ross_noscatt = np.exp(ln_rossland_noscatt)
    kappa_ross_noscatt = []
    for i in range(len(T_ext)):
        kappa_ross_noscatt.append(ross_noscatt[i, :]/Rho_ext)
    kappa_ross_noscatt = np.array(kappa_ross_noscatt)

    ratio_scatt = []
    for idx in range(len(kappa_ross_final)):
        ratio_scatt.append(find_ratio(kappa_ross_final[idx], kappa_ross_noscatt[idx]))
    ratio_scatt = np.array(ratio_scatt)

    # import one random photosphere file among the ones with the last extrapolation
    snap = 340
    check = 'LowResNewAMR'
    folder = f'R0.47M0.5BH10000beta1S60n1.5Compton{check}'
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snap_Lum, time = data[:, 0], data[:, 1]
    tfb = time[np.where(snap_Lum == snap)[0][0]]
    _, _, _, _, denph, Tempph, _, _, _, _, alpha, rph, Lph = \
                np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
    fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize = (20,10))
    plt.suptitle('Final Rosseland extrapolation', fontsize = 20)
    img = ax0.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), kappa_ross_final.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) 
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa$ [cm$^2$/g]')
    ax0.set_ylabel(r'$\log_{10} \rho$ [g/cm$^3$]')
    ax0.set_title('with scattering limit', fontsize = 20)

    img = ax1.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), kappa_ross_noscatt.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) 
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'$\kappa$ [cm$^2$/g]')
    ax1.set_title('without scattering limit', fontsize = 20)

    img = ax2.pcolormesh(np.log10(T_ext), np.log10(Rho_ext), ratio_scatt.T, vmin = 1, vmax=2, cmap = 'plasma') 
    cbar = plt.colorbar(img, orientation = 'horizontal')
    cbar.set_label(r'${\rm R}$')
    ax2.set_title('Ratio', fontsize = 20)

    for ax in [ax0, ax1, ax2]:
        ax.scatter(np.log10(Tempph), np.log10(denph), facecolors = 'none', edgecolor = {'k' if ax in [ax0, ax1] else 'white'}, s = 25, label = f'photosphere t={tfb:.2f} ' + r't$_{\rm fb}$')
        ax.axvline(np.log10(min_T), color = 'grey', linestyle = '--', label = 'lim table')
        ax.axvline(np.log10(max_T), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(min_Rho), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(max_Rho), color = 'grey', linestyle = '--')
        ax.axhline(np.log10(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3), color = 'grey', linestyle = ':', label = 'simulation cut')
        # Get the existing ticks on the x-axis
        big_ticks = [-10, -5, 0, 5, 10, 15] #ax.get_xticks()
        # Calculate midpoints between each pair of ticks
        midpointsx = np.arange(big_ticks[0], big_ticks[-1])
        # Combine the original ticks and midpointsx
        new_ticksx = np.sort(np.concatenate((big_ticks, midpointsx)))
        labelsx = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticksx]   
        ax.set_xticks(new_ticksx)
        ax.set_xticklabels(labelsx)

        big_ticks = [-20, -15, -10, -5, 0, 5, 10, 15] #ax.get_yticks()
        # Calculate midpoints between each pair of ticks
        midpoints = np.arange(big_ticks[0], big_ticks[-1])
        # Combine the original ticks and midpoints
        new_ticks = np.sort(np.concatenate((big_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in big_ticks else "" for tick in new_ticks]   
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(labels)

        ax.tick_params(axis='x', which='major', width=1.2, length=7, color = 'k')
        ax.tick_params(axis='y', which='major', width=1.2, length=7, color = 'k')
        ax.set_xlabel(r'$\log_{10} T$ [K]')
        ax.set_xlim(0.8,11)
        ax.set_ylim(-19.5,11)
    ax0.legend(fontsize=12, loc='center right')
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/Test/extrapScatt_kappa_mesh.png')

    
    #%% check with OPAL opacities 
#     import pandas as pd
#     optable = 'opal0'
#     opal = pd.read_csv(f'{opac_path}/{optable}.txt', sep = '\s+')
#     Tpd, Rhopd, Kpd = opal['t=log(T)'], opal['r=log(rho)'], opal['G=log(ross)']
#     Tpd_plot, Rhopd_plot, Kpd_plot = 10**(Tpd), 10**(Rhopd), 10**(Kpd)

#     # Colormesh
#     fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
#     img = ax1.pcolormesh(np.log10(T_tab), np.log10(Rho_tab), ross_rho_tab.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
#     # cbar = plt.colorbar(img)
#     ax1.set_ylabel(r'$\log_{10} \rho$')
#     ax1.set_title('RICH')
#     ax2.set_xlabel(r'$\log_{10}$ T')

#     img = ax2.scatter(np.log10(Tpd_plot), np.log10(Rhopd_plot), c = Kpd_plot, cmap = 'jet', norm = LogNorm(vmin=1e-5, vmax=1e5))
#     cbar = plt.colorbar(img)
#     ax2.set_xlim(np.log10(np.min(T_tab)), np.log10(np.max(T_tab)))
#     ax2.set_ylim(np.log10(np.min(Rho_tab)), np.log10(np.max(Rho_tab)))
#     ax2.set_xlabel(r'$\log_{10}$ T')
#     # ax2.ylabel(r'$\rho [g/cm^3]$')
#     cbar.set_label(r'$\kappa [cm^2/g]$')
#     ax2.set_title('OPAL')
#     plt.tight_layout()
#     if save:
#         plt.savefig(f'{abspath}Figs/Test/OPAL/{optable}_mesh.png')

#     #%% Line
#     values = np.array([1e-9, 1e-5])
#     valueslog = np.log10(values)
#     fig, ax = plt.subplots(1,2, figsize = (10,5))
#     for i,vallog in enumerate(valueslog):
#         indOP = np.concatenate(np.where(Rhopd == vallog))
#         TOP, KOP = Tpd_plot[indOP], Kpd_plot[indOP]
#         iln_Rho_extr = np.argmin(np.abs(Rho_ext - values[i]))
#         irho_table = np.argmin(np.abs(Rho_tab - values[i]))
#         # print(Rho_plot[irho_table], Rho_ext[iln_Rho_extr])

#         ax[i].plot(TOP, KOP, c = 'forestgreen', label = r'OPAL')
#         ax[i].plot(T_tab, ross_rho_tab[:, irho_table], linestyle = '--', c = 'r', label = r'RICH Table')
#         ax[i].plot(T_Oldext, ross_rhoRICH[:, iln_Rho_extr], linestyle = ':', c = 'b', label = r'RICH extrapolation')
#         ax[i].set_title(r'$\rho$ = '+f'{values[i]} g/cm3')
#         ax[i].grid()
#         ax[i].loglog()
#         ax[i].set_xlim(1e1,1e7)
#         ax[i].set_xlabel(r'T [K]')

#     ax[0].set_ylabel(r'$\kappa [cm^2/g]$')
#     ax[0].set_ylim(7e-3, 2e2) #the axis from 7e-4 to 2e1 m2/g
#     ax[1].set_ylim(1e-1, 5e4) #the axis from 7e-4 to 2e1 m2/g
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     if save:
#         plt.savefig(f'{abspath}Figs/Test/OPAL/{optable}_lines.png')

# %%
