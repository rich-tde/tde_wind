#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:06 2024

@author: konstantinos
"""
import numpy as np

def first_rich_extrap(x, y, K, scatter = None, slope_length = 7, highT_slope = 0, extrarowsx = 101, 
                 extrarowsy = 100):
    ''' 
    Extra/Interpolation as in the last runs of RICH.
    Look at:
    - https://gitlab.com/eladtan/RICH/-/blob/master/source/misc/utils.cpp 
    x: array of ln(T)
    y: array of ln(rho)
    K: array of ln(kappa) [1/cm]
    what, str: either scattering_limit or ''.
               if scattering_limit, brings to opacity always above thomson. It has to be applied for rosseland.
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
    # rho_ext_Ld, slope_rho_Ld, Temp_ext_Ld = [], [], []
    # rho_ext_Hd, slope_rho_Hd, Temp_ext_Hd = [], [], []
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            if xsel < x[0]: # Too cold
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]: # Too rarefied
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = (K[0, slope_length - 1] - K[0, 0]) / deltay
                    # rho_ext.append(ysel)
                    # slope_rho.append(Kyslope)
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                
                    if np.logical_and(highT_slope != -3.5, scatter is not None):
                        # the condition on highT_slope is beacuse for the first runs of RICH we didn't have the thomson limit
                        thomson_this_den = scatter[ix][iy]
                        if Kn[ix][iy] < thomson_this_den:
                            Kn[ix][iy] = thomson_this_den

                elif ysel > y[-1]: # Too dense
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = (K[0, -1] - K[0, -slope_length]) / deltay
                    # rho_ext.append(ysel)
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
                    thomson_this_den = scatter[ix][iy]
                    if Kn[ix][iy] < thomson_this_den:
                        Kn[ix][iy] = thomson_this_den

            else: 
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]: # Too rarefied, Temperature is inside table
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[ix_inK, slope_length - 1] - K[ix_inK, 0]) / deltay
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])

                    if np.logical_and(highT_slope != -3.5, scatter is not None):
                        thomson_this_den =  scatter[ix][iy] # 1/cm
                        if Kn[ix][iy] < thomson_this_den:
                            Kn[ix][iy] = thomson_this_den
                    
                elif ysel > y[-1]:  # Too dense, Temperature is inside table
                    deltay = y[-1] - y[-slope_length]
                    Kyslope = (K[ix_inK, -1] - K[ix_inK, -slope_length]) / deltay
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])

                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]

    return xn, yn, Kn

def linear_rich(x, y, K, scatter = None, extrarowsx = 101, 
                 extrarowsy = 100, slope_length = 7, highT_slope = 0):
    ''' 
    Extra/Interpolation as in the middle-new runs of RICH.
    Look at:
    - https://gitlab.com/eladtan/RICH/-/blob/master/source/misc/utils.cpp 
    - CalcDiffusionCoefficient, which gives you the inverse of Rosseland in https://gitlab.com/eladtan/RICH/-/blob/master/source/Radiation/STAgreyOpacity.cpp 
    x: array of ln(T)
    y: array of ln(rho)
    K: array of ln(kappa) [1/cm]
    what, str: either scattering_limit or ''.
               if scattering_limit, brings to opacity always above thomson. It has to be applied for rosseland.
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
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            if xsel < x[0]: # Too cold
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]: # Too rarefied
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = 1 
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: # Too dense
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = 1 
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
            
            # Too hot
            elif xsel > x[-1]: 
                if ysel < y[0]: # Too rarefied
                    Kxslope = highT_slope #(K[-1, 0] - K[-slope_length, 0]) / deltax
                    Kyslope = 1 
                    Kn[ix][iy] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: # Too dense
                    Kxslope = highT_slope #(K[-1, -1] - K[-slope_length, -1]) / deltax
                    Kyslope = 1 
                    Kn[ix][iy] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = highT_slope #(K[-1, iy_inK] - K[-slope_length, iy_inK]) / deltax
                    Kn[ix][iy] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
                
                if scatter is not None:
                    thomson_this_den = scatter[ix][iy]
                    if Kn[ix][iy] < thomson_this_den:
                        Kn[ix][iy] = thomson_this_den

            else: 
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]: # Too rarefied, Temperature is inside table
                    Kyslope = 1
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])
                    
                elif ysel > y[-1]:  # Too dense, Temperature is inside table
                    Kyslope = 1
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])

                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]

    return xn, yn, Kn

if __name__ == '__main__':
    #%% Test opacities
    abspath = '/Users/paolamartire/shocks/'
    opac_path = f'{abspath}/src/Opacity'
    import sys
    sys.path.append(abspath)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import Utilities.prelude as prel

    save = True
    # Load data (they are the ln of the values)
    T_tab = np.loadtxt(f'{opac_path}/T.txt') 
    Rho_tab = np.loadtxt(f'{opac_path}/rho.txt') 
    rossland_tab = np.loadtxt(f'{opac_path}/ross.txt') # Each row is a fixed T, column a fixed rho
    planck_tab = np.loadtxt(f'{opac_path}/planck.txt') # Each row is a fixed T, column a fixed rho
    scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm

    T_plot_tab = np.exp(T_tab)
    Rho_plot_tab = np.exp(Rho_tab)
    ross_plot_tab = np.exp(rossland_tab)
    min_T, max_T = np.min(T_plot_tab), np.max(T_plot_tab)
    min_Rho, max_Rho = np.min(Rho_plot_tab), np.max(Rho_plot_tab)
    # multiply column i of ross by Rho_plot_tab[i] to get kappa
    ross_rho_tab = []
    for i in range(len(T_plot_tab)):
        ross_rho_tab.append(ross_plot_tab[i, :]/Rho_plot_tab)
    ross_rho_tab = np.array(ross_rho_tab)
    planck_plot_tab = np.exp(planck_tab)
    scatt = 0.2*(1+0.7381) * Rho_plot_tab #1/cm
    #%%
    # check slopes from table
    den_exp_ross_Elad = np.zeros(len(T_plot_tab))
    den_exp_planck_Elad = np.zeros(len(T_plot_tab))
    for i in range(len(T_plot_tab)):
        Rho_for_fit = Rho_tab 
        Ross_for_fit = rossland_tab[i,:]
        Planck_for_fit = planck_tab[i,:] #np.log10(planck_plot_tab[i,:])
        den_exp_ross_Elad[i] = (Ross_for_fit[9]-Ross_for_fit[0])/(Rho_for_fit[9]-Rho_for_fit[0]) # this is the slope of the line between the first and the 10th point in Elad's table
        den_exp_planck_Elad[i] = (Planck_for_fit[9]-Planck_for_fit[0])/(Rho_for_fit[9]-Rho_for_fit[0]) # this is the slope of the line between the first and the 10th point in Elad's table
    # do the same for T
    den_exp_T_ross_Elad = np.zeros(len(Rho_plot_tab))
    den_exp_T_planck_Elad = np.zeros(len(Rho_plot_tab))
    for i in range(len(Rho_plot_tab)):
        T_for_fit = T_tab 
        Ross_for_fit = rossland_tab[:,i]
        Planck_for_fit = planck_tab[:,i]
        den_exp_T_ross_Elad[i] = (Ross_for_fit[9]-Ross_for_fit[0])/(T_for_fit[9]-T_for_fit[0])
        den_exp_T_planck_Elad[i] = (Planck_for_fit[9]-Planck_for_fit[0])/(T_for_fit[9]-T_for_fit[0])
    # Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))
    ax1.plot(T_plot_tab, den_exp_ross_Elad,  c = 'darkviolet', label = 'Rosseland')
    ax1.plot(T_plot_tab, den_exp_planck_Elad, c = 'b', label = 'Planck')
    ax1.set_xlabel(r'$T$ [K]')
    ax1.set_ylabel(r'$\kappa(\rho)$ slope')
    ax1.set_xscale('log')
    ax1.set_xlim(1e3, 1e8)
    ax1.set_ylim(0.4, 2.8)
    ax1.legend()
    ax1.set_title('Opacity slope for low density')
    ax2.plot(Rho_plot_tab, den_exp_T_ross_Elad,  c = 'darkviolet', label = 'Rosseland')
    ax2.plot(Rho_plot_tab, den_exp_T_planck_Elad, c = 'b', label = 'Planck')
    ax2.set_xlabel(r'$\rho$ [g/cm$^3$]')
    ax2.set_ylabel(r'$\kappa(T)$ slope')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.set_title('Opacity slope for low temperature')
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/Test/extrap_kappa_slope.png')

    #%% Extrapolate
    _, _, scatter2 = linear_rich(T_tab, Rho_tab, scattering, slope_length = 7, highT_slope = 0)
    scatter2_plot = np.exp(scatter2) # 1/cm

    T_RICH, Rho_RICH, rosslandRICH = \
        first_rich_extrap(T_tab, Rho_tab, rossland_tab, scatter = scatter2, slope_length = 5, highT_slope=-3.5)
    T_plotRICH = np.exp(T_RICH)
    Rho_plotRICH = np.exp(Rho_RICH)
    ross_plotRICH = np.exp(rosslandRICH)
    T_linRICH, Rho_linRICH, ross_linRICH = \
        linear_rich(T_tab, Rho_tab, rossland_tab, scatter = scatter2, highT_slope=0)
    # T_linRICH, Rho_linRICH, ross_linRICH = \
    #     first_rich_extrap(T_tab, Rho_tab, rossland_tab, what = 'scattering_limit', slope_length = 5, highT_slope=0)
    T_plotRICH_lin = np.exp(T_linRICH)
    Rho_plotRICH_lin = np.exp(Rho_linRICH)
    ross_plotRICH_lin = np.exp(ross_linRICH)

    _, _, rossland_finalRICH = \
        first_rich_extrap(T_tab, Rho_tab, rossland_tab, scatter = scatter2, slope_length = 7, highT_slope= 0)
    ross_plot_finalRICH = np.exp(rossland_finalRICH)

    # fixed T
    chosenTs = [1e4, 1e5, 1e7] #all inside the table
    fig, ax = plt.subplots(1,3, figsize = (15,5))
    for i,chosenT in enumerate(chosenTs):
        iT = np.argmin(np.abs(T_plot_tab - chosenT))
        ax[i].plot(Rho_plot_tab, ross_plot_tab[iT, :]/Rho_plot_tab, c = 'k', label = 'original')
        iT_4 = np.argmin(np.abs(T_plotRICH - chosenT))
        ax[i].plot(Rho_plotRICH, ross_plotRICH[iT_4, :]/Rho_plotRICH, '--', c = 'yellowgreen', label = 'old extrapolation')
        # print the angular coefficien of the line above
        # idx_overcome = np.where(Rho_plotRICH>np.max(Rho_plot_tab))[0][0]
        # print(np.gradient(np.log(ross_plotRICH[iT_4, idx_overcome:]), np.log(Rho_plotRICH[idx_overcome:])))
        iT_lin = np.argmin(np.abs(T_plotRICH_lin - chosenT))
        ax[i].plot(Rho_plotRICH_lin, ross_plotRICH_lin[iT_lin, :]/Rho_plotRICH_lin, ':', c = 'orchid', label = 'new extrapolation')
        ax[i].plot(Rho_plot_tab, scatt/Rho_plot_tab,  color = 'r', linestyle = '--', label = 'scattering')
        # ax[i].set_ylim(.2, 1e4)
        # ax[i].set_xlim(5e-1, 1e-12)
        ax[i].set_xlabel(r'$\rho$ [g/cm$^3$]')
        ax[i].axvline(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3, color = 'grey', linestyle = ':', label = 'simulation cut')
        ax[i].axvline(min_Rho, color = 'grey', linestyle = '--', label = 'lim table')
        ax[i].axvline(max_Rho, color = 'grey', linestyle = '--')
        ax[i].set_title(f'T = {chosenT:.0e} K')        
        ax[i].loglog()
        ax[i].legend()
    ax[0].set_ylabel(r'$\kappa$ [cm$^{2}$/g]')
    plt.tight_layout()

    fig, ax = plt.subplots(1,3, figsize = (15,5))
    for i,chosenT in enumerate(chosenTs):
        iT = np.argmin(np.abs(T_plot_tab - chosenT))
        ax[i].plot(Rho_plot_tab, ross_plot_tab[iT, :], c = 'k', label = 'original')
        iT_RICH = np.argmin(np.abs(T_plotRICH - chosenT))
        ax[i].plot(Rho_plotRICH, ross_plotRICH[iT_RICH, :], '--', c = 'yellowgreen', label = 'old extrapolation')
        # print the angular coefficien of the line above
        # idx_overcome = np.where(Rho_plotRICH>np.max(Rho_plot_tab))[0][0]
        # print(np.gradient(np.log(ross_plotRICH[iT_4, idx_overcome:]), np.log(Rho_plotRICH[idx_overcome:])))
        iT_lin = np.argmin(np.abs(T_plotRICH_lin - chosenT))
        ax[i].plot(Rho_plotRICH_lin, ross_plotRICH_lin[iT_lin, :], ':', c = 'orchid', label = 'new extrapolation')
        ax[i].set_ylim(1e-18, 1e-9)
        ax[i].set_xlim(1e-18, 1e-9)
        ax[i].set_xlabel(r'$\rho$ [g/cm$^3$]')
        ax[i].axvline(1e-19*prel.Msol_cgs/prel.Rsol_cgs**3, color = 'grey', linestyle = ':', label = 'simulation cut')
        ax[i].axvline(min_Rho, color = 'grey', linestyle = '--', label = 'lim table')
        ax[i].axvline(max_Rho, color = 'grey', linestyle = '--')
        # write title exponentially as 1e4, 1e5, 1e7
        ax[i].set_title(f'T = {chosenT:.0e} K')
        ax[i].loglog()
        ax[i].legend()
    ax[0].set_ylabel(r'$\alpha=\kappa\rho$ [1/cm]')
    plt.tight_layout()

    #%% fixed rho
    chosenRhos = [1e-9, 1e-11] # you want 1e-6, 1e-11 kg/m^3 (too far from Elad's table, u want plot it)
    colors_plot = ['forestgreen', 'r']
    lines = ['solid', 'dashed']
    fig, ax = plt.subplots(2,2,figsize = (15,12))
    for i,chosenRho in enumerate(chosenRhos):
        if np.logical_and(chosenRho < max_Rho, chosenRho > min_Rho):
            irho = np.argmin(np.abs(Rho_plot_tab - chosenRho))
            ax[0][i].plot(T_plot_tab, ross_plot_tab[:, irho], c = 'k', label = 'original')
            ax[1][i].plot(T_plot_tab, ross_plot_tab[:, irho]/Rho_plot_tab[irho], c = 'k', label = 'original')
        irho_RICH = np.argmin(np.abs(Rho_plotRICH - chosenRho))
        ax[1][i].plot(T_plotRICH, ross_plotRICH[:, irho_RICH]/Rho_plotRICH[irho_RICH], c = 'yellowgreen', ls = '--', label = r'old extrapolation')
        ax[0][i].plot(T_plotRICH, ross_plotRICH[:, irho_RICH], c = 'yellowgreen', ls = '--', label = 'old extrapolation')
        irho_lin = np.argmin(np.abs(Rho_plotRICH_lin - chosenRho))
        ax[1][i].plot(T_plotRICH_lin, ross_plotRICH_lin[:, irho_lin]/Rho_plotRICH_lin[irho_lin], c = 'orchid', ls = ':', label = 'new extrapolation')
        ax[0][i].plot(T_plotRICH_lin, ross_plotRICH_lin[:, irho_lin], c = 'orchid', ls = ':', label = 'new extrapolation')
        ax[1][i].set_ylim(1e-1, 2e2) #the axis from 7e-4 to 2e1 m2/g
        ax[0][i].set_title(f'Density: {chosenRho:.0e} g/cm$^3$', fontsize = 16)
        for j in range(2):
            ax[j][i].set_xlabel(r'T [K]')
            ax[j][i].set_xlim(1e1,2e8)
            ax[j][i].axvline(min_T, color = 'grey', linestyle = '--', label = 'lim table')
            ax[j][i].axvline(max_T, color = 'grey', linestyle = '--')
            ax[1][j].axhline(0.2 * (1 + prel.X_nf), color = 'grey', linestyle = '--')
            ax[j][i].loglog()
            ax[j][i].grid()
    ax[0][0].set_ylabel(r'$\alpha=\kappa\rho$ [1/cm]')
    ax[1][0].set_ylabel(r'$\kappa$ [cm$^2$/g]')
    ax[0][0].set_ylim(1e-10, 2e-7)
    ax[0][1].set_ylim(1e-13, 2e-9)
    ax[0][0].legend(fontsize=15, loc='upper right')
    plt.tight_layout()

    #%% Mesh
    scatter_rho_RICH = []
    for i in range(len(T_plotRICH)):
        scatter_rho_RICH.append(scatter2_plot[i, :]/Rho_plotRICH)
    scatter_rho_RICH = np.array(scatter_rho_RICH)

    ross_rho_finalRICH = []
    for i in range(len(T_plotRICH)):
        ross_rho_finalRICH.append(ross_plot_finalRICH[i, :]/Rho_plotRICH)
    ross_rho_finalRICH = np.array(ross_rho_finalRICH)

    ross_rho_RICH = []
    for i in range(len(T_plotRICH)):
        ross_rho_RICH.append(ross_plotRICH[i, :]/Rho_plotRICH)
    ross_rho_RICH = np.array(ross_rho_RICH)
    ross_rho_linRICH = []
    for i in range(len(T_plotRICH_lin)):
        ross_rho_linRICH.append(ross_plotRICH_lin[i, :]/Rho_plotRICH_lin)
    ross_rho_linRICH = np.array(ross_rho_linRICH)

    fig, (ax0, axfin, ax1,ax2) = plt.subplots(1,4, figsize = (20,5))
    img = ax0.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), scatter_rho_RICH.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    ax0.set_title('Scattering')

    img = axfin.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), ross_rho_finalRICH.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    axfin.set_title('Final extrap')

    img = ax1.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), ross_rho_RICH.T,  norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    ax1.set_title('Old extrapolation')
    ax0.set_ylabel(r'$\log_{10} \rho$ [g/cm$^3$]')
    img = ax2.pcolormesh(np.log10(T_plotRICH_lin), np.log10(Rho_plotRICH_lin), ross_rho_linRICH.T, norm = LogNorm(vmin = 1e-5, vmax=1e4), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
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
        original_ticksx = ax.get_xticks()
        # Calculate midpoints between each pair of ticks
        midpointsx = np.arange(original_ticksx[0], original_ticksx[-1])
        # Combine the original ticks and midpointsx
        new_ticksx = np.sort(np.concatenate((original_ticksx, midpointsx)))
        labelsx = [str(np.round(tick,2)) if tick in original_ticksx else "" for tick in new_ticksx]   
        ax.set_xticks(new_ticksx)
        ax.set_xticklabels(labelsx)

        original_ticks = ax.get_yticks()
        # Calculate midpoints between each pair of ticks
        midpoints = np.arange(original_ticks[0], original_ticks[-1], 2)
        # Combine the original ticks and midpoints
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]   
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(labels)

        ax.tick_params(axis='x', which='major', width=1.2, length=7, color = 'k')
        ax.tick_params(axis='y', which='major', width=1.2, length=7, color = 'k')
        ax.set_xlabel(r'$\log_{10} T$ [K]')
        ax.set_xlim(0.8,11)
        ax.set_ylim(-19.5,11)
    ax1.legend(fontsize=12, loc='center right')
    plt.tight_layout()
    if save:
        plt.savefig(f'{abspath}/Figs/Test/extrap_kappa_mesh.png')
    #%% check with OPAL opacities 
#     import pandas as pd
#     optable = 'opal0'
#     opal = pd.read_csv(f'{opac_path}/{optable}.txt', sep = '\s+')
#     Tpd, Rhopd, Kpd = opal['t=log(T)'], opal['r=log(rho)'], opal['G=log(ross)']
#     Tpd_plot, Rhopd_plot, Kpd_plot = 10**(Tpd), 10**(Rhopd), 10**(Kpd)

#     # Colormesh
#     fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
#     img = ax1.pcolormesh(np.log10(T_plot_tab), np.log10(Rho_plot_tab), ross_rho_tab.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
#     # cbar = plt.colorbar(img)
#     ax1.set_ylabel(r'$\log_{10} \rho$')
#     ax1.set_title('RICH')
#     ax2.set_xlabel(r'$\log_{10}$ T')

#     img = ax2.scatter(np.log10(Tpd_plot), np.log10(Rhopd_plot), c = Kpd_plot, cmap = 'jet', norm = LogNorm(vmin=1e-5, vmax=1e5))
#     cbar = plt.colorbar(img)
#     ax2.set_xlim(np.log10(np.min(T_plot_tab)), np.log10(np.max(T_plot_tab)))
#     ax2.set_ylim(np.log10(np.min(Rho_plot_tab)), np.log10(np.max(Rho_plot_tab)))
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
#         irho_rich = np.argmin(np.abs(Rho_plotRICH - values[i]))
#         irho_table = np.argmin(np.abs(Rho_plot_tab - values[i]))
#         # print(Rho_plot[irho_table], Rho_plotRICH[irho_rich])

#         ax[i].plot(TOP, KOP, c = 'forestgreen', label = r'OPAL')
#         ax[i].plot(T_plot_tab, ross_rho_tab[:, irho_table], linestyle = '--', c = 'r', label = r'RICH Table')
#         ax[i].plot(T_plotRICH, ross_rhoRICH[:, irho_rich], linestyle = ':', c = 'b', label = r'RICH extrapolation')
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
