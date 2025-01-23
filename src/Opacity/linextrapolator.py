#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:06 2024

@author: konstantinos
"""
import numpy as np

# def linearpad(D0,z0):
#     factor = 100
#     dz = z0[-1] - z0[-2]
#     # print(np.shape(D0))
#     dD = D0[:,-1] - D0[:,-2]
    
#     z = [zi for zi in z0]
#     z.append(z[-1] + factor*dz)
#     z = np.array(z)
#     #D = [di for di in D0]

#     to_stack = np.add(D0[:,-1], factor*dD)
#     to_stack = np.reshape(to_stack, (len(to_stack),1) )
#     D = np.hstack((D0, to_stack))
#     #D.append(to_stack)
#     return np.array(D), z

# def pad_interp(x,y,V):
#     Vn, xn = linearpad(V, x)
#     Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
#     Vn = Vn.T
#     Vn, yn = linearpad(Vn, y)
#     Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
#     Vn = Vn.T
#     return xn, yn, Vn

# def new_interp(V, y, extrarows = 60):
#     # Low extrapolation
#     yslope_low = y[1] - y[0]
#     y_extra_low = [y[0] - yslope_low * (i + 1) for i in range(extrarows)]
    
#     # High extrapolation
#     yslope_h = y[-1] - y[-2]
#     y_extra_high = [y[-1] + yslope_h * (i + 1) for i in range(extrarows)]
    
#     # Stack, reverse low to stack properly
#     yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
#     # 2D low
#     Vslope_low = V[1, :] - V[0, :]
#     Vextra_low = [V[0, :] - 10*Vslope_low * (i + 1) for i in range(extrarows)]
    
#     # 2D high
#     Vslope_high = V[-1, :] - V[-2, :]  # Linear difference
#     Vextra_high = [V[-1, :] + Vslope_high * (i + 1) for i in range(extrarows)]

#     Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 

#     return Vn, yn

# def lin_extrapolator(y, V, slope_length, extrarows):
#     # Low extrapolation
#     deltay_low = y[1] - y[0]
#     y_extra_low = [y[0] - deltay_low * (i + 1) for i in range(extrarows)]
#     # High extrapolation
#     deltay_high= y[-1] - y[-2]    
#     y_extra_high = [y[-1] + deltay_high * (i + 1) for i in range(extrarows)]
    
#     # Stack, reverse low to stack properly
#     yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
#     # 2D low
#     yslope_low = y[slope_length - 1] - y[0]
#     Vslope_low = (V[slope_length - 1, :] - V[0, :]) / yslope_low
#     Vextra_low = [V[0, :] + Vslope_low * (y_extra_low[i] - y[0]) for i in range(extrarows)]

#     # 2D high
#     yslope_high = y[-1] - y[-slope_length]
#     Vslope_high = (V[-1, :] - V[-slope_length, :]) / yslope_high
#     Vextra_high = [V[-1, :] + Vslope_high * (y_extra_high[i] - y[-1]) for i in range(extrarows)]
    
#     Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
#     return yn, Vn

# def extrapolator_flipper(x ,y, V, slope_length = 5, extrarowsx = 99, extrarowsy = 100):
#     xn, Vn = lin_extrapolator(x, V, slope_length, extrarowsx) 
#     yn, Vn = lin_extrapolator(y, Vn.T, slope_length, extrarowsy)
#     return xn, yn, Vn.T

def rich_extrapolator(x, y, K, slope_length = 5, extrarowsx= 99, extrarowsy= 100, highT_slope = -3.5):
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
    
    # 2D low
    Kn = np.zeros((len(xn), len(yn)))
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            if xsel < x[0]:
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]:
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = (K[0, slope_length - 1] - K[0, 0]) / deltay
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: #this cover the extrapolation from Elad's code
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = (K[0, -1] - K[0, -slope_length]) / deltay
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1])
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
                continue
            if xsel > x[-1]:
                # deltax = x[-1] - x[-slope_length]
                if ysel < y[0]:
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = highT_slope #(K[-1, 0] - K[-slope_length, 0]) / deltax
                    Kyslope = (K[-1, slope_length - 1] - K[-1, 0]) / deltay
                    Kn[ix][iy] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]: # this cover the interpolation in Elad's code
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = highT_slope #(K[-1, -1] - K[-slope_length, -1]) / deltax
                    Kyslope = (K[-1, -1] - K[-1, -slope_length]) / deltay
                    Kn[ix][iy] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = highT_slope #(K[-1, iy_inK] - K[-slope_length, iy_inK]) / deltax
                    Kn[ix][iy] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
                continue
            if ysel < y[0]: # x is in the ranege of the table, check y
                ix_inK = np.argmin(np.abs(x - xsel))
                deltay = y[slope_length - 1] - y[0]
                Kyslope = (K[ix_inK, slope_length - 1] - K[ix_inK, 0]) / deltay
                Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])
                continue

            ix_inK = np.argmin(np.abs(x - xsel))
            if ysel > y[-1]:
                deltay = y[-1] - y[-slope_length]
                Kyslope = (K[ix_inK, -1] - K[ix_inK, -slope_length]) / deltay
                Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])
                continue
    
            iy_inK = np.argmin(np.abs(y - ysel))
            Kn[ix][iy] = K[ix_inK, iy_inK]
    
    return xn, yn, Kn


def nouveau_rich(x, y, K, what = 'scatter', slope_length = 26, extrarowsx = 100,
                 extrarowsy = 100, highT_slope = -3.5):
    '''
    what, str: either scattering or absorption
    
    should be linear in log for absorption, everywhere,
    for scattering/density should be linear, irregardless of temperature,
    +opacity should never be below thompson'''
    
    X = 0.9082339738214822 # From table prescription
    thompson = 0.2 * (1 + X)
    
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
            
            # Too cold
            if xsel < x[0]:
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]: # Too rarefied
                    # slope_length = 2
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = (K[0, slope_length - 1] - K[0, 0]) / deltay
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                    # if what == 'abs':
                    #    Kn[ix][iy] = K[0, 0] + Kxslope * (x[0] - xsel) + Kyslope * (y[0]-ysel)
                elif ysel > y[-1]: # Too dense
                    deltay = y[-1] - y[-slope_length]
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = (K[0, -1] - K[0, -slope_length]) / deltay
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
                #continue
            
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
                
                if what == 'scattering':
                    thompson_this_den = np.log(thompson * np.exp(ysel)) # 1/cm
                    if Kn[ix][iy] < thompson_this_den:
                        # print(Kn[ix][iy], thompson_this_den)
                        Kn[ix][iy] = thompson_this_den
                #continue
            else:
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]: # Too rarefied, Temperature is inside table
                    # Something fucky is going on here
                    # BS change i make to avoid the line
                    # slope_length = 25
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[ix_inK, slope_length - 1] - K[ix_inK, 0]) / deltay
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])
                    #continue
                elif ysel > y[-1]:  # Too dense, Temperature is inside table
                    deltay = y[-1] - y[-slope_length]
                    Kyslope = (K[ix_inK, -1] - K[ix_inK, -slope_length]) / deltay
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])
                    #continue
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]
                    # continue

    return xn, yn, Kn


if __name__ == '__main__':
    # Test opacities
    abspath = '/Users/paolamartire/shocks/'
    opac_path = f'{abspath}/src/Opacity'
    import sys
    sys.path.append(abspath)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import Utilities.prelude as prel

    #%% Load data (they are the ln of the values)
    T_tab = np.loadtxt(f'{opac_path}/T.txt') 
    Rho_tab = np.loadtxt(f'{opac_path}/rho.txt') 
    rossland_tab = np.loadtxt(f'{opac_path}/ross.txt') # Each row is a fixed T, column a fixed rho
    T_plot_tab = np.exp(T_tab)
    Rho_plot_tab = np.exp(Rho_tab)
    ross_plot_tab = np.exp(rossland_tab)
    scatt = 0.2*(1+0.7381) * Rho_plot_tab #cm^2/g

    # multiply column i of ross by Rho_plot_tab[i] to get kappa
    ross_rho_tab = []
    for i in range(len(T_plot_tab)):
        ross_rho_tab.append(ross_plot_tab[i, :]/Rho_plot_tab)
    ross_rho_tab = np.array(ross_rho_tab)

    # Extrapolate
    T_RICH, Rho_RICH, rosslandRICH = rich_extrapolator(T_tab, Rho_tab, rossland_tab)
    T_plotRICH = np.exp(T_RICH)
    Rho_plotRICH = np.exp(Rho_RICH)
    ross_plotRICH = np.exp(rosslandRICH)
    ross_rhoRICH = []
    for i in range(len(T_plotRICH)):
        ross_rhoRICH.append(ross_plotRICH[i, :]/Rho_plotRICH)
    ross_rhoRICH = np.array(ross_rhoRICH)
    
    #%% fixed T
    chosenTs = [1e4, 1e5, 1e7]
    fig, ax = plt.subplots(1,3, figsize = (15,5))
    for i,chosenT in enumerate(chosenTs):
        iT = np.argmin(np.abs(T_plot_tab - chosenT))
        iT_4 = np.argmin(np.abs(T_plotRICH - chosenT))
        ax[i].plot(Rho_plotRICH, ross_rhoRICH[iT_4, :], ':', label = 'double Extrapolation')
        ax[i].plot(Rho_plot_tab, ross_rho_tab[iT, :], '--', label = 'original')
        ax[i].plot(Rho_plot_tab, scatt/Rho_plot_tab,  color = 'r', linestyle = '--', label = 'scattering')
        ax[i].loglog()
        ax[i].set_ylim(5e-2, 1e4)
        ax[i].set_xlim(1e-18,1e6)
        ax[i].set_xlabel(r'$\rho$')
        ax[i].set_title(f'T = {chosenT} K')
        ax[i].legend()
    ax[0].set_ylabel(r'$\kappa [cm^2g^{-1}]$')
    plt.tight_layout()

    #%% fixed rho
    chosenRhos = [1e-9, 1e-14] # you want 1e-6, 1e-11 kg/m^3 (too far from Elad's table, u want plot it)
    colors_plot = ['forestgreen', 'r']
    lines = ['solid', 'dashed']
    plt.figure(figsize = (10,5))
    for i,chosenRho in enumerate(chosenRhos):
        irho_4 = np.argmin(np.abs(Rho_plotRICH - chosenRho))
        plt.plot(T_plotRICH, ross_rhoRICH[:, irho_4], linestyle = lines[i], c = colors_plot[i], label = r'$\rho$ = '+f'{chosenRho} g/cm3')
    plt.xlabel(r'T')
    plt.ylabel(r'$\kappa [cm^2/g]$')
    plt.ylim(7e-3, 2e2) #the axis from 7e-4 to 2e1 m2/g
    plt.xlim(1e1,1e7)
    plt.loglog()
    plt.legend()
    plt.grid()
    plt.tight_layout()

    #%%
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,5))
    img = ax1.pcolormesh(np.log10(T_plot_tab), np.log10(Rho_plot_tab), ross_rho_tab.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    ax1.set_ylabel(r'$\log_{10} \rho$')
    ax1.set_title('Table')

    img = ax2.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), ross_rhoRICH.T,  norm = LogNorm(vmin = 1e-5, vmax=1e5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\kappa [cm^2/g]$')
    ax2.axvline(np.log10(np.min(T_plot_tab)), color = 'k', linestyle = '--')
    ax2.axvline(np.log10(np.max(T_plot_tab)), color = 'k', linestyle = '--')
    ax2.axhline(np.log10(np.min(Rho_plot_tab)), color = 'k', linestyle = '--')
    ax2.axhline(np.log10(np.max(Rho_plot_tab)), color = 'k', linestyle = '--')
    ax2.set_title('Extrapolation')

    for ax in [ax1, ax2]:
        # Get the existing ticks on the x-axis
        original_ticksx = ax.get_xticks()
        # Calculate midpoints between each pair of ticks
        if ax==ax1:
            midpointsx = (original_ticksx[:-1] + original_ticksx[1:]) / 2
        else:
            midpointsx = np.arange(original_ticksx[0], original_ticksx[-1])
        # Combine the original ticks and midpointsx
        new_ticksx = np.sort(np.concatenate((original_ticksx, midpointsx)))
        labelsx = [str(np.round(tick,2)) if tick in original_ticksx else "" for tick in new_ticksx]   
        ax.set_xticks(new_ticksx)
        ax.set_xticklabels(labelsx)

        original_ticks = ax.get_yticks()
        # Calculate midpoints between each pair of ticks
        if ax==ax1:
            midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        else:
            midpoints = np.arange(original_ticks[0], original_ticks[-1], 2)
        # Combine the original ticks and midpoints
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]   
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(labels)

        ax.tick_params(axis='x', which='major', width=1.6, length=7, color = 'k')
        ax.tick_params(axis='y', which='major', width=1.6, length=7, color = 'k')
        ax.set_xlabel(r'$\log_{10} T$')
        if ax == ax1:
            ax.set_xlim(np.min(np.log10(T_plot_tab)), np.max(np.log10(T_plot_tab)))
            ax.set_ylim(np.min(np.log10(Rho_plot_tab)), np.max(np.log10(Rho_plot_tab)))
        else:
            ax.set_xlim(0.8,11)
            ax.set_ylim(-19,11)

    plt.tight_layout()
    #%% OPAL
    import pandas as pd
    optable = 'opal0'
    opal = pd.read_csv(f'{opac_path}/{optable}.txt', sep = '\s+')
    Tpd, Rhopd, Kpd = opal['t=log(T)'], opal['r=log(rho)'], opal['G=log(ross)']
    Tpd_plot, Rhopd_plot, Kpd_plot = 10**(Tpd), 10**(Rhopd), 10**(Kpd)

    # Colormesh
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
    img = ax1.pcolormesh(np.log10(T_plot_tab), np.log10(Rho_plot_tab), ross_rho_tab.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
    # cbar = plt.colorbar(img)
    ax1.set_ylabel(r'$\log_{10} \rho$')
    ax1.set_title('RICH')
    ax2.set_xlabel(r'$\log_{10}$ T')

    img = ax2.scatter(np.log10(Tpd_plot), np.log10(Rhopd_plot), c = Kpd_plot, cmap = 'jet', norm = LogNorm(vmin=1e-5, vmax=1e5))
    cbar = plt.colorbar(img)
    ax2.set_xlim(np.log10(np.min(T_plot_tab)), np.log10(np.max(T_plot_tab)))
    ax2.set_ylim(np.log10(np.min(Rho_plot_tab)), np.log10(np.max(Rho_plot_tab)))
    ax2.set_xlabel(r'$\log_{10}$ T')
    # ax2.ylabel(r'$\rho [g/cm^3]$')
    cbar.set_label(r'$\kappa [cm^2/g]$')
    ax2.set_title('OPAL')
    plt.tight_layout()
    plt.savefig(f'{abspath}Figs/Test/OPAL/{optable}_mesh.png')

    #%% Line
    values = np.array([1e-9, 1e-5])
    valueslog = np.log10(values)
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    for i,vallog in enumerate(valueslog):
        indOP = np.concatenate(np.where(Rhopd == vallog))
        TOP, KOP = Tpd_plot[indOP], Kpd_plot[indOP]
        irho_rich = np.argmin(np.abs(Rho_plotRICH - values[i]))
        irho_table = np.argmin(np.abs(Rho_plot_tab - values[i]))
        # print(Rho_plot[irho_table], Rho_plotRICH[irho_rich])

        ax[i].plot(TOP, KOP, c = 'forestgreen', label = r'OPAL')
        ax[i].plot(T_plot_tab, ross_rho_tab[:, irho_table], linestyle = '--', c = 'r', label = r'RICH Table')
        ax[i].plot(T_plotRICH, ross_rhoRICH[:, irho_rich], linestyle = ':', c = 'b', label = r'RICH extrapolation')
        ax[i].set_title(r'$\rho$ = '+f'{values[i]} g/cm3')
        ax[i].grid()
        ax[i].loglog()
        ax[i].set_xlim(1e1,1e7)
        ax[i].set_xlabel(r'T [K]')

    ax[0].set_ylabel(r'$\kappa [cm^2/g]$')
    ax[0].set_ylim(7e-3, 2e2) #the axis from 7e-4 to 2e1 m2/g
    ax[1].set_ylim(1e-1, 5e4) #the axis from 7e-4 to 2e1 m2/g
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{abspath}Figs/Test/OPAL/{optable}_lines.png')
# %%
