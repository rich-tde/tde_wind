import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from Utilities.basic_units import radians

from Utilities.operators import make_tree, Ryan_sampler, to_cylindric
import Utilities.sections as sec
import src.orbits as orb
from Utilities.time_extractor import days_since_distruption
import Utilities.prelude
matplotlib.rcParams['figure.dpi'] = 150

#
## Parameters
#

#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'HiRes' # 'Low' or 'HiRes' or 'Res20'
snap = '199'
snapL = '199'
threshold =  1/3

#
## Constants
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
saving_path = f'Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

#
## MAIN
#

do = False
plot = True
save = True
compare = True
Ryan =  False
theta_lim =  np.pi
step = 0.02
theta_init = np.arange(-theta_lim, theta_lim, step)
theta_arr = Ryan_sampler(theta_init)

#%% Load data
if do:
    data = make_tree(path, snap, energy = False)
    density = np.load(f'TDE/{folder}{check}/{snap}/smoothed_Den_{snap}.npy')
    R = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    THETA, RADIUS_cyl = to_cylindric(data.X, data.Y)
    V = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
    dim_cell = data.Vol**(1/3) # according to Elad
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')

    #%% Cross section at midplane
    midplane = np.abs(data.Z) < dim_cell
    X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Temp_midplane = \
        sec.make_slices([data.X, data.Y, data.Z, dim_cell, density, data.Temp], midplane)

    # cfr tidal disruption 
    xRt = np.linspace(-Rt, Rt, 100)
    yRt = np.linspace(-Rt, Rt, 100)
    xcfr, ycfr = np.meshgrid(xRt,yRt)
    cfr = xcfr**2 + ycfr**2 - Rt**2

    # cfr smoothing lenght
    xR0 = np.linspace(-R0, R0, 100)
    yR0 = np.linspace(-R0, R0, 100)
    xcfr0, ycfr0 = np.meshgrid(xR0,yR0)
    cfr0 = xcfr0**2 + ycfr0**2 - R0**2

    streamdata_path = f'data/{folder}/stream_{check}{snap}_{step}.npy'
    streamL = np.load(f'data/{folder}/stream_Low{snapL}_{step}.npy')
    indeces_orbitL = streamL[1].astype(int)
    DenL = np.load(f'TDE/{folder}Low/{snapL}/smoothed_Den_{snapL}.npy')
    stream_thresh = DenL[indeces_orbitL]/3
    indeces_stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr = orb.follow_the_stream(data.X, data.Y, data.Z, dim_cell, density, theta_arr, stream_thresh, Rt, path = streamdata_path)

    if save:
        try:
            file = open(f'data/{folder}/width_time{np.round(tfb,1)}_thr{np.round(threshold,1)}.txt', 'r')
            # Perform operations on the file
            file.close()
        except FileNotFoundError:
            with open(f'data/{folder}/width_time{np.round(tfb,1)}_thr{np.round(threshold,1)}.txt','a') as fstart:
                # if file exist
                fstart.write(f'# theta \n')
                fstart.write((' '.join(map(str, theta_arr)) + '\n'))

        with open(f'data/{folder}/width_time{np.round(tfb,1)}_thr{np.round(threshold,1)}.txt','a') as file:
            file.write(f'# {check}, snap {snap} width \n')
            file.write((' '.join(map(str, w_params[0])) + '\n'))
            file.write(f'# {check}, snap {snap} Ncells \n')
            file.write((' '.join(map(str, w_params[1])) + '\n'))
            file.write(f'################################ \n')

        # same for height
        try:
            file = open(f'data/{folder}/height_time{np.round(tfb,1)}_thr{np.round(threshold,1)}.txt', 'r')
            # Perform operations on the file
            file.close()
        except FileNotFoundError:
            with open(f'data/{folder}/height_time{np.round(tfb,1)}_thr{np.round(threshold,1)}.txt','a') as fstart:
                # if file exist
                fstart.write(f'# theta \n')
                fstart.write((' '.join(map(str, theta_arr)) + '\n'))

        with open(f'data/{folder}/height_time{np.round(tfb,1)}_thr{np.round(threshold,1)}.txt','a') as file:
            file.write(f'# {check}, snap {snap} height \n')
            file.write((' '.join(map(str, h_params[0])) + '\n'))
            file.write(f'# {check}, snap {snap} Ncells \n')
            file.write((' '.join(map(str, h_params[1])) + '\n'))
            file.write(f'################################ \n')


#%% 
if plot:
    if do:
        # Plot width over r
        plt.figure(figsize=(6,4))
        plt.plot(theta_arr * radians, w_params[0], c = 'k')
        img = plt.scatter(theta_arr * radians, w_params[0], c = w_params[1], vmin=20, vmax=100, cmap = 'viridis')
        cbar = plt.colorbar(img)
        cbar.set_label(r'Ncells', fontsize = 16)
        plt.xlabel(r'$\theta$', fontsize = 14)
        plt.ylabel(r'Width [$R_\odot$]', fontsize = 14)
        plt.xlim(-3/4*np.pi, 3/4*np.pi)
        plt.ylim(-5,20)
        plt.grid()
        plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,3)) + f', check: {check}, threshold: {np.round(threshold,1)}', fontsize = 16)
        plt.tight_layout()
        plt.show()

    if compare:
        datawidth5 = np.loadtxt(f'data/{folder}/width_time0.5_thr{np.round(threshold,1)}.txt')
        theta_width = datawidth5[0]
        widthL5 = datawidth5[1]
        NcellL5 = datawidth5[2]
        widthHiRes5 = datawidth5[3]
        NcellHiRes5 = datawidth5[4]
        widthRes205 = datawidth5[5]
        NcellRes205 = datawidth5[6]
        datawidth7 = np.loadtxt(f'data/{folder}/width_time0.7_thr{np.round(threshold,1)}.txt')
        widthL7 = datawidth7[1]
        NcellL7 = datawidth7[2]
        widthHiRes7 = datawidth7[3]
        NcellHiRes7 = datawidth7[4]

        dataheight5 = np.loadtxt(f'data/{folder}/height_time0.5_thr{np.round(threshold,1)}.txt')
        theta_height = dataheight5[0]
        heightL5 = dataheight5[1]
        NhcellL5 = dataheight5[2]
        heightHiRes5 = dataheight5[3]
        NhcellHiRes5 = dataheight5[4]
        heightRes205 = dataheight5[5]
        NhcellRes205 = dataheight5[6]
        dataheight7 = np.loadtxt(f'data/{folder}/height_time0.7_thr{np.round(threshold,1)}.txt')
        heightL7 = dataheight7[1]
        NhcellL7 = dataheight7[2]
        heightHiRes7 = dataheight7[3]
        NhcellHiRes7 = dataheight7[4]

        fig, ax = plt.subplots(2,1, figsize = (10,7))
        ax[0].plot(theta_width, widthL5, '--', c = 'k', label = 'Low 0.5')
        ax[0].plot(theta_width, widthL7, c = 'k', label = 'Low 0.7')
        ax[0].plot(theta_width, widthHiRes5, '--', c = 'r', label = 'Middle 0.5')
        ax[0].plot(theta_width, widthHiRes7, c = 'r', label = 'Middle 0.7')
        ax[0].plot(theta_width, widthRes205, '--', c = 'b',  label = 'High 0.5')
        ax[0].legend()
        ax[0].set_ylabel(r'Width [$R_\odot$]', fontsize = 14)
        ax[0].set_xlim(theta_width[30], theta_width[230])
        ax[0].set_ylim(0,20)
        ax[0].grid()
        ax[1].plot(theta_width, NcellL5, '--', c = 'k', label = 'Low 0.5')
        ax[1].plot(theta_width, NcellL7, c = 'k', label = 'Low 0.7')
        ax[1].plot(theta_width, NcellHiRes5,  '--', c = 'r', label = 'Middle 0.5')
        ax[1].plot(theta_width, NcellHiRes7, c = 'r', label = 'Middle 0.7')
        ax[1].plot(theta_width, NcellRes205, '--', c = 'b',  label = 'High 0.5')
        ax[1].legend()
        ax[1].set_xlim(theta_width[30], theta_width[230])
        ax[1].set_ylim(0,80)
        ax[1].set_xlabel(r'$\theta$', fontsize = 14)
        ax[1].set_ylabel(r'N$_{cell}$', fontsize = 14)
        ax[1].grid()
        plt.suptitle(f'Threshold: {np.round(threshold,1)} of Low maxima density', fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/width_comparison_thr{np.round(threshold,1)}.png')
        plt.show()

        fig, ax = plt.subplots(2,1, figsize = (10,7))
        ax[0].plot(theta_height, heightL5, '--', c = 'k', label = 'Low 0.5')
        ax[0].plot(theta_height, heightL7, c = 'k', label = 'Low 0.7')
        ax[0].plot(theta_height, heightHiRes5, '--', c = 'r', label = 'Middle 0.5')
        ax[0].plot(theta_height, heightHiRes7, c = 'r', label = 'Middle 0.7')
        ax[0].plot(theta_height, heightRes205, '--', c = 'b',  label = 'High 0.5')
        ax[0].legend()
        ax[0].set_xlabel(r'$\theta$', fontsize = 14)
        ax[0].set_ylabel(r'Height [$R_\odot$]', fontsize = 14)
        ax[0].set_xlim(theta_height[30], theta_height[230])
        ax[0].set_ylim(0,8)
        ax[0].grid()
        ax[1].plot(theta_height, NhcellL5, '--', c = 'k', label = 'Low 0.5')
        ax[1].plot(theta_height, NhcellL7, c = 'k', label = 'Low 0.7')
        ax[1].plot(theta_height, NhcellHiRes5,  '--', c = 'r', label = 'Middle 0.5')
        ax[1].plot(theta_height, NhcellHiRes7, c = 'r', label = 'Middle 0.7')
        ax[1].plot(theta_height, NhcellRes205, '--', c = 'b',  label = 'High 0.5')
        ax[1].legend()
        ax[1].set_xlim(theta_height[30], theta_height[230])
        ax[1].set_ylim(0,50)
        ax[1].set_xlabel(r'$\theta$', fontsize = 14)
        ax[1].set_ylabel(r'N$_{cell}$', fontsize = 14)
        ax[1].grid()
        plt.suptitle(f'Threshold: {np.round(threshold,1)} of Low maxima density', fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/H_comparison_thr{np.round(threshold,1)}.png')
        plt.show()

        if Ryan:
            diff5 = widthL5 - widthHiRes5
            diff5Res20 = widthL5 - widthRes205
            diff5Res20middle = widthHiRes5 - widthRes205
            diff7 = widthL7 - widthHiRes7
            diffh5 = heightL5 - heightHiRes5
            diffh5Res20 = heightL5 - heightRes205
            diffh5Res20middle = heightHiRes5 - heightRes205
            diffh7 = heightL7 - heightHiRes7
            fig, ax = plt.subplots(2, 1,  figsize=(8,6))
            ax[0].plot(theta_width, diff5, '--', c = 'r', label = r'Low - Middle t/t$_{fb}=$ 0.5')
            ax[0].plot(theta_width, diff5Res20middle, '--', c = 'b', label = r'Middle - High t/t$_{fb}=$ 0.5')
            ax[0].plot(theta_width, diff7, c = 'r', label = r'Low - Middle t/t$_{fb}=$ 0.7')
            ax[0].set_xlabel(r'$\theta$', fontsize = 14)
            ax[0].set_ylabel(r'$\Delta_{ref}-\Delta$', fontsize = 14)
            ax[0].set_xlim(theta_width[30], theta_width[230])
            ax[0].set_ylim(-5,5)
            ax[0].legend()
            ax[0].grid()
            ax[1].plot(theta_height, diffh5, '--', c = 'r', label = r'Low - Middle t/t$_{fb}=$ 0.5')
            ax[1].plot(theta_height, diffh5Res20middle, '--', c = 'b', label = r'Middle - High t/t$_{fb}=$ 0.5')
            ax[1].plot(theta_height, diffh7, c = 'r', label = r'Low - Middle t/t$_{fb}=$ 0.7')
            ax[1].set_xlabel(r'$\theta$', fontsize = 14)
            ax[1].set_ylabel(r'$H_{ref} - H$', fontsize = 14)
            ax[1].set_xlim(theta_height[30], theta_height[230])
            ax[1].set_ylim(-2,5)
            ax[1].legend()
            ax[1].grid()
            plt.suptitle(f'Threshold: {np.round(threshold,1)} of Low maxima density', fontsize = 16)
            if save:
                plt.savefig(f'Figs/{folder}/diffWH_thr{np.round(threshold,1)}.png')
            plt.show()

        else: 
            ratio5 = 1 - widthHiRes5/widthL5
            ratio5Res20 = 1 - widthRes205/widthL5
            ratio5Res20middle = 1 - widthRes205/widthHiRes5
            ratio7 = 1- widthHiRes7/widthL7
            ratioh5 = 1 - heightHiRes5/heightL5
            ratioh5Res20 = 1 - heightRes205/heightL5
            ratioh5Res20middle = 1 - heightRes205/heightHiRes5
            ratioh7 = 1- heightHiRes7/heightL7

            plt.figure(figsize=(8,6))
            plt.plot(theta_width, ratio5, '--', c = 'r', label = r'Middle - Low t/t$_{fb}=$ 0.5')
            # plt.plot(theta_width, ratio5Res20, '--', c = 'green', label = r'High - Low t/t$_{fb}=$ 0.5')
            plt.plot(theta_width, ratio5Res20middle, '--', c = 'b', label = r'High - Middle t/t$_{fb}=$ 0.5')
            plt.plot(theta_width, ratio7, c = 'r', label = r'Middle - Low t/t$_{fb}=$ 0.7')
            plt.xlabel(r'$\theta$', fontsize = 14)
            plt.ylabel(r'1-$\Delta/\Delta_{ref}$', fontsize = 14)
            plt.xlim(theta_width[30], theta_width[230])
            plt.ylim(-0.6,0.95)
            plt.legend()
            plt.grid()
            plt.suptitle(f'Threshold: {np.round(threshold,1)} of Low maxima density', fontsize = 16)
            if save:
                plt.savefig(f'Figs/{folder}/Deltawidth_thr{np.round(threshold,1)}.png')
            plt.show()

            plt.figure(figsize=(8,6))
            plt.plot(theta_height, ratioh5, '--', c = 'r', label = r'Middle - Low t/t$_{fb}=$ 0.5')
            #plt.plot(theta_height, ratioh5Res20, '--', c = 'b', label = r'High - Low t/t$_{fb}=$ 0.5')
            plt.plot(theta_height, ratioh5Res20middle, '--', c = 'b', label = r'High - Middle t/t$_{fb}=$ 0.5')
            plt.plot(theta_height, ratioh7, c = 'r', label = r'Middle - Low t/t$_{fb}=$ 0.7')
            plt.xlabel(r'$\theta$', fontsize = 14)
            plt.ylabel(r'1-$H/H_{ref}$', fontsize = 14)
            plt.xlim(theta_height[30], theta_height[230])
            plt.ylim(-0.7,1)
            plt.legend()
            plt.grid()
            plt.suptitle(f'Threshold: {np.round(threshold,1)} of Low maximum density', fontsize = 16)
            if save:
                plt.savefig(f'Figs/{folder}/DeltaH_thr{np.round(threshold,1)}.png')
            plt.show()

    