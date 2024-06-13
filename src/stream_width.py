import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import os
from Utilities.basic_units import radians

from Utilities.operators import make_tree
import Utilities.sections as sec
import Utilities.orbits as orb
from Utilities.time_extractor import days_since_distruption
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
snap = '164'
is_tde = True
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

##
# FUNCTIONS
##

def parameters_orbit(p, a):
    En = G * Mbh * (p**2 * (a-Rs) - a**2 * (p-Rs)) / ((a**2-p**2) * (p-Rs) * (a-Rs))
    L = np.sqrt(2 * a**2 * (En + G*Mbh/(a-Rs)))
    return En, L

def solvr(x, theta):
    _, L = parameters_orbit(Rp, apo)
    u,y = x
    res =  np.array([y, (-u + G * Mbh / ((1 - Rs*u) * L)**2)])
    return res

def Witta_orbit(theta_data):
    u,y = odeint(solvr, [0, 0], theta_data).T 
    r = 1/u
    return r
#
## MAIN
#

do = True
plot = True
save = False
compare = False
theta_lim =  np.pi
step = 0.1
theta_params = [-theta_lim, theta_lim, step]

#%% Load data
data = make_tree(path, snap, is_tde, energy = False)
R = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
THETA, RADIUS_cyl = orb.to_cylindric(data.X, data.Y)
V = np.sqrt(data.VX**2 + data.VY**2 + data.VZ**2)
dim_cell = data.Vol**(1/3) # according to Elad
tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')

#%% Cross section at midplane
midplane = np.abs(data.Z) < dim_cell
X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Temp_midplane = \
    sec.make_slices([data.X, data.Y, data.Z, dim_cell, data.Den, data.Temp], midplane)

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

# Parabolic Keplerian orbit
ecc = 1
theta_arr_kep = np.arange(0, 2*np.pi, 0.01)
r_orbit = orb.keplerian_orbit(theta_arr_kep, apo, a = Rp, ecc = ecc)
x_K_orbit, y_K_orbit = orb.from_cylindric(theta_arr_kep, r_orbit) 

# Witta
theta_arr_kep = np.arange(-np.pi, np.pi, 0.01)
Witta_r = Witta_orbit(theta_arr_kep)
x_Witta_orbit, y_Witta_orbit = orb.from_cylindric(theta_arr_kep, Witta_r) 

# Density maxima orbit
theta_cm, r_cm = orb.find_maximum(X_midplane, Y_midplane, dim_midplane, Den_midplane, theta_params)
x_cm, y_cm = orb.from_cylindric(theta_cm, r_cm)

#%%
if plot:
    plt.figure(figsize=(7,4))
    #plt.scatter(X_midplane, Y_midplane, c = np.log10(Den_midplane), cmap = 'viridis', s=1, vmin =-8, vmax = -7)
    plt.plot(x_Witta_orbit, y_Witta_orbit, c = 'b',  label='Witta orbit')
    plt.plot(x_K_orbit, y_K_orbit, c = 'r',  label='Keplerian orbit')
    plt.plot(x_cm, y_cm, c = 'g', label='Density maxima')
    plt.xlim(-60,40)
    plt.ylim(-40, 40)
    plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
    plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
    plt.legend()
    plt.show()

#%%
theta_arr, cm, upper_tube, lower_tube, width, ncells  = orb.find_width_stream(X_midplane, Y_midplane, dim_midplane, Den_midplane, theta_params, threshold=threshold)
cm_r = np.sqrt(cm[0]**2 + cm[1]**2)
width_over_r = width / cm_r

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
        file.write((' '.join(map(str, width)) + '\n'))
        file.write(f'# {check}, snap {snap} Ncells \n')
        file.write((' '.join(map(str, ncells)) + '\n'))
        file.write(f'################################ \n')
#%% 
if plot:
    if do:
        vdenmax = 5e-8
        vdenmin = threshold * vdenmax
        plt.figure(figsize = (16,4))
        img = plt.scatter(X_midplane, Y_midplane, c = Den_midplane, s = 0.1, cmap = 'viridis', vmin = vdenmin, vmax = vdenmax)
        plt.contour(xcfr, ycfr, cfr, [0], linestyles = 'dotted', colors = 'k')
        plt.plot(cm[0], cm[1], c = 'k')
        plt.plot(upper_tube[0], upper_tube[1], linestyle = 'dotted', c = 'k')
        plt.plot(lower_tube[0], lower_tube[1],  '--', c = 'k')
        plt.xlim(-apo, 30)
        plt.ylim(-50,70)
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.show()
        # Plot width on density
        # vdenmax = 8e-7
        # vdenmin = threshold * vdenmax
        # fig, ax = plt.subplots(1,2, figsize = (12,5))
        # img = ax[0].scatter(X_midplane, Y_midplane, c = Den_midplane, s = 0.1, cmap = 'viridis', vmin = vdenmin, vmax = vdenmax)
        # cbar= plt.colorbar(img)
        # cbar.set_label(r'Density', fontsize = 16)
        # ax[0].plot(cm[0], cm[1], c = 'k')
        # ax[0].plot(upper_tube[0], upper_tube[1], linestyle = 'dotted', c = 'k')
        # ax[0].plot(lower_tube[0], lower_tube[1],  '--', c = 'k')
        # # ax[0].set_xlim(-60,20)
        # # ax[0].set_ylim(-40,40)
        # ax[0].set_xlim(-apo, -200)
        # ax[0].set_ylim(0,70)
        # ax[0].set_xlabel(r'X [$R_\odot$]', fontsize = 18)
        # ax[0].set_ylabel(r'Y [$R_\odot$]', fontsize = 18)

        # Plot orbits
        # img1 = ax[1].scatter(X_midplane, Y_midplane, c = Den_midplane, s = 0.1, cmap = 'viridis', vmin = vdenmin, vmax = vdenmax)
        # cbar1 = plt.colorbar(img1)
        # cbar1.set_label(r'Density', fontsize = 16)
        # ax[1].plot(x_K_orbit, y_K_orbit, c = 'b', label = 'Keplerian orbit')
        # ax[1].plot(x_Witta_orbit, y_Witta_orbit, c = 'k', linestyle = '--', label = 'Witta orbit')
        # ax[1].plot(cm[0], cm[1], c = 'r', linestyle = '--', label = 'Maxima density')
        # ax[1].set_xlim(-60,20)
        # ax[1].set_ylim(-40, 40)
        # ax[1].set_xlabel(r'X [$R_\odot$]', fontsize = 18)
        # ax[1].legend(loc = 'upper left')
        # plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,3)) + f', threshold: {np.round(threshold, 1)}, check: {check}', fontsize = 16)
        # plt.savefig(f'{saving_path}/width&orb{snap}_thr{np.round(threshold,1)}.png')
        # plt.show()

        # Plot width over r
        plt.figure(figsize=(6,4))
        plt.plot(theta_arr * radians, width, c = 'k')
        img = plt.scatter(theta_arr * radians, width, c = ncells, vmin=20, vmax=100, cmap = 'viridis')
        cbar = plt.colorbar(img)
        cbar.set_label(r'Ncells', fontsize = 16)
        plt.xlabel(r'$\theta$', fontsize = 14)
        plt.ylabel(r'Width [$R_\odot$]', fontsize = 14)
        plt.xlim(-3/4*np.pi, 3/4*np.pi)
        plt.ylim(-5,20)
        plt.grid()
        plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,3)) + f', check: {check}, threshold: {np.round(threshold,1)}', fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{saving_path}/width_theta{snap}_thr{np.round(threshold,1)}.png')
        plt.show()

    if compare:
        datawidth5 = np.loadtxt(f'data/{folder}/width_time0.5_thr{np.round(threshold,1)}.txt')
        theta_width = datawidth5[0]
        widthC5 = datawidth5[1]
        widthHiRes5 = datawidth5[2]
        widthRes205 = datawidth5[3]
        datawidth7 = np.loadtxt(f'data/{folder}/width_time0.7_thr{np.round(threshold,1)}.txt')
        widthC7 = datawidth7[1]
        widthHiRes7 = datawidth7[2]

        plt.figure(figsize=(6,4))
        plt.plot(theta_width, widthC5, '--', c = 'r', label = 'Low 0.5')
        plt.plot(theta_width, widthC7, c = 'r', label = 'Low 0.7')
        plt.plot(theta_width, widthHiRes5, '--', c = 'b', label = 'HiRes 0.5')
        plt.plot(theta_width, widthHiRes7, c = 'b', label = 'HiRes 0.7')
        plt.plot(theta_width, widthRes205, '--', c = 'g',  label = 'Res20 0.5')
        plt.legend()
        plt.xlabel(r'$\theta$', fontsize = 14)
        plt.ylabel(r'Width [$R_\odot$]', fontsize = 14)
        plt.xlim(-3/4*np.pi, 3/4*np.pi)
        plt.ylim(0,15)
        plt.grid()
        plt.suptitle(f'Threshold: {np.round(threshold,1)}', fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/width_comparison_thr{np.round(threshold,1)}.png')
        plt.show()

        ratio5 = 1 - widthHiRes5/widthC5
        ratio5Res20 = 1 - widthRes205/widthC5
        ratio5Res20middle = 1 - widthRes205/widthHiRes5
        ratio7 = 1- widthHiRes7/widthC7

        plt.figure(figsize=(6,4))
        plt.plot(theta_width, ratio5, '--', c = 'b', label = r'Middle - Low t/t$_{fb}=$ 0.5')
        #plt.plot(theta_width, ratio5Res20, '--', c = 'green', label = r'High - Low t/t$_{fb}=$ 0.5')
        plt.plot(theta_width, ratio5Res20middle, '--', c = 'green', label = r'High - Middle t/t$_{fb}=$ 0.5')
        plt.plot(theta_width, ratio7, c = 'b', label = r'Middle - Low t/t$_{fb}=$ 0.7')
        plt.xlabel(r'$\theta$', fontsize = 14)
        plt.ylabel(r'1-$H_{hr}/H_{lr}$', fontsize = 14)
        plt.xlim(-3/4*np.pi, 3/4*np.pi)
        plt.ylim(-0.2,0.95)
        plt.legend()
        plt.grid()
        plt.suptitle(f'Threshold: {np.round(threshold,1)}', fontsize = 16)
        if save:
            plt.savefig(f'Figs/{folder}/Deltawidth_thr{np.round(threshold,1)}.png')
        plt.show()
# %%
theta_arr, cm, upper_tube, lower_tube, width, ncells  = orb.find_width_stream(X_midplane, Y_midplane, dim_midplane, Den_midplane, theta_params, threshold=threshold)
cm_r = np.sqrt(cm[0]**2 + cm[1]**2)
width_over_r = width / cm_r