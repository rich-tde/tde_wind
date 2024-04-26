import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import pickle 
from Utilities.operators import make_tree
import Utilities.prelude
from Utilities.time_extractor import days_since_distruption

choose = 'surface' # 'one'
folder = 'sedov'
save = False
is_tde = False
cross_section = False 
z_chosen = 0
according_apocenter = False
snap = '100'
path = f'{folder}/{snap}'
m = 5
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1

#_, days = days_since_distruption(f'{path}/snap_{snap}.h5', 'tfb')

sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde)
dim_cell = Vol**(1/3)

idx_zone = np.loadtxt(f'data/{snap}/shockzone_{snap}.txt')
idx_zone = [int(i) for i in idx_zone]
X_shock = X[idx_zone]
Y_shock = Y[idx_zone]
Z_shock = Z[idx_zone]
surface = np.loadtxt(f'data/{snap}/shocksurface_{snap}.txt')
idx_surface = [int(i) for i in surface[0]]
X_surf = X[idx_surface]
Y_surf = Y[idx_surface]
Z_surf = Z[idx_surface]
dim_cell_surf = dim_cell[idx_surface]

if according_apocenter:
    apocenter = 2 * Rt * Mbh**(1/3)
else:
    apocenter = 1


plt.figure(figsize=(10,10))
if choose == 'zone':
    if folder == 'TDE':
        img = plt.scatter(X[::200]/apocenter, Y[::200]/apocenter, c = np.log10(Temp[::200]), alpha = 0.5)#, vmin = 2, vmax = 8)
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\log_{10}$Temperature', fontsize = 18)
        plt.plot(X_shock[np.abs(Z_shock-z_chosen)<cross_section]/apocenter, Y_shock[np.abs(Z_shock-z_chosen)<cross_section]/apocenter, 'ks',  markerfacecolor='none', ms=5, markeredgecolor='k', label = 'shock zone')
        plt.xlabel(r'X [x/$R_a$]', fontsize = 18)
        plt.ylabel(r'Y [x/$R_a$]', fontsize = 18)
        plt.ylim(-16, 21)
        plt.xlim(3,29)
    else: 
        plt.plot(X_shock[np.abs(Z_shock-z_chosen)<cross_section], Y_shock[np.abs(Z_shock-z_chosen)<cross_section], 'ks',  markerfacecolor='none', ms=5, markeredgecolor='k', label = 'shock zone')
        plt.xlabel(r'X', fontsize = 18)
        plt.ylabel(r'Y', fontsize = 18)
        plt.ylim(-1, 1)
        plt.xlim(-1,1)
    # plt.title(r'Shock zone (projection) t/t$_{fb}$= ' + f'{np.round(days,3)}', fontsize = 18)
    plt.grid()
    if save:
        plt.savefig(f'Figs/{snap}/shockzone_{snap}.png')
    plt.show()

if choose == 'surface':
    if folder == 'sedov':
        plt.plot(X_surf[np.abs(Z_surf-z_chosen)<dim_cell_surf], Y_surf[np.abs(Z_surf-z_chosen)<dim_cell_surf], 'ks',  markerfacecolor='none', ms=5, markeredgecolor='k', label = 'shock surface')
        plt.xlabel(r'X', fontsize = 18)
        plt.ylabel(r'Y', fontsize = 18)
        plt.ylim(-1, 1)
        plt.xlim(-1,1)
    plt.grid()
    if save:
        plt.savefig(f'Figs/{snap}/shockzone_{snap}.png')
    plt.show()