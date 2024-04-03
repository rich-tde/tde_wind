import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import pickle 
from Utilities.operators import make_tree
import Utilities.prelude
from Utilities.time_extractor import days_since_distruption

choose = 'zone' # 'surface'
folder = 'TDE'
is_tde = True
cross_section = False 
z_chosen = 0
according_apocenter = False
snap = '196'
path = f'{folder}/{snap}'
m = 5
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1

_, days = days_since_distruption(f'{path}/snap_{snap}.h5', 'tfb')

sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde)

zone = np.loadtxt(f'data/{snap}/shockzone_{snap}.txt')
X_shock = zone[0]
Y_shock = zone[1]
Z_shock = zone[2]
#surface = np.loadtxt(f'data/{snap}/shocksurface_{snap}.txt')

if according_apocenter:
    apocenter = 2 * Rt * Mbh**(1/3)
else:
    apocenter = 1

if cross_section:
    delta = 0.3
else:
    delta  = 1e6

plt.figure(figsize=(10,10))
if choose == 'zone':
    img = plt.scatter(X[::200]/apocenter, Y[::200]/apocenter, c = np.log10(Temp[::200]), alpha = 0.5)#, vmin = 2, vmax = 8)
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\log_{10}$Temperature', fontsize = 18)
    plt.plot(X_shock[np.abs(Z_shock-z_chosen)<delta]/apocenter, Y_shock[np.abs(Z_shock-z_chosen)<delta]/apocenter, 'ks',  markerfacecolor='none', ms=5, markeredgecolor='k', label = 'shock zone')
    plt.xlabel(r'X [x/$R_a$]', fontsize = 18)
    plt.ylabel(r'Y [x/$R_a$]', fontsize = 18)
    plt.ylim(-16, 21)
    plt.xlim(3,29)
    plt.title(r'Shock zone (projection) t/t$_{fb}$= ' + f'{np.round(days,3)}', fontsize = 18)
    plt.grid()
    if cross_section:
        plt.savefig(f'Figs/{snap}/shockzone_{snap}_z{z_chosen}.png')
    else:
        plt.savefig(f'Figs/{snap}/shockzone_{snap}.png')
    plt.show()