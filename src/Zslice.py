""" Make slices of the orbital plane.
If alice: make the section (with density cut 1e-19) and save it.
If not alice: load the section and plot it.
"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'

import numpy as np
from Utilities.selectors_for_snap import select_snap
import src.orbits as orb
import Utilities.prelude as prel
import Utilities.sections as sec
from Utilities.isalice import isalice
alice, plot = isalice()
from Utilities.operators import make_tree

##
# CHOICES
##
z_chosen = 0

#
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # 'Low' or 'HiRes'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Mbh = 10**m
Rs = 2*prel.G*Mbh / prel.csol_cgs**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

if alice:
    # get ready to slice and save
    do = True
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
    with open(f'{abspath}/data/{folder}/slices/z{z_chosen}_time.txt','w') as file:
        file.write('#Snap \n') 
        file.write(' '.join(map(str, snaps)) + '\n')
        file.write('#Time \n')
        file.write(' '.join(map(str, tfb)) + '\n')
        file.close()
else:
    # get ready to plot
    do = False
    time = np.loadtxt(f'{abspath}data/{folder}/slices/z{z_chosen}_time.txt')
    snaps = time[0]
    snaps = [int(snap) for snap in snaps]
    tfb = time[1]

for idx, snap in enumerate(snaps):
    if do:
        # you are in alice
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'

        data = make_tree(path, snap, energy = True)
        Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
        vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
        mass, vol, ie_den, Rad_den = data.Mass, data.Vol, data.IE, data.Rad
        orb_en = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh)
        orb_en_den = orb_en/vol
        dim_cell = (3/(4*np.pi) * vol)**(1/3)

        # make Z slices 
        density_cut = data.Den > 1e-19
        midplane_cut = np.abs(data.Z-z_chosen) < dim_cell
        cut = np.logical_and(density_cut, midplane_cut)
        x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid = \
            sec.make_slices([data.X, data.Y, data.Z, dim_cell, data.Den, data.Temp, ie_den, orb_en_den, Rad_den], cut)
        
        np.save(f'{abspath}/data/{folder}/slices/z{z_chosen}slice_{snap}.npy',\
                 [x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid])
        
    else:
        # you are not in alice
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        # choose what to plot
        choice = 'rad' # 'IE' or 'orb' or 'rad' or 'den'

        # load the data
        data = np.load(f'{abspath}data/{folder}/slices/z{z_chosen}slice_{snap}.npy')
        x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid =\
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]
        
        if choice == 'orb': 
            orb_en_onmass_mid = orb_en_den_mid/den_mid
            coloring = np.abs(orb_en_onmass_mid) * prel.en_converter / prel.Msol_to_g
        elif choice == 'IE':
            ie_onmass_mid = ie_den_mid/den_mid
            coloring = ie_onmass_mid * prel.en_converter / prel.Msol_to_g
        elif choice == 'rad':
            coloring = Rad_den_mid * prel.en_den_converter
        elif choice == 'den':
            coloring = den_mid * prel.Msol_to_g / prel.Rsol_to_cm**3

        fig, ax = plt.subplots(1,1, figsize = (14,5))
        img = ax.scatter(x_mid/apo, y_mid/apo, c = coloring, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(coloring, 5), vmax = np.percentile(coloring, 95)))
        cb = plt.colorbar(img)
        ax.set_xlabel(r'$X/R_a$', fontsize = 20)
        ax.set_ylabel(r'$Y/R_a$', fontsize = 20)
        ax.set_xlim(-380/apo, 0.1)#(-340,25)
        ax.set_ylim(-0.5, 0.5)#(-70,70)
        ax.text(-400/apo, -0.4, f't = {np.round(tfb[idx], 2)}' + r'$t_{fb}$', fontsize = 20)
        plt.tight_layout()

        if choice == 'IE':
            cb.set_label(r'Specific IE [erg/g]', fontsize = 16)
            plt.savefig(f'{abspath}Figs/{folder}/slices/midplaneIE_{cut}{snap}.png')
        elif choice == 'orb':
            cb.set_label(r'Absolute specific orbital energy [erg/g]', fontsize = 16)
            plt.savefig(f'{abspath}Figs/{folder}/slices/midplaneorb_{cut}{snap}.png')
        elif choice == 'rad':
            cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 16)
            plt.savefig(f'{abspath}Figs/{folder}/slices/midplaneRad_{cut}{snap}.png')
        elif choice == 'den':
            cb.set_label(r'Density [g/$cm^3$]', fontsize = 16)  
            ax.text(-400/apo, -120/apo, f'snap {int(snaps[idx])}', fontsize = 16)
            plt.savefig(f'{abspath}Figs/{folder}/slices/midplaneDen_{cut}{snap}.png')
        
        # plt.show()
        plt.close()

    