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
        file.write('# Time \n')
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
        npanels = 6 # 3 or 6

        # load the data
        data = np.load(f'{abspath}data/{folder}/slices/z{z_chosen}slice_{snap}.npy')
        x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid =\
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]

        if npanels == 3:
            orb_en_onmass_mid = np.abs(orb_en_den_mid/den_mid) * prel.en_converter / prel.Msol_cgs
            ie_onmass_mid = (ie_den_mid/den_mid) * prel.en_converter / prel.Msol_cgs
            Rad_den_mid *= prel.en_den_converter
            fig, ax = plt.subplots(1,3, figsize = (20,5))
            img = ax[0].scatter(x_mid/apo, y_mid/apo, c = orb_en_onmass_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(orb_en_onmass_mid, 5), vmax = np.percentile(orb_en_onmass_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Absolute specific orbital energy [erg/g]', fontsize = 16)
            ax[0].set_ylabel(r'$Y/R_{\rm a}$', fontsize = 20)

            img = ax[1].scatter(x_mid/apo, y_mid/apo, c = ie_onmass_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(ie_onmass_mid, 5), vmax = np.percentile(ie_onmass_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Specific IE [erg/g]', fontsize = 16)

            img = ax[2].scatter(x_mid/apo, y_mid/apo, c = Rad_den_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(Rad_den_mid, 5), vmax = np.percentile(Rad_den_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 16)
            for i in range(npanels):
                ax[i].set_xlabel(r'$X/R_{\rm a}$', fontsize = 20)
                ax[i].set_xlim(-1.2, 0.1)#(-340,25)
                ax[i].set_ylim(-0.5, 0.5)#(-70,70)
            ax[0].text(-1.05, 0.42, f't = {np.round(tfb[idx], 2)}' + r'$t_{\rm fb}$', fontsize = 20)
            ax[0].text(-1.05, 0.38, f'snap {int(snaps[idx])}', fontsize = 16)
            ax[2].text(-.4, 0.42, f'Max: %.2e' % np.max(Rad_den_mid), fontsize = 16)
        
        if npanels == 6:
            if int(snap)<319:
                continue
            orb_en_mid = np.abs(orb_en_den_mid * dim_mid**3) * prel.en_converter
            ie_mid = (ie_den_mid * dim_mid**3) * prel.en_converter
            Rad_mid = (Rad_den_mid * dim_mid**3) * prel.en_converter

            fig, ax = plt.subplots(2,3, figsize = (20,10))
            img = ax[0][0].scatter(x_mid/apo, y_mid/apo, c = den_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(den_mid, 5), vmax = np.percentile(den_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Density [$M_\odot/R_\odot^3$]', fontsize = 14)

            img = ax[0][1].scatter(x_mid/apo, y_mid/apo, c = temp_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(temp_mid, 5), vmax = np.percentile(temp_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'T [K]', fontsize = 14)

            img = ax[0][2].scatter(x_mid/apo, y_mid/apo, c = dim_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(dim_mid, 5), vmax = np.percentile(dim_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Cell size [$R_\odot$]', fontsize = 14)

            img = ax[1][0].scatter(x_mid/apo, y_mid/apo, c = orb_en_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(orb_en_mid, 5), vmax = np.percentile(orb_en_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Absolute specific orbital energy [erg/g]', fontsize = 14)

            img = ax[1][1].scatter(x_mid/apo, y_mid/apo, c = ie_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(ie_mid, 5), vmax = np.percentile(ie_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Specific IE [erg/g]', fontsize = 14)

            img = ax[1][2].scatter(x_mid/apo, y_mid/apo, c = Rad_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = np.percentile(Rad_mid, 5), vmax = np.percentile(Rad_mid, 99)))
            cb = plt.colorbar(img)
            cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 14)
        for i in range(2):
            ax[i][0].set_ylabel(r'$Y/R_{\rm a}$', fontsize = 20)
            for j in range(3):
                ax[i][j].set_xlabel(r'$X/R_{\rm a}$', fontsize = 20)
                ax[i][j].set_xlim(-1.2, 0.1)#(-340,25)
                ax[i][j].set_ylim(-0.5, 0.5)#(-70,70)
        ax[1][0].text(-1.05, 0.42, f't = {np.round(tfb[idx], 2)}' + r'$t_{\rm fb}$', fontsize = 20)
        ax[0][0].text(-1.05, 0.38, f'snap {int(snaps[idx])}', fontsize = 16)

        plt.tight_layout()
        plt.savefig(f'{abspath}Figs/{folder}/slices/Panel{npanels}Slice{snap}.png')
        plt.close()
        
        # plt.show()

    