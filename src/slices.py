""" Make slices of the orbital plane.
If alice: make the section (with density cut 1e-19) and save it: coordinates, T, Den, cells size, ie density, oe density, rad density, 3 components of velocity.
If not alice: load the section and plot it in 3 (density/specific energies in CGS) or 6 panels (density, T, cells size, energies all in CGS except mass/size).
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

#
##
# CHOICES
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiRes' # 'LowRes' or 'HiRes'
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
coord_to_cut = 'z' # 'x', 'y', 'z'
cut_chosen = 0

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Mbh = 10**m
Rs = 2*prel.G*Mbh / prel.csol_cgs**2
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
if cut_chosen == Rp:
    cut_name = 'Rp'
else:
    cut_name =  int(cut_chosen)

if alice:
    # get ready to slice and save
    do = True
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
    with open(f'{abspath}/data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}_time.txt','w') as file:
        file.write('#Snap \n') 
        file.write(' '.join(map(str, snaps)) + '\n')
        file.write('# Time \n')
        file.write(' '.join(map(str, tfb)) + '\n')
        file.close()
else:
    # get ready to plot
    import healpy as hp
    do = False
    time = np.loadtxt(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}_time.txt')
    snaps = time[0]
    snaps = [int(snap) for snap in snaps]
    tfb = time[1]
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
    observers_xyz = np.array(observers_xyz).T
    x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
    theta = np.arctan2(y, x)          # Azimuthal angle in radians
    phi = np.arccos(z / r)            # Elevation angle in radians
    longitude_moll = theta              
    latitude_moll = np.pi / 2 - phi 
    indecesorbital = np.concatenate(np.where(latitude_moll==0))

ratio_E = np.zeros(len(snaps))
for idx, snap in enumerate(snaps):
    print(snap)
    if do:
        # you are in alice
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'

        data = make_tree(path, snap, energy = True)
        X, Y, Z, den, mass, vol, ie_den, Rad_den, VX, VY, VZ = data.X, data.Y, data.Z, data.Den, data.Mass, data.Vol, data.IE, data.Rad, data.VX, data.VY, data.VZ
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        vel = np.sqrt(np.power(VX, 2) + np.power(VY, 2) + np.power(VZ, 2))
        orb_en = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh)
        orb_en_den = orb_en/vol
        dim_cell = (3/(4*np.pi) * vol)**(1/3)
        if coord_to_cut == 'x':
            cutcoord = X
        elif coord_to_cut == 'y':
            cutcoord = Y
        elif coord_to_cut == 'z':
            cutcoord = Z

        # make slices 
        density_cut = den > 1e-19
        coordinate_cut = np.abs(cutcoord-cut_chosen) < dim_cell
        cut = np.logical_and(density_cut, coordinate_cut)
        # x_cut, y_cut, z_cut, dim_cut, den_cut, temp_cut, ie_den_cut, orb_en_den_cut, Rad_den_cut = \
        #     sec.make_slices([X, Y, Z, dim_cell, den, data.Temp, ie_den, orb_en_den, Rad_den], cut)
        x_cut, y_cut, z_cut, dim_cut, den_cut, temp_cut, ie_den_cut, orb_en_den_cut, Rad_den_cut, VX_cut, VY_cut, VZ_cut = \
            sec.make_slices([X, Y, Z, dim_cell, den, data.Temp, ie_den, orb_en_den, Rad_den, VX, VY, VZ], cut)
        
        np.save(f'{abspath}/data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npy',\
                 [x_cut, y_cut, z_cut, dim_cut, den_cut, temp_cut, ie_den_cut, orb_en_den_cut, Rad_den_cut, VX_cut, VY_cut, VZ_cut])
        
    else:
        # you are not in alice
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        # choose what to plot
        npanels = '' # 3 or 6

        # load the data
        data = np.load(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npy', allow_pickle=True)
        x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid, VX_mid, VY_mid, VZ_mid =\
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]

        V_mid = np.sqrt(VX_mid**2 + VY_mid**2 + VZ_mid**2)
        R_mid = np.sqrt(x_mid**2 + y_mid**2 + z_mid**2)
        KE = 0.5 * V_mid**2
        PE = - prel.G * Mbh / R_mid
        ratio_E[idx] = np.mean(KE / np.abs(PE))

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
            orb_en_mid = np.abs(orb_en_den_mid * dim_mid**3) * prel.en_converter
            ie_mid = (ie_den_mid * dim_mid**3) * prel.en_converter
            Rad_mid = (Rad_den_mid * dim_mid**3) * prel.en_converter

            vminden = 2e-9 #np.percentile(den_mid, 5)
            vmaxden = 5e-6 #np.percentile(den_mid, 99)
            vminT = 1e4 #np.percentile(temp_mid, 5)
            vmaxT = 4e6 #np.percentile(temp_mid, 99)
            vminDim = 2e-2 ##np.percentile(dim_mid, 5)
            vmaxDim = 1.2 #np.percentile(dim_mid, 99)
            vminOrb = 5e39 #np.percentile(orb_en_mid, 5)
            vmaxOrb = 7e42 #np.percentile(orb_en_mid, 99)
            vminIE = 2e36 #np.percentile(ie_mid, 5)
            vmaxIE = 2e39 #np.percentile(ie_mid, 99)
            vminRad = 2e32 #np.percentile(Rad_mid, 5)
            vmaxRad = 2e40 #np.percentile(Rad_mid, 99)

            fig, ax = plt.subplots(2,3, figsize = (20,10))
            img = ax[0][0].scatter(x_mid/apo, y_mid/apo, c = den_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = vminden, vmax = vmaxden))
            cb = plt.colorbar(img)
            cb.set_label(r'Density [$M_\odot/R_\odot^3$]', fontsize = 14)

            img = ax[0][1].scatter(x_mid/apo, y_mid/apo, c = temp_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = vminT, vmax = vmaxT))
            cb = plt.colorbar(img)
            cb.set_label(r'T [K]', fontsize = 14)

            img = ax[0][2].scatter(x_mid/apo, y_mid/apo, c = dim_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = vminDim, vmax = vmaxDim))
            cb = plt.colorbar(img)
            cb.set_label(r'Cell size [$R_\odot$]', fontsize = 14)

            img = ax[1][0].scatter(x_mid/apo, y_mid/apo, c = orb_en_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = vminOrb, vmax = vmaxOrb))
            cb = plt.colorbar(img)
            cb.set_label(r'Absolute specific orbital energy [erg]', fontsize = 14)

            img = ax[1][1].scatter(x_mid/apo, y_mid/apo, c = ie_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = vminIE, vmax = vmaxIE))
            cb = plt.colorbar(img)
            cb.set_label(r'Specific IE [erg]', fontsize = 14)

            img = ax[1][2].scatter(x_mid/apo, y_mid/apo, c = Rad_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = vminRad, vmax = vmaxRad))
            cb = plt.colorbar(img)
            cb.set_label(r'Radiation energy density [erg]', fontsize = 14)

        if npanels == 6 or npanels == 3:    
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

if not do:
    print(len(ratio_E))
    np.save(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}_ratioE.npy', [snaps, tfb, ratio_E])

    
