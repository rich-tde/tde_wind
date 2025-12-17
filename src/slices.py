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
check = 'HiResNewAMR' # 'LowRes' or 'HiRes'

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
apo = things['apo']
coord_to_cut = 'z' # 'x', 'y', 'z'
cut_chosen = 0

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
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

for idx, snap in enumerate(snaps):
    if snap != 0:
            continue
    print(snap, flush=True)
    if do:
        # you are in alice
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'

        data = make_tree(path, snap, energy = True)
        X, Y, Z, vol, den, mass, Temp, ie_den, Rad_den, VX, VY, VZ, Diss_den, IE_den, Press, DivV = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Temp, data.IE, data.Rad, data.VX, data.VY, data.VZ, data.Diss, data.IE, data.Press, data.DivV
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        dim_cell = vol**(1/3)
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
        x_cut, y_cut, z_cut, dim_cut, mass_cut, den_cut, temp_cut, ie_den_cut, Rad_den_cut, VX_cut, VY_cut, VZ_cut, Diss_den_cut, IE_den_cut, Press_cut, DivV_cut = \
            sec.make_slices([X, Y, Z, dim_cell, den, mass, Temp, ie_den, Rad_den, VX, VY, VZ, Diss_den, IE_den, Press, DivV], cut)

        slice_data = {
            'x': x_cut,
            'y': y_cut,
            'z': z_cut,
            'dim': dim_cut,
            'mass': mass_cut,
            'den': den_cut,
            'temp': temp_cut,
            'ie_den': ie_den_cut,
            'Rad_den': Rad_den_cut,
            'VX': VX_cut,
            'VY': VY_cut,
            'VZ': VZ_cut,
            'Diss_den': Diss_den_cut,
            'IE_den': IE_den_cut,
            'Press': Press_cut,
            'DivV': DivV_cut
        }
        np.savez(f'{abspath}/data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npz', **slice_data)
        
    else:
        # you are not in alice
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        # choose what to plot
        npanels = 2 # 3 or 6

        # load the data
        data = np.load(f'{abspath}data/{folder}/slices/{coord_to_cut}/{coord_to_cut}{cut_name}slice_{snap}.npz', allow_pickle=True)
        x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, Rad_den_mid, VX_mid, VY_mid, VZ_mid, Diss_den_mid, IE_den_mid, Press_mid =\
            data['x'], data['y'], data['z'], data['dim'], data['den'], data['temp'], data['ie_den'], data['Rad_den'], data['VX'], data['VY'], data['VZ'], data['Diss_den'], data['IE_den'], data['Press']
        Diss_mid = Diss_den_mid * dim_mid**3 * prel.en_converter
        Temp_rad_mid = (Rad_den_mid * prel.en_den_converter/prel.alpha_cgs)**(0.25)
        vel_mid = np.sqrt(VX_mid**2 + VY_mid**2 + VZ_mid**2)
        cs_mid = Press_mid/den_mid

        if npanels == 2:
            fig, ax = plt.subplots(1,2, figsize = (14,8))
            img = ax[0].scatter(x_mid, y_mid, c = den_mid*prel.den_converter, cmap = 'plasma', s= .1, \
                        norm = colors.LogNorm(vmin = 5e-9, vmax = 1e-4))
            cb = plt.colorbar(img, orientation='horizontal')
            cb.set_label(r'Density [g/cm$^3$]')
            cb.ax.tick_params(which='major', length=7, width=1.2)
            cb.ax.tick_params(which='minor', length=4, width=1)
            ax[0].set_ylabel(r'$Y/R_\odot$')
 
            img = ax[1].scatter(x_mid, y_mid, c = Diss_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = 1e38, vmax = 8e40))
            cb = plt.colorbar(img, orientation='horizontal')
            cb.set_label(r'Dissipation [erg/s]')
            cb.ax.tick_params(which='major', length=7, width=1.2)
            cb.ax.tick_params(which='minor', length=4, width=1)

        
        if npanels == 3:
            fig, ax = plt.subplots(1,3, figsize = (20,5))
            img = ax[0].scatter(x_mid, y_mid, c = np.abs(DivV), cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = 1e-1, vmax = 1e2))
            cb = plt.colorbar(img)
            cb.set_label(r'$|\nabla \cdot \mathbf{v}|$', fontsize = 16)
            ax[0].set_ylabel(r'$Y/R_{\rm a}$')

            img = ax[1].scatter(x_mid, y_mid, c = cs_mid/vel_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = 1e-1, vmax = 10))
            cb = plt.colorbar(img)
            cb.set_label(r'$c_{\rm s}/v$ [erg/g]')

            img = ax[2].scatter(x_mid, y_mid, c = Temp_rad_mid, cmap = 'viridis', s= .1, \
                        norm = colors.LogNorm(vmin = 1e4, vmax = 1e6))
            cb = plt.colorbar(img)
            cb.set_label(r'$T_{\rm rad}$ [K]')
        
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

        plt.suptitle(f't = {np.round(tfb[idx], 4)}' + r'$t_{\rm fb}$', fontsize = 20)
        plt.tight_layout()
        if npanels == 6:
            for i in range(2):
                ax[i][0].set_ylabel(r'$Y/R_{\rm a}$', fontsize = 20)
                for j in range(3):
                    ax[i][j].set_xlabel(r'$X/R_{\rm a}$', fontsize = 20)
                    ax[i][j].set_xlim(-1.2, 0.1)#(-340,25)
                    ax[i][j].set_ylim(-0.5, 0.5)#(-70,70)
        if npanels == 2 or npanels == 3:
            ax[0].set_ylabel(r'$ [R_\odot]$')
            for i in range(npanels):
                ax[i].set_xlabel(r'$X [R_\odot]$')
                ax[i].set_xlim(-1, 1)#(-340,25)
                ax[i].set_ylim(-1, 1)#(-70,70)
                ax[i].set_aspect('equal', 'box')
        # ax[0][0].text(-1.05, 0.38, f'snap {int(snaps[idx])}', fontsize = 16)

        plt.savefig(f'{abspath}Figs/{folder}/slices/Panel{npanels}Slice{snap}.png')
        # plt.close()
