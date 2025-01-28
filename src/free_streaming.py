""" Space-time colorplot of energies using tree. 
It has been written to be run on alice.
Cut in density at 1e-19 code units."""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    abspath_snap = '/home/martirep/data_pi-rossiem/TDE_data'
else:
    abspath = '/Users/paolamartire/shocks'
    abspath_snap = abspath

import numpy as np
from Utilities.selectors_for_snap import select_snap
from Utilities.operators import make_tree, single_branch
import Utilities.sections as sec
import src.orbits as orb
import Utilities.prelude as prel

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
checks = ['LowRes', '', 'HiRes']
weigh = 'Weighted' # '' or 'Weighted'

plot = True

Mbh = 10**m
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

# snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
# col_Radcut_den = []
# col_Rad_den = []

snap = 267

if not plot: 
    for check in checks:
        print(snap, check)
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        # print(snap, check)
        if alice:
            path = f'{abspath_snap}/{folder}/snap_{snap}'
        else:
            path = f'{abspath_snap}/TDE/{folder}/{snap}'
        
        data = make_tree(path, snap, energy = True)
        X, Y, Z, vol, den, Rad_den = \
            data.X, data.Y, data.Z,  data.Vol, data.Den, data.Rad
        
        Rad = Rad_den * vol
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        radii = np.logspace(np.log10(.8*apo),np.log10(np.max(Rsph)),
                        num = 200)  # simulator units

        # throw fluff
        cut = den > 1e-19 
        Rsph_cut, Rad_den_cut, Rad_cut =  Rsph[cut], Rad_den[cut], Rad[cut]
            
        # Cast down 
        if weigh == 'Weighted':
            Rad_den_cast = single_branch(radii, Rsph, Rad_den, weights = Rad)
            Rad_dencut_cast = single_branch(radii, Rsph_cut, Rad_den_cut, weights = Rad_cut)
        else: 
            Rad_den_cast = single_branch(radii, Rsph, Rad_den, weights = 1)
            Rad_dencut_cast = single_branch(radii, Rsph_cut, Rad_den_cut, weights = 1)

        # col_Radcut_den.append(Rad_dencut_cast)
        # col_Rad_den.append(Rad_den_cast)

        np.save(f'{abspath}/data/{folder}/fs{weigh}{snap}.npy', [radii, Rad_den_cast, Rad_dencut_cast])
# %%
if plot:
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    time_diff = False
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

    if time_diff: 
        snaps = [115, 164, 267]
        styles = ['-.', '--', 'solid']
        plt.figure(figsize = (10,5))
        for i,snap in enumerate(snaps):
            dataL = np.load(f'{abspath}/data/{folder}LowRes/fs{weigh}{snap}.npy')
            radiiL, Rad_den_castL, Rad_dencut_castL = dataL[0], dataL[1], dataL[2]
            data = np.load(f'{abspath}/data/{folder}/fs{weigh}{snap}.npy')
            radii, Rad_den_cast, Rad_dencut_cast = data[0], data[1], data[2]
            dataH = np.load(f'{abspath}/data/{folder}HiRes/fs{weigh}{snap}.npy')
            radiiH, Rad_den_castH, Rad_dencut_castH = dataH[0], dataH[1], dataH[2]

            # Luminosity free streaming
            LumL = 4 * np.pi * (radiiL*prel.Rsol_cgs)**2 * prel.c_cgs * (Rad_den_castL * prel.en_den_converter)
            Lum = 4 * np.pi * (radii*prel.Rsol_cgs)**2 * prel.c_cgs * (Rad_den_cast * prel.en_den_converter)
            LumH = 4 * np.pi * (radiiH*prel.Rsol_cgs)**2 * prel.c_cgs * (Rad_den_castH * prel.en_den_converter)
            fL = interp1d(radiiL, LumL, fill_value="extrapolate")#, kind = 'cubic')
            LumL_int = fL(radii)
            fH = interp1d(radiiH, LumH, fill_value="extrapolate")#, kind = 'cubic')
            LumH_int = fH(radii)
            diffL = 2*np.abs(Lum - LumL_int)/(Lum + LumL_int)
            diffH = 2*np.abs(Lum - LumH_int)/(Lum + LumH_int)

            plt.plot(radii, diffL, c = 'darkorange', linestyle = styles[i])
            plt.plot(radii, diffH, label = f'Snap {snap}', c = 'darkviolet', linestyle = styles[i])
            plt.ylabel('Relative difference in luminosity')
            plt.xlabel(r'$R [R_\odot]$')
            plt.grid()
            plt.legend()
            plt.loglog()
            plt.tick_params(axis='both', which='minor', size=4)
            plt.tick_params(axis='both', which='major', size=6)
            plt.savefig(f'{abspath}/Figs/multiple/convRad/Ldifftime{weigh}.png')
    else:
        dataL = np.load(f'{abspath}/data/{folder}LowRes/fs{weigh}{snap}.npy')
        radiiL, Rad_den_castL, Rad_dencut_castL = dataL[0], dataL[1], dataL[2]
        data = np.load(f'{abspath}/data/{folder}/fs{weigh}{snap}.npy')
        radii, Rad_den_cast, Rad_dencut_cast = data[0], data[1], data[2]
        dataH = np.load(f'{abspath}/data/{folder}HiRes/fs{weigh}{snap}.npy')
        radiiH, Rad_den_castH, Rad_dencut_castH = dataH[0], dataH[1], dataH[2]

        img, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
        ax1.plot(radiiL/apo, Rad_dencut_castL * prel.en_den_converter, label = 'LowRes', c = 'darkorange')
        ax1.plot(radii/apo, Rad_dencut_cast * prel.en_den_converter, label = 'Fid', c = 'yellowgreen')
        ax1.plot(radiiH/apo, Rad_dencut_castH * prel.en_den_converter, label = 'HiRes', c = 'darkviolet')
        ax1.set_title(f'Snap {snap}, cut in density, {weigh}')
        ax1.set_ylabel(r'Rad energy density [erg/cm$^3$]')

        ax2.plot(radiiL/apo, Rad_den_castL * prel.en_den_converter, label = 'LowRes', c = 'darkorange')
        ax2.plot(radii/apo, Rad_den_cast * prel.en_den_converter, label = 'Fid', c = 'yellowgreen')
        ax2.plot(radiiH/apo, Rad_den_castH * prel.en_den_converter, label = 'HiRes', c = 'darkviolet')
        ax2.set_title(f'Snap {snap}, NO density cut, {weigh}')
        ax2.legend()    

        for ax in [ax1, ax2]:
            ax.set_xlabel(r'$R [R_{\rm a}]$')
            ax.set_yscale('log')
            ax.grid()
            ax.tick_params(axis='both', which='minor', size=4)
            ax.tick_params(axis='both', which='major', size=6)
        plt.tight_layout()
        plt.savefig(f'{abspath}/Figs/multiple/convRad/Erad{weigh}{snap}.png')

        # Luminosity free streaming
        LumL = 4 * np.pi * (radiiL*prel.Rsol_cgs)**2 * prel.c_cgs * (Rad_den_castL * prel.en_den_converter)
        Lum = 4 * np.pi * (radii*prel.Rsol_cgs)**2 * prel.c_cgs * (Rad_den_cast * prel.en_den_converter)
        LumH = 4 * np.pi * (radiiH*prel.Rsol_cgs)**2 * prel.c_cgs * (Rad_den_castH * prel.en_den_converter)

        img, (ax1, ax2) = plt.subplots(2,1, figsize = (10,10))
        ax1.plot(radiiL/apo, Rad_den_castL * prel.en_den_converter, label = 'LowRes', c = 'darkorange')
        ax1.plot(radii/apo, Rad_den_cast * prel.en_den_converter, label = 'Fid', c = 'yellowgreen')
        ax1.plot(radiiH/apo, Rad_den_castH * prel.en_den_converter, label = 'HiRes', c = 'darkviolet')
        ax1.set_ylabel(r'Rad energy density [erg/cm$^3$]')
        ax1.set_ylim(1, 100)

        ax2.plot(radiiL/apo, LumL, label = 'LowRes', c = 'darkorange')
        ax2.plot(radii/apo, Lum, label = 'Fid', c = 'yellowgreen')
        ax2.plot(radiiH/apo, LumH, label = 'HiRes', c = 'darkviolet')
        ax2.set_xlabel(r'$R [R_{\rm a}]$')
        ax2.set_ylabel(r'Luminosity [erg/s]')
        ax2.set_ylim(1e41, 1e43)
        ax2.legend()   
        
        for ax in [ax1, ax2]:
            ax.loglog()
            ax.grid()
            ax.tick_params(axis='both', which='minor', size=4)
            ax.tick_params(axis='both', which='major', size=6)
            ax.set_xlim(10, np.max(radiiH)/apo)
        plt.suptitle(f'Snap {snap}, NO density cut, {weigh}')
        plt.tight_layout()
        plt.savefig(f'{abspath}/Figs/multiple/convRad/L{weigh}{snap}.png')

        # interpolate LumL in the same radii as Lum
        fL = interp1d(radiiL, LumL, fill_value="extrapolate")#, kind = 'cubic')
        LumL_int = fL(radii)
        fH = interp1d(radiiH, LumH, fill_value="extrapolate")#, kind = 'cubic')
        LumH_int = fH(radii)
        diffL = 2*np.abs(Lum - LumL_int)/(Lum + LumL_int)
        diffH = 2*np.abs(Lum - LumH_int)/(Lum + LumH_int)

        img, (ax1, ax2) = plt.subplots(2, 1, figsize = (10,10))
        ax1.plot(radiiL, LumL, label = 'LowRes', c = 'darkorange')
        ax1.plot(radii, Lum, label = 'Fid', c = 'yellowgreen')
        ax1.plot(radiiH, LumH, label = 'HiRes', c = 'darkviolet')
        ax1.set_ylabel(r'Luminosity [erg/s]')
        ax1.legend()

        ax2.plot(radii, diffL, label = 'LowRes', c = 'darkorange')
        ax2.plot(radii, diffH, label = 'HiRes', c = 'darkviolet')
        ax2.set_ylabel('Relative difference in luminosity')
        ax2.axvline(radiiH[len(radiiH)-1], c = 'darkviolet', linestyle = '--')
        ax2.set_xlabel(r'$R [R_\odot]$')

        for ax in [ax1, ax2]:
            ax.loglog()
            ax.grid()
            ax.tick_params(axis='both', which='minor', size=4)
            ax.tick_params(axis='both', which='major', size=6)
        plt.suptitle(f'Snap {snap}, NO density cut, {weigh}', fontsize = 15)
        plt.tight_layout()
        plt.savefig(f'{abspath}/Figs/multiple/convRad/Ldiff{weigh}{snap}.png')



    # %%
