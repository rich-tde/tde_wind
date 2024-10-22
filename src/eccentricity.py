from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'

import numpy as np
import matplotlib.pyplot as plt
import colorcet
import pickle
import matplotlib.colors as colors
from Utilities.operators import make_tree, single_branch
import Utilities.sections as sec
import src.orbits as orb
from Utilities.selectors_for_snap import select_snap


#
## CONSTANTS
#
G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

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
step = ''
check = 'LowRes' # '' or 'LowRes' or 'HiRes'
save = True

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}{step}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, step, time = True) #[100,115,164,199,216]

Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

def specific_j(r, vel):
    """ (Magnitude of) specific angular momentum """
    j = np.cross(r, vel)
    magnitude_j = np.linalg.norm(j, axis = 1)
    return magnitude_j

def eccentricity_squared(r, vel, specOE, Mbh, G):
    j = specific_j(r, vel)
    ecc2 = 1 + 2 * specOE * j**2 / (G * Mbh)**2
    return ecc2

if __name__ == '__main__':
    if alice: 
        col_ecc2 = []
        # col_Rsph = []
        radii = np.logspace(np.log10(R0), np.log10(apo),
                        num=200) 
        for i,snap in enumerate(snaps):
            print(snap)
            if alice:
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
            else:
                path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
            data = make_tree(path, snap, energy = True)
            R_vec = np.transpose(np.array([data.X, data.Y, data.Z]))
            vel_vec = np.transpose(np.array([data.VX, data.VY, data.VZ]))
            Rsph = np.linalg.norm(R_vec, axis = 1)
            vel = np.linalg.norm(vel_vec, axis=1)
            orb_en = orb.orbital_energy(Rsph, vel, data.Mass, G, c, Mbh)
            spec_orb_en = orb_en / data.Mass
            ecc2 = eccentricity_squared(R_vec, vel_vec, spec_orb_en, Mbh, G)

            # throw fluff and unbound material
            cut = np.logical_and(data.Den > 1e-19, orb_en < 0)
            Rsph_cut, mass_cut, ecc_cut = sec.make_slices([Rsph, data.Mass, ecc2], cut)
            ecc_cast = single_branch(radii,'radii', Rsph_cut, ecc_cut, weights = mass_cut)

            col_ecc2.append(ecc_cast)

        if save:
            np.save(f'{abspath}/data/{folder}/Ecc2_{check}{step}.npy', col_ecc2)
            with open(f'{abspath}/data/{folder}/Ecc_{check}{step}_days.txt', 'w') as file:
                file.write(f'# {folder}_{check}{step} \n' + ' '.join(map(str, snaps)) + '\n')
                file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
                file.close()
            np.save(f'{abspath}/data/{folder}/radiiEcc_{check}{step}.npy', radii)
    
    else:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        path = f'/Users/paolamartire/shocks/data/{folder}'
        # Low data
        ecc2 = np.load(f'{path}/Ecc2_.npy') 
        ecc = np.sqrt(ecc2)
        ecc_from_ecc2 = np.sqrt(ecc2)
        tfb_data = np.loadtxt(f'{path}/Ecc2__days.txt')
        snap_, tfb = tfb_data[0], tfb_data[1]
        radii = np.load(f'{path}/radiiEcc2_.npy')

        #%%
        img = plt.pcolormesh(radii/apo, tfb, ecc, vmin = 0.6, vmax = 1, cmap = 'cet_rainbow4')
        cb = plt.colorbar(img)
        plt.xscale('log')
        cb.set_label(r'Eccentricity', fontsize = 20, labelpad = 2)
        plt.xlabel(r'$R/R_{a}$', fontsize = 20)
        plt.ylabel(r'$t/t_{fb}$', fontsize = 20)
        plt.savefig(f'{abspath}Figs/{folder}/ecc.pdf')
        #%% HiRes data
        # eccHiRes = np.load(f'{path}/Ecc_HiRes.npy')
        ecc2HiRes = np.load(f'{path}/Ecc2_HiRes.npy')
        eccHiRes = np.sqrt(ecc2HiRes)
        tfb_dataHiRes = np.loadtxt(f'{path}/Ecc_HiRes_days.txt')
        snap_HiRes, tfb_HiRes = tfb_dataHiRes[0], tfb_dataHiRes[1]
        radiiHiRes = np.load(f'{path}/radiiEcc_HiRes.npy')

        n_HiRes = len(eccHiRes)
        snap_ = snap_[:n_HiRes]
        tfb = tfb[:n_HiRes]
        ecc = ecc[:n_HiRes]
        rel_diff = 2*np.abs(eccHiRes - ecc) / (eccHiRes+ecc)
        
        #%%
        fig, ax = plt.subplots(1,3, figsize = (25,6))
        # 
        img = ax[0].pcolormesh(radii/apo, tfb, ecc, vmin = 0.7, vmax = 1, cmap = 'cet_rainbow4')
        cb = fig.colorbar(img)
        ax[0].set_xscale('log')
        cb.set_label(r'Eccentricity', fontsize = 20, labelpad = 2)
        ax[0].text(0.4, np.max(tfb)-0.05,' res', fontsize = 20)

        img = ax[1].pcolormesh(radiiHiRes/apo, tfb_HiRes, eccHiRes, vmin = 0.7, vmax = 1, cmap = 'cet_rainbow4')
        cb = fig.colorbar(img)
        cb.set_label(r'Eccentricity', fontsize = 20, labelpad = 2)
        ax[1].set_xscale('log')
        ax[1].text(0.4, np.max(tfb)-0.05,'High res', fontsize = 20)

        img = ax[2].pcolormesh(radii/apo, tfb, rel_diff, cmap = 'plasma', norm = colors.LogNorm(vmin = 1e-4, vmax=1e-1))
        cb = fig.colorbar(img)
        ax[2].set_xscale('log')
        cb.set_label(r'Relative difference', fontsize = 20, labelpad = 2)

        for i in range(3):
            ax[i].set_xlabel(r'$R/R_{a}$', fontsize = 20)
        ax[0].set_ylabel(r'$t/t_{fb}$', fontsize = 20)

        plt.savefig(f'{abspath}Figs/multiple/ecc.pdf')
