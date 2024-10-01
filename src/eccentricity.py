abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import matplotlib.pyplot as plt
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
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{step}'

check = 'Low' # 'Low' or 'HiRes'
save = True
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, step, time = True) #[100,115,164,199,216]

Mbh = 10**m
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

def eccentricity(r, vel, OE, mstar, G, c, Mbh):
    specific_OE = OE / mstar
    j = specific_j(r, vel)
    ecc = np.sqrt(1 + 2 * np.abs(specific_OE) * j**2 / (G * mstar**2))
    return ecc

if __name__ == '__main__':
    if alice: 
        col_ecc = []
        radii = np.logspace(np.log10(R0), np.log10(1.5*apo),
                        num=200) 
        for i,snap in enumerate(snaps):
            print(snap)
            if alice:
                if check == 'Low':
                    check = ''
                path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}{step}/snap_{snap}'
            else:
                path = f'/Users/paolamartire/shocks/TDE/{folder}{check}{step}/{snap}'
            data = make_tree(path, snap, energy = True)
            R_vec = np.transpose(np.array([data.X, data.Y, data.Z]))
            vel_vec = np.transpose(np.array([data.VX, data.VY, data.VZ]))
            Rsph = np.linalg.norm(R_vec, axis = 1)
            vel = np.linalg.norm(vel_vec, axis=1)
            orb_en = orb.orbital_energy(Rsph, vel, data.Mass, G, c, Mbh)
            ecc = eccentricity(R_vec, vel_vec, orb_en, mstar, G, c, Mbh)

            # throw fluff (cut from Konstantinos) and unbound material
            cut = np.logical_and(data.Den > 1e-12, orb_en < 0)
            Rsph_cut, mass_cut, ecc_cut = sec.make_slices([Rsph, data.Mass, ecc], cut)
            ecc_cast = single_branch(radii,'radii', Rsph_cut, ecc_cut, weights = mass_cut)

            col_ecc.append(ecc_cast)

        if save:
            if alice:
                if check == '':
                    check = 'Low'
                prepath = f'/data1/martirep/shocks/shock_capturing'
            else: 
                prepath = f'/Users/paolamartire/shocks'
            np.save(f'{prepath}/data/{folder}/Ecc_{check}{step}.npy', [col_ecc])
            with open(f'{prepath}/data/{folder}/Ecc_{check}{step}_days.txt', 'w') as file:
                file.write(f'# {folder}_{check}{step} \n' + ' '.join(map(str, snaps)) + '\n')
                file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
                file.close()
            np.save(f'{prepath}/data/{folder}/Ecc_{check}{step}.npy', radii)
    
    else:
        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
        path = f'/Users/paolamartire/shocks/data/{folder}/ecc'
        # Low data
        eccLow = np.load(f'{path}/Ecc_Low.npy') 
        tfb_dataLow = np.loadtxt(f'{path}/Ecc_Low_days.txt')
        snap_Low, tfb_Low = tfb_dataLow[0], tfb_dataLow[1]
        radiiLow = np.load(f'{path}/radiiEcc_Low.npy')
        # HiRes data
        eccHiRes = np.load(f'{path}/Ecc_HiRes.npy')
        tfb_dataHiRes = np.loadtxt(f'{path}/Ecc_HiRes_days.txt')
        snap_HiRes, tfb_HiRes = tfb_dataHiRes[0], tfb_dataHiRes[1]
        radiiHiRes = np.load(f'{path}/radiiEcc_HiRes.npy')

        n_HiRes = len(eccHiRes)
        snap_Low = snap_Low[:n_HiRes]
        tfb_Low = tfb_Low[:n_HiRes]
        eccLow = eccLow[:n_HiRes]

        fig, ax = plt.subplots(1,3, figsize = (14,8))
        # Low
        img = ax[0].pcolormesh(radiiLow/apo, tfb_Low, eccLow, vmin = 0, vmax = 1, cmap = 'viridis')
        cb = fig.colorbar(img)
        cb.set_label(r'Eccentricity', fontsize = 20, labelpad = 2)
        ax[0].text(np.min(radiiLow/apo), 0.1,'Low res', fontsize = 20)

        img = ax[1].pcolormesh(radiiHiRes/apo, tfb_HiRes, eccHiRes, vmin = 0, vmax = 1, cmap = 'viridis')
        cb = fig.colorbar(img)
        cb.set_label(r'Eccentricity', fontsize = 20, labelpad = 2)
        ax[1].text(np.min(radiiHiRes/apo), 0.1,'Low res', fontsize = 20)

        # img = ax[2].pcolormesh(radiiLow/apo, tfb_Low, col_Rad, norm=norm_Radsix, cmap = cmap)
        # cb = fig.colorbar(img)
        # ax[2].set_title('Radiation energy density', fontsize = 20)
        # cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

