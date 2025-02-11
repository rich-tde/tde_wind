""" Space-time colorplot of energies using tree. 
It has been written to be run on alice.
Cut in density at 1e-19 code units."""
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
from Utilities.operators import make_tree, single_branch, multiple_branch
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
check = 'LowRes' 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
save = True

Mbh = 10**m
Rs = 2*prel.G*Mbh / prel.csol_cgs**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)


radii = np.logspace(np.log10(R0), np.log10(1.5*apo),
                    num=200)  # simulator units

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]

col_ie = []
col_orb_en = []
col_Radcut = []
col_Radcut_den = []
col_Rad_den = []


for i,snap in enumerate(snaps):
    print(snap)
    if alice:
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
    
    data = make_tree(path, snap, energy = True)
    X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den = \
        data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Rad
    Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
    cut = den > 1e-19 #NOT for radiation
    Rad = Rad_den * vol
    Rsph_cut, VX_cut, VY_cut, VZ_cut, mass_cut, vol_cut, den_cut, ie_den_cut = \
        sec.make_slices([Rsph, VX, VY, VZ, mass, vol, den, ie_den], cut)
    vel_cut = np.sqrt(np.power(VX_cut, 2) + np.power(VY_cut, 2) + np.power(VZ_cut, 2))
    orb_en_cut = orb.orbital_energy(Rsph_cut, vel_cut, mass_cut, prel.G, prel.csol_cgs, Mbh)
    ie_cut = ie_den_cut * vol_cut
        
    ie_onmass_cut = ie_den_cut / den_cut
    orb_en_onmass_cut = orb_en_cut / mass_cut
    Rad_onmass = Rad / mass
    ie_cast, orb_en_cast = multiple_branch(radii, Rsph_cut, [ie_onmass_cut, orb_en_onmass_cut], [mass_cut, mass_cut])
    casted = multiple_branch(radii, Rsph, [Rad_onmass, Rad_den], [mass, Rad])
    Rad_cast, Rad_den_cast = casted[0], casted[1]

    col_ie.append(ie_cast)
    col_orb_en.append(orb_en_cast)
    col_Radcut.append(Rad_cast)
    col_Rad_den.append(Rad_den_cast)

#%%
if save:
    np.save(f'{abspath}/data/{folder}/coloredE_{check}_radii.npy', radii)
    np.save(f'{abspath}/data/{folder}/coloredE_{check}.npy', [col_ie, col_orb_en, col_Radcut, col_Radcut_den, col_Rad_den])
    with open(f'{abspath}/data/{folder}/coloredE_{check}_days.txt', 'w') as file:
            file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()