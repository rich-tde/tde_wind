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
from Utilities.operators import make_tree, single_branch, to_cylindric
import Utilities.sections as sec
import src.orbits as orb

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
check = '' 
xaxis = 'radii'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
save = True

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)


radii = np.logspace(np.log10(R0), np.log10(1.5*apo),
                    num=200)  # simulator units

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
col_ie = []
col_orb_en = []
col_Rad = []
for i,snap in enumerate(snaps):
    print(snap)
    if alice:
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
    data = make_tree(path, snap, energy = True)
    Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
    vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
    mass, vol, den, ie_den, Rad_den = data.Mass, data.Vol, data.Den, data.IE, data.Rad
    theta, _ = to_cylindric(data.X, data.Y)
    orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
    Rad = Rad_den * vol
    ie = ie_den * vol
    dim_cell = (3/(4*np.pi) * vol)**(1/3)
    ie_onmass = ie_den / data.Den
    orb_en_onmass = orb_en / mass

    # throw fluff
    cut = den > 1e-19 
    Rsph_cut, theta_cut, mass_cut, den_cut, ie_cut, ie_onmass_cut, orb_en_cut, orb_en_onmass_cut, Rad_cut, Rad_den_cut, vol_cut = \
            sec.make_slices([Rsph, theta, mass, den, ie, ie_onmass, orb_en, orb_en_onmass, Rad, Rad_den, vol], cut)

    tocast_cut = Rsph_cut
        
    # Cast down 
    ie_cast = single_branch(radii, xaxis, tocast_cut, ie_onmass_cut, weights = mass_cut)
    orb_en_cast = single_branch(radii, xaxis, tocast_cut, orb_en_onmass_cut, weights = mass_cut)
    Rad_cast = single_branch(radii, xaxis, tocast_cut, Rad_den_cut, weights = Rad_cut)

    col_ie.append(ie_cast)
    col_orb_en.append(orb_en_cast)
    col_Rad.append(Rad_cast)

#%%
if save:
    np.save(f'{abspath}/data/{folder}/coloredE_{check}_{xaxis}.npy', [col_ie, col_orb_en, col_Rad])
    with open(f'{abspath}/data/{folder}/coloredE_{check}_days.txt', 'w') as file:
        file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()
    np.save(f'{abspath}/data/{folder}/{xaxis}En_{check}.npy', radii)