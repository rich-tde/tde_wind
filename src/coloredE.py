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
from Utilities.operators import make_tree, single_branch, multiple_branch, to_cylindric
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
who = 'all' #'' or 'all'

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
if who == '':
    col_ie = []
    col_orb_en = []
    col_Rad = []
    col_Rad_den = []
else:
    col_ie = np.zeros(len(snaps))
    col_orb_en = np.zeros(len(snaps))
    col_Rad = np.zeros(len(snaps))
    col_Rad_den = np.zeros(len(snaps))
for i,snap in enumerate(snaps):
    print(snap)
    if alice:
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
    
    data = make_tree(path, snap, energy = True)
    X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den = \
        data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Rad
    box = np.load(f'{path}/box_{snap}.npy')
    if int(snap) <= 317:
        boxL = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/snap_{snap}/box_{snap}.npy')
        if int(snap) <= 267:
            boxH = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/snap_{snap}/box_{snap}.npy')
        else:
            boxH = box
    else: 
        boxH = box
        boxL = box

    xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0], -240]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
    xmax, ymax, zmax = np.min([box[3], boxL[3], boxH[3]]), np.min([box[4], boxL[4], boxH[4]]), np.min([box[5], boxL[5], boxH[5]])
    cutx = (X > xmin) & (X < xmax)
    cuty = (Y > ymin) & (Y < ymax)
    cutz = (Z > zmin) & (Z < zmax)
    cut_coord = cutx & cuty & cutz
    X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den = \
        sec.make_slices([X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den], cut_coord)
    Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
    vel = np.sqrt(np.power(VX, 2) + np.power(VY, 2) + np.power(VZ, 2))
    orb_en = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh)
    Rad = Rad_den * vol
    ie = ie_den * vol
    dim_cell = (3/(4*np.pi) * vol)**(1/3)
    ie_onmass = ie_den / den
    orb_en_onmass = orb_en / mass

    # throw fluff
    cut = den > 1e-19 
    Rsph_cut, mass_cut, den_cut, ie_cut, ie_onmass_cut, orb_en_cut, orb_en_onmass_cut, Rad_cut, Rad_den_cut, vol_cut = \
            sec.make_slices([Rsph, mass, den, ie, ie_onmass, orb_en, orb_en_onmass, Rad, Rad_den, vol], cut)
    Rad_onmass_cut = Rad_cut / mass_cut
    tocast_cut = Rsph_cut
        
    # Cast down 
    # ie_cast = single_branch(radii, tocast_cut, ie_onmass_cut, weights = mass_cut)
    # orb_en_cast = single_branch(radii, tocast_cut, orb_en_onmass_cut, weights = mass_cut)
    # Rad_den_cast = single_branch(radii, tocast_cut, Rad_den_cut, weights = Rad_cut)
    if who == '':
        tocast_matrix = [ie_onmass_cut, orb_en_onmass_cut, Rad_onmass_cut, Rad_den_cut]
        weights_matrix = [mass_cut, mass_cut, mass_cut, Rad_cut]
        casted = multiple_branch(radii, tocast_cut, tocast_matrix, weights_matrix)
        ie_cast, orb_en_cast, Rad_cast, Rad_den_cast = casted[0], casted[1], casted[2], casted[3]

        col_ie.append(ie_cast)
        col_orb_en.append(orb_en_cast)
        col_Rad.append(Rad_cast)
        col_Rad_den.append(Rad_den_cast)
    else:
        col_ie[i] = np.sum(ie_cut)
        col_orb_en[i] = np.sum(orb_en_cut)
        # NO cut for radiation
        col_Rad[i] = np.sum(Rad)
#%%
if save:
    if who == '':
        np.save(f'{abspath}/data/{folder}/coloredE{who}_{check}.npy', [col_ie, col_orb_en, col_Rad, col_Rad_den])
    else:
        np.save(f'{abspath}/data/{folder}/coloredE{who}_{check}_thresh.npy', [col_ie, col_orb_en, col_Rad])
    with open(f'{abspath}/data/{folder}/coloredE{who}_{check}_days.txt', 'w') as file:
        file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()
    np.save(f'{abspath}/data/{folder}/coloredE{who}_{check}_radii.npy', radii)