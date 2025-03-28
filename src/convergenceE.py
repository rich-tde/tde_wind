""" Total energies (orbital, internal and radiation) at each snapshot, both with cut in coordinates and not.
Cut in density (at 1e-19 code units) is done in both the cases, but not for radiation.
Written to be run on alice"""
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
from Utilities.operators import make_tree
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
Mbh = 10**m
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]

col_ie = np.zeros(len(snaps))
col_orb_en = np.zeros(len(snaps))
col_Rad = np.zeros(len(snaps))
col_ie_thres = np.zeros(len(snaps))
col_orb_en_thres = np.zeros(len(snaps))
col_Rad_thres = np.zeros(len(snaps))

for i,snap in enumerate(snaps):
    print(snap, flush=False)
    sys.stdout.flush()

    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den = \
        data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Rad
    Rad = Rad_den * vol
    # cut in density NOT in radiation
    cut = den > 1e-19 
    X_cut, Y_cut, Z_cut, VX_cut, VY_cut, VZ_cut, mass_cut, vol_cut, den_cut, ie_den_cut = \
        sec.make_slices([X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den], cut)
    Rsph_cut = np.sqrt(np.power(X_cut, 2) + np.power(Y_cut, 2) + np.power(Z_cut, 2))
    vel_cut = np.sqrt(np.power(VX_cut, 2) + np.power(VY_cut, 2) + np.power(VZ_cut, 2))
    orb_en_cut = orb.orbital_energy(Rsph_cut, vel_cut, mass_cut, prel.G, prel.csol_cgs, Mbh)
    ie_cut = ie_den_cut * vol_cut

    # total energies with only the cut in density (not in radiation)
    col_ie[i] = np.sum(ie_cut)
    col_orb_en[i] = np.sum(orb_en_cut)
    col_Rad[i] = np.sum(Rad)

    # consider the small box for the cut in coordinates
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
    xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0], -0.75*apo]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
    xmax, ymax, zmax = np.min([box[3], boxL[3], boxH[3]]), np.min([box[4], boxL[4], boxH[4]]), np.min([box[5], boxL[5], boxH[5]])
    cut_coord = (X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax) & (Z > zmin) & (Z < zmax) 
    Rad_thresh = Rad[cut_coord]
    cut_den_coord = (X_cut > xmin) & (X_cut < xmax) & (Y_cut > ymin) & (Y_cut < ymax) & (Z_cut > zmin) & (Z_cut < zmax)
    den_thresh, orb_en_thresh, ie_thresh = \
        sec.make_slices([den_cut, orb_en_cut, ie_cut], cut_den_coord)

    # total energies with the cut in density (not in radiation) and coordinates
    col_ie_thres[i] = np.sum(ie_thresh)
    col_orb_en_thres[i] = np.sum(orb_en_thresh)
    col_Rad_thres[i] = np.sum(Rad_thresh)

np.save(f'{abspath}/data/{folder}/convE_{check}.npy', [col_ie, col_orb_en, col_Rad])
np.save(f'{abspath}/data/{folder}/convE_{check}_thresh.npy', [col_ie_thres, col_orb_en_thres, col_Rad_thres])
with open(f'{abspath}/data/{folder}/convE_{check}_days.txt', 'w') as file:
        file.write(f'# In convE_{check}_thresh you find internal, orbital and radiation energy [NO denisty/specific] inside the biggest box enclosed in the three simulation volumes and cut in density.\n')
        file.write(f'# In convE_{check} only cut in density.')
        file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()