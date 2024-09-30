abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import matplotlib.pyplot as plt
from Utilities.operators import make_tree
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

check = 'Low'
save = True
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) #[100,115,164,199,216]

Mbh = 10**m
# Rs = 2*G*Mbh / c**2
# Rt = Rstar * (Mbh/mstar)**(1/3)
# Rp =  Rt / beta
# R0 = 0.6 * Rt
# apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

def specific_j(r, vel):
    """ (Magnitude of) specific angular momentum """
    j = np.cross(r, vel)
    magnitude_j = np.linalg.norm(j)
    return magnitude_j

def eccentricity(r, vel, mstar, G, c, Mbh):
    OE = orb.orbital_energy(r, vel, mstar, G, c, Mbh)
    specific_OE = OE / mstar
    j = specific_j(r, vel)
    ecc = np.sqrt(1 + 2 * specific_OE * j**2 / (G * mstar**2))
    return ecc

if __name__ == '__main__':
    for i,snap in enumerate(snaps):
        print(snap)
        if alice:
            if check == 'Low':
                check = ''
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
        data = make_tree(path, snap, energy = True)