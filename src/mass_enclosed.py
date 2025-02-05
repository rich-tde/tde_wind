""" Find the mass enclosed in a sphere of radius R0"""

import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
import src.orbits as orb

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

Rcheck = np.array([R0, Rt, a_mb, apo])
Mass_encl = np.zeros((len(snaps), len(Rcheck)))
Mass_encl_cut = np.zeros((len(snaps), len(Rcheck)))
for i,snap in enumerate(snaps):
    print(snap)
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    mass, den = data.Mass, data.Den
    Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    cut = den > 1e-19
    mass_cut, Rsph_cut = mass[cut], Rsph[cut]
    for j,R in enumerate(Rcheck):
        Mass_encl[i,j] = np.sum(mass[Rsph < R])
        Mass_encl_cut[i,j] = np.sum(mass_cut[Rsph_cut < R])

np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl.txt', Mass_encl)
np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl_cut.txt', Mass_encl_cut)
np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl_time.txt', tfb)