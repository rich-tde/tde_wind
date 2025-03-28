""" If alice: Compute and save the mass enclosed and the total diss rate in a sphere of radius R0, Rt, a_mb, apo for all the snapshots.
If local: plots"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import make_tree, calc_deriv
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
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

##
# MAIN
##
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rcheck = np.array([R0, Rt, a_mb, apo])
tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    Mass_unbound = np.zeros(len(snaps))
    for i,snap in enumerate(snaps):
        print(snap)
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        X, Y, Z, mass, den, vx, vy, vz = \
            data.X, data.Y, data.Z, data.Mass, data.Den, data.VX, data.VY, data.VZ
        cut = den > 1e-19
        X, Y, Z, mass, den, vx, vy, vz = \
            make_slices([X, Y, Z, mass, den, vx, vy, vz], cut)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        vel = np.sqrt(vx**2 + vy**2 + vz**2)
        orb_en_cut = orb.orbital_energy(Rsph, vel, mass, prel.G, prel.csol_cgs, Mbh)
        for i in range(len(Mass_unbound)):
            Mass_unbound[i] = np.sum(mass[orb_en_cut > 0])

    with open(f'{abspath}/data/{folder}/Mass_unbound{check}.txt','w') as file:
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')  
        file.write('# unbound mass [M_odot] \n' + ' '.join(map(str, Mass_unbound)) + '\n')  
        file.close()


    