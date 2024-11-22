""" Compute Rdissipation. Written to be run on alice."""
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
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap

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
check = 'DoubleRad'
Rt = Rstar * (Mbh/mstar)**(1/3)

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

Rdiss = np.zeros(len(snaps))
Edisstot = np.zeros(len(snaps))
Eradtot = np.zeros(len(snaps))
for i,snap in enumerate(snaps):
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    cut = data.Den > 1e-19
    Rsph, vol, Ediss_den, Rad_den = Rsph[cut], data.Vol[cut], data.Diss[cut], data.Rad[cut]
    # Ediss = np.abs(Ediss_den) * vol
    # Edisstot[i] = np.sum(Ediss)
    Erad = Rad_den * vol
    Eradtot[i] = np.sum(Erad)
    # Rdiss[i] = np.sum(Rsph * vol * Ediss) / np.sum(vol * Ediss)

with open(f'{abspath}/data/{folder}/Rdiss_{check}.txt','a') as file:
    # file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
    # file.write(f'# Rdiss \n' + ' '.join(map(str, Rdiss)) + '\n')
    # file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
    # file.write(f'# Edotdiss \n' + ' '.join(map(str, Edisstot)) + '\n')
    file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
    file.write(f'# Erad \n' + ' '.join(map(str, Eradtot)) + '\n')
    file.close()