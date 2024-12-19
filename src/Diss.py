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
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = ''
Rt = Rstar * (Mbh/mstar)**(1/3)

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

Rdiss = np.zeros(len(snaps))
Eradtot = np.zeros(len(snaps))
for i,snap in enumerate(snaps):
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
    cut = data.Den > 1e-19
    if check == '':
        Ediss_den = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/dissdata{m}/Diss_{snap}.npy')
    else:
        Ediss_den = data.Diss
    Rsph, vol, Rad_den, Ediss_den = Rsph[cut], data.Vol[cut], data.Rad[cut], Ediss_den[cut]
    Ediss_tot = np.sum(np.abs(Ediss_den) * vol) 
    # average weighted by energy dissipation rate [energy/time]
    Rdiss[i] = np.sum(Rsph * vol * np.abs(Ediss_den)) / Ediss_tot
    Eradtot[i] = np.sum(Rad_den * vol)

with open(f'{abspath}/data/{folder}/Rdiss_{check}.txt','a') as file:
    file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
    file.write(f'# Rdiss \n' + ' '.join(map(str, Rdiss)) + '\n')
    file.write(f'# Total radiation energy [energy in code units] \n' + ' '.join(map(str, Eradtot)) + '\n')
    file.close()