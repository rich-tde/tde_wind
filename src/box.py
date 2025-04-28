""" See how different is the box sizes in the different runs"""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'

import numpy as np
from Utilities.selectors_for_snap import select_snap
import Utilities.prelude as prel

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
Mbh = 10**m

for check in ['LowRes', '', 'HighRes']:
    folder = f'{commonfolder}{check}'
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    xmin = np.zeros(len(snaps))
    xmax = np.zeros(len(snaps))
    ymin = np.zeros(len(snaps))
    ymax = np.zeros(len(snaps))
    zmin = np.zeros(len(snaps))
    zmax = np.zeros(len(snaps))
    for i, snap in enumerate(snaps):
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        box = np.load(f'{path}/box_{snap}.npy')
        xmin[i] = box[0]
        ymin[i] = box[1]
        zmin[i] = box[2]
        xmax[i] = box[3]
        ymax[i] = box[4]
        zmax[i] = box[5]
    np.save(f'{abspath}/data/{folder}/boxlim_{check}.npy', np.array([xmin, ymin, zmin, xmax, ymax, zmax]))
    
