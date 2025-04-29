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

if alice:
    for check in ['LowRes', '', 'HiRes']:
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
        
else:
    import matplotlib.pyplot as plt
    _, tfbL = np.loadtxt(f'{abspath}/data/{commonfolder}LowRes/convE_LowRes_days.txt')
    _, tfb = np.loadtxt(f'{abspath}/data/{commonfolder}/convE__days.txt')
    _, tfbH = np.loadtxt(f'{abspath}/data/{commonfolder}HiRes/convE_HiRes_days.txt')
    dataL = np.load(f'{abspath}/data/{commonfolder}LowRes/boxlim_LowRes.npy')
    data = np.load(f'{abspath}/data/{commonfolder}/boxlim_.npy')
    dataH = np.load(f'{abspath}/data/{commonfolder}HiRes/boxlim_HiRes.npy')
    label_coord = ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']
    for i in range(len(label_coord)):
        plt.figure(figsize=(10, 5))
        plt.plot(tfbL, np.abs(dataL[0]), c = 'C1', label = 'xmin Low')
        plt.plot(tfb, np.abs(data[0][1:]), c = 'yellowgreen', label = 'xmin Fid')
        plt.plot(tfbH, np.abs(dataH[0][1:]), c = 'darkviolet', label = 'xmin High')
        plt.title(f'{label_coord[i]}', fontsize = 20)
        plt.yscale('log')
        plt.xlabel(r't [t$_{\rm fb}$]')
        plt.ylabel(r'$|$lim box$| [R_\odot]$')
        plt.legend()