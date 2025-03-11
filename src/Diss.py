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
import Utilities.sections as sec

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
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

Rdiss = np.zeros(len(snaps))
Eradtot = np.zeros(len(snaps))
Ldisstot = np.zeros(len(snaps))
# for the cut in density 
Rdiss_cut = np.zeros(len(snaps))
Eradtot_cut = np.zeros(len(snaps))
Ldisstot_cut = np.zeros(len(snaps))
for i,snap in enumerate(snaps):
    print(snap, flush=False) 
    sys.stdout.flush()
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    X, Y, Z, vol, den, Rad_den, Ediss_den = \
        data.X, data.Y, data.Z, data.Vol, data.Den, data.Rad, data.Diss
    # box = np.load(f'{path}/box_{snap}.npy')
    # if int(snap) <= 317:
    #     boxL = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/snap_{snap}/box_{snap}.npy')
    #     if int(snap) <= 267:
    #         boxH = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/snap_{snap}/box_{snap}.npy')
    #     else:
    #         boxH = box
    # else: 
    #     boxH = box
    #     boxL = box
    # xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0], -0.75*apo]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
    # xmax, ymax, zmax = np.min([box[3], boxL[3], boxH[3]]), np.min([box[4], boxL[4], boxH[4]]), np.min([box[5], boxL[5], boxH[5]])
    # cutx = (X > xmin) & (X < xmax)
    # cuty = (Y > ymin) & (Y < ymax)
    # cutz = (Z > zmin) & (Z < zmax)
    # cut_coord = cutx & cuty & cutz
    # X, Y, Z, vol, den, Rad_den, Ediss_den = \
    #     sec.make_slices([X, Y, Z, vol, den, Rad_den, Ediss_den], cut_coord)
        
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    # Ediss = np.abs(Ediss_den) * vol # energy dissipation rate [energy/time] in code units
    # Rdiss[i] = np.sum(Rsph * Ediss) / np.sum(Ediss)
    # Eradtot[i] = np.sum(Rad_den * vol)
    # Ldisstot[i] = np.sum(Ediss)

    cut = den > 1e-19
    Rsph_cut, vol_cut, Rad_den_cut, Ediss_den_cut = Rsph[cut], vol[cut], Rad_den[cut], Ediss_den[cut]
    Ediss_cut = np.abs(Ediss_den_cut) * vol_cut # energy dissipation rate [energy/time] in code units
    Rdiss_cut[i] = np.sum(Rsph_cut * Ediss_cut) / np.sum(Ediss_cut)
    Eradtot_cut[i] = np.sum(Rad_den_cut * vol_cut)
    Ldisstot_cut[i] = np.sum(Ediss_cut)

# with open(f'{abspath}/data/{folder}/Rdiss_{check}cutCoord.txt','a') as file:
#     file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
#     file.write(f'# Rdiss [R_\odot] \n' + ' '.join(map(str, Rdiss)) + '\n')
#     file.write(f'# Total radiation energy [energy in code units] \n' + ' '.join(map(str, Eradtot)) + '\n')
#     file.write(f'# Total dissipation luminosity [energy/time in code units] \n' + ' '.join(map(str, Ldisstot)) + '\n')
#     file.close()

# with open(f'{abspath}/data/{folder}/Rdiss_{check}cutDenCoord.txt','a') as file:
with open(f'{abspath}/data/{folder}/Rdiss_{check}.txt','a') as file:
    file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
    file.write(f'# Rdiss [R_\odot] \n' + ' '.join(map(str, Rdiss_cut)) + '\n')
    file.write(f'# Total radiation energy [energy in code units] \n' + ' '.join(map(str, Eradtot_cut)) + '\n')
    file.write(f'# Total dissipation luminosity [energy/time in code units] \n' + ' '.join(map(str, Ldisstot_cut)) + '\n')
    file.close()

#     cut = den > 1e-19
#     Rsph_cut, vol_cut, Rad_den_cut, Ediss_den_cut = Rsph[cut], vol[cut], Rad_den[cut], Ediss_den[cut]
#     Ediss_cut = np.abs(Ediss_den_cut) * vol_cut # energy dissipation rate [energy/time] in code units
#     Ldisstot_cut[i] = np.sum(Ediss_cut)

# with open(f'{abspath}/data/{folder}/Ldiss_{check}.txt','a') as file:
#     file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
#     file.write(f'# Total dissipation luminosity [energy/time in code units] \n' + ' '.join(map(str, Ldisstot_cut)) + '\n')
    # file.close()
