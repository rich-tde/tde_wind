""" Find sizes and an approximation of the number of cells at Rp using height scale """
import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
    path = f'{abspath}/TDE'

import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
# import colorcet
import k3match # ciao Nicco, mi sono appena ricordata di questo consiglio del primo anno
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.selectors_for_snap import select_snap
from Utilities.operators import make_tree

#
# PARAMETERS
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 
compton = 'Compton'
check = ''
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt / beta
omegaRp = np.sqrt(prel.G*Mbh/Rp**3)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Rp_min_size = np.zeros(len(snaps))
Rp_avg_size = np.zeros(len(snaps))
scale_height = np.zeros(len(snaps))
for i, snap in enumerate(snaps):
    print(snap, flush=False)
    sys.stdout.flush()
    # Load data and slice at midplane
    if alice:
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    else:
        path = f'{abspath}/TDE/{folder}/{snap}'
    data = make_tree(path, snap, energy = False)
    X, Y, Z, Vol, Den, Press = data.X, data.Y, data.Z, data.Vol, data.Den, data.Press
    dim_cell = Vol**(1/3) 
  
    # find the nearest cells to Rp within a sphere of diameter = 0.5 
    indices_Rp, _, dist = k3match.cartesian(X, Y, Z, Rp, 0, 0, 0.25)
    if len(indices_Rp) == 0:
        print('No cells found at 0.25 from Rp')
        idxRp, _ = k3match.nearest_cartesian(X, Y, Z, Rp, 0, 0)
        idxRp = idxRp[0]
        indices_Rp = idxRp
    else:
        idxRp = indices_Rp[np.argmin(dist)]

    Rp_min_size[i] = np.min(dim_cell[indices_Rp])
    Rp_avg_size[i] = np.mean(dim_cell[indices_Rp])
    csmid = np.sqrt(5/3 * Press[indices_Rp]/Den[indices_Rp])
    Hmid = csmid/omegaRp
    scale_height[i] = np.mean(Hmid)

if alice:
    with open(f'{abspath}/data/{folder}/Rp_statistics.txt', 'a') as file:
        file.write(f'# Minimum size around pericenter [R_\odot] \n' + ' '.join(map(str, Rp_min_size)) + '\n')
        file.write(f'# Average size around pericenter [R_\odot] \n' + ' '.join(map(str, Rp_avg_size)) + '\n')
        file.write(f'# Average scale height [R_\odot] as H = c_s / Omega(Rp) with c_s = sqrt(P/Den) \n' + ' '.join(map(str, scale_height)) + '\n')
        file.close()