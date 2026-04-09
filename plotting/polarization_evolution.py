abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import sort_list
from src.Wind.polarization import compute_polarization

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)

# n_obs_all_params = [[[1, 0, 0], 'solid', 'navy'],
#             [[-1, 0, 0], 'dashed', 'dodgerblue'],
#             [[0, 1, 0], 'solid', 'darkorange'],
#             [[0, -1, 0], 'dashed', 'r'],
#             [[0, 0, 1], 'solid', 'forestgreen'],
#             [[0, 0, -1], 'dashed', 'yellowgreen'],
#             [[1, 0, 1], 'dotted', 'k'],
#             [[np.sin(np.pi/3), 0, np.cos(np.pi/3)], 'dotted', 'magenta']]
# n_obs_all = [params[0] for params in n_obs_all_params]
# P_all = np.zeros((len(snaps), len(n_obs_all)))

P_all = np.zeros(len(snaps))
n_obs = [0,0,1]
for s, snap in enumerate(snaps): 
    print(snap)
    photo = np.loadtxt(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}POL.txt')
    x, y, z, Lum, Fx, Fy, Fz = photo[0], photo[1], photo[2], photo[14], photo[16], photo[17], photo[18]
    # for i_o, n_obs in enumerate(n_obs_all):
    P, I, Q, U = compute_polarization(x, y, z, Fx, Fy, Fz, n_obs, flux=True)
    P_all[s] = P
    # print(P)

#%%
plt.plot(tfb, P_all) 
# plt.ylim(0, 1)

print(P_all[-1])
# %%
