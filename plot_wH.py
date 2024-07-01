import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from Utilities.basic_units import radians

from Utilities.operators import make_tree, Ryan_sampler, to_cylindric
import Utilities.sections as sec
import src.orbits as orb
from Utilities.time_extractor import days_since_distruption
import Utilities.prelude
matplotlib.rcParams['figure.dpi'] = 150

#
## Parameters
#

#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'Low' # '' or 'HiRes' or 'Res20'
check1 = 'HiRes' 
check2 = 'Res20'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
snap = '199'
snap2 = '169'
path = f'TDE/{folder}{check}/{snap}'
path1 = f'TDE/{folder}{check1}/{snap}'
path2 = f'TDE/{folder}{check2}/{snap2}'
threshold =  1/3
step = 0.02

#
## Constants
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
saving_path = f'Figs/{folder}'

##
#  Main
##
high = True
if snap == '199':
    high = False
# Load data (ig high == True, do it also for high resolution)
#%%
data = make_tree(path, snap, energy = False)
data1 = make_tree(path1, snap, energy = False)
tfb = days_since_distruption(f'{path1}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
time_to_print = np.round(tfb,1)
#%% 
density = np.load(f'TDE/{folder}{check}/{snap}/smoothed_Den_{snap}.npy') 
density1 = np.load(f'TDE/{folder}{check1}/{snap}/smoothed_Den_{snap}.npy') 
# W and H
datawidth5 = np.loadtxt(f'data/{folder}/width_time0.5_thr{np.round(threshold,1)}.txt')
theta_width = datawidth5[0]
widthL5 = datawidth5[1]
NcellL5 = datawidth5[2]
widthMid5 = datawidth5[3]
NcellMid5 = datawidth5[4]
widthHigh5 = datawidth5[5]
NcellHigh5 = datawidth5[6]
dataheight5 = np.loadtxt(f'data/{folder}/height_time0.5_thr{np.round(threshold,1)}.txt')
theta_height = dataheight5[0]
heightL5 = dataheight5[1]
NhcellL5 = dataheight5[2]
heightMid5 = dataheight5[3]
NhcellMid5 = dataheight5[4]
heightHigh5 = dataheight5[5]
NhcellHigh5 = dataheight5[6]
datawidth7 = np.loadtxt(f'data/{folder}/width_time0.7_thr{np.round(threshold,1)}.txt')
widthL7 = datawidth7[1]
NcellL7 = datawidth7[2]
widthMid7 = datawidth7[3]
NcellMid7 = datawidth7[4]
dataheight7 = np.loadtxt(f'data/{folder}/height_time0.7_thr{np.round(threshold,1)}.txt')
heightL7 = dataheight7[1]
NhcellL7 = dataheight7[2]
heightMid7 = dataheight7[3]
NhcellMid7 = dataheight7[4]
# Stream 
streamL = np.load(f'data/{folder}/stream_{check}{snap}_{step}.npy')
indeces_orbit = streamL[1].astype(int)
streamMid = np.load(f'data/{folder}/stream_{check1}{snap}_{step}.npy')
indeces_orbit1 = streamMid[1].astype(int)

# Find R_stream
_, RADIUS_cyl = to_cylindric(data.X, data.Y)
_, RADIUS_cyl1 = to_cylindric(data1.X, data1.Y)
r_stream, r_stream1 = \
    RADIUS_cyl[indeces_orbit], RADIUS_cyl1[indeces_orbit1]
density_stream, density_stream1 = \
    density[indeces_orbit], density1[indeces_orbit1]

#%%
if high:
    data2 = make_tree(path2, snap2, energy = False)
    density2 = np.load(f'TDE/{folder}{check2}/{snap2}/smoothed_Den_{snap2}.npy') 
    streamHigh = np.load(f'data/{folder}/stream_{check2}{snap2}_{step}.npy')
    indeces_orbit2 = streamHigh[1].astype(int)
    _, RADIUS_cyl2 = to_cylindric(data2.X, data2.Y)
    r_stream2 = RADIUS_cyl2[indeces_orbit2]
    density_stream2 = density2[indeces_orbit2]

#%% Plot
img, ax = plt.subplots(2,1, figsize=(8, 6))
ax[0].plot(theta_width, r_stream, c = 'k', label = f'Low {time_to_print}')
ax[0].plot(theta_width, r_stream1, c = 'r', label = f'Mid {time_to_print}')
if high:
    ax[0].plot(theta_width, r_stream2, c = 'b', label = f'High {time_to_print}')
ax[0].set_xlim(theta_width[30], theta_width[230])
ax[0].set_ylim(0, 80)
ax[0].set_ylabel(r'$R_{stream}$', fontsize = 16)
ax[1].plot(theta_width, density_stream, c = 'k', label = f'Low {time_to_print}')
ax[1].plot(theta_width, density_stream1, c = 'r', label = f'Mid {time_to_print}')
if high:
    ax[1].plot(theta_width, density_stream2, c = 'b', label = f'High {time_to_print}')
ax[1].set_xlim(theta_width[30], theta_width[230])
ax[1].set_ylim(0, 1e-6)
ax[1].set_xlabel(r'$\theta$', fontsize = 16)
ax[1].set_ylabel(r'$\rho$', fontsize = 16)
ax[1].legend()
plt.suptitle(f'Time {time_to_print}', fontsize = 16)
plt.savefig(f'{saving_path}/theta_stream{time_to_print}.png')
plt.show()



