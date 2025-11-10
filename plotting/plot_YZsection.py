abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
matplotlib.rcParams['figure.dpi'] = 150
import Utilities.prelude as prel
from Utilities.operators import make_tree
import Utilities.sections as sec
import src.orbits as orb
import colorcet
from Utilities.time_extractor import days_since_distruption

#
## CONSTANTS
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

#
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' # 'Low' or 'HiRes'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Rt = orb.tidal_radius(Rstar, mstar, Mbh)

snap = 76
xchosen = 0
energy = 'internal' # 'internal' or 'radiation'
save = False

path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

#%% Load data and cut
data = make_tree(path, snap, energy = True)
tfb = np.loadtxt(f'/Users/paolamartire/shocks/TDE/{folder}/{snap}/tfb_{snap}.txt')
Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
mass, vol, ie_den, rad_den = data.Mass, data.Vol, data.IE, data.Rad
dim_cell =  vol**(1/3)
rad_den = rad_den * prel.en_den_converter
yz_cut = np.abs(data.X-xchosen) < dim_cell
X_yz, Y_yz, Z_yz, dim_yz, Temp_yz, Rad_yz, Den_yz, mass_yz = \
    sec.make_slices([data.X, data.Y, data.Z, dim_cell, data.Temp, data.Rad, data.Den, data.Mass], yz_cut)
#%% YZ plane plot
fig, ax = plt.subplots(1,1, figsize = (10,7))
img = ax.scatter(Y_yz/Rt, Z_yz/Rt, c = Den_yz,  cmap = 'rainbow', s = 20, norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-8))
cbar = plt.colorbar(img)
ax.set_xlim(-1,1)
ax.set_ylim(-.4, .4)
ax.set_xlabel(r'$y [R_t]$')
ax.set_ylabel(r'$z [R_t]$')
# plt.suptitle(f'YZ plane at X={np.round(xchosen,2)}, t = {np.round(tfb, 2)}' + r' $t_{fb}$', fontsize = 25)
plt.tight_layout()
# %%
