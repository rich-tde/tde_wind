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
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
Rt = orb.tidal_radius(Rstar, mstar, Mbh)

check = '' # 'Low' or 'HiRes'
snap = 237
xchosen = Rt
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
tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
mass, vol, ie_den, rad_den = data.Mass, data.Vol, data.IE, data.Rad
orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
rad = rad_den * vol
ie = ie_den * vol
dim_cell = (3/(4*np.pi) * vol)**(1/3)
ie_onmass = (ie_den / data.Den) * prel.en_converter/prel.Msol_cgs
orb_en_onmass = (orb_en / mass) * prel.en_converter/prel.Msol_cgs
rad_den = rad_den * prel.en_den_converter
yz_cut = np.abs(data.X-xchosen) < dim_cell
X_yz, Y_yz, Z_yz, dim_yz, Temp_yz, Rad_yz, Den_yz, mass_yz, orb_en_yz, ie_yz, rad_yz, IE_onmass_yz, rad_den_yz = \
    sec.make_slices([data.X, data.Y, data.Z, dim_cell, data.Temp, data.Rad, data.Den, data.Mass, orb_en, ie, rad, ie_onmass, rad_den], yz_cut)
#%% YZ plane plot
fig, ax = plt.subplots(1,1, figsize = (12,10))
img = ax.scatter(Y_yz/Rt, Z_yz/Rt, c = X_yz/Rt,  cmap = 'jet', s = 20, vmin = 0.5, vmax = 1.5)
cbar = plt.colorbar(img)
ax.axvline(0, color = 'black')
ax.set_xlim(-1.2,1.2)
ax.set_ylim(0,25)
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$z$')
plt.suptitle(f'YZ plane at X={np.round(xchosen,2)}, t = {np.round(tfb, 2)}' + r' $t_{fb}$', fontsize = 25)
plt.tight_layout()
# %%
