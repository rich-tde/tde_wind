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
from 

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
step = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

check = 'Low' # 'Low' or 'HiRes'
check1 = 'HiRes'
snap = 164
save = False

path = f'/Users/paolamartire/shocks/TDE/{folder}{check}{step}/{snap}'
path1 = f'/Users/paolamartire/shocks/TDE/{folder}{check1}{step}/{snap}'
Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

# Load data
data = make_tree(path, snap, energy = True)
Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
mass, vol, ie_den, rad_den = data.Mass, data.Vol, data.IE, data.Rad
orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
rad = rad_den * vol
ie = ie_den * vol
dim_cell = (3/(4*np.pi) * vol)**(1/3)
ie_onmass = (ie_den / data.Den) * prel.en_converter/prel.Msol_to_g
orb_en_onmass = (orb_en / mass) * prel.en_converter/prel.Msol_to_g

tfb = days_since_distruption(f'{path1}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')

data1 = make_tree(path1, snap, energy = True)
THETA1, RADIUS_cyl1 = to_cylindric(data1.X, data1.Y)
R1 = np.sqrt(data1.X**2 + data1.Y**2 + data1.Z**2)
vel1 = np.sqrt(data1.VX**2 + data1.VY**2 + data1.VZ**2)
dim_cell1 = data1.Vol**(1/3) 
ie_onmass1 = data1.IE / data1.Den
ie1 = ie_onmass1 * data1.Mass
rad1 = data1.Rad * data1.Vol
orb_en1 = orb.orbital_energy(R1, vel1, data1.Mass, G, c, Mbh)


# cut
xchosen = 0
yz_cut = np.abs(data.X-xchosen) < dim_cell
Y_yz, Z_yz, dim_yz, Rad_yz, Den_yz, mass_yz, orb_en_yz, ie_yz, rad_yz, IE_onmass_yz = \
    sec.make_slices([data.Y, data.Z, dim_cell, data.Rad, data.Den, data.Mass, orb_en, ie, rad, ie_onmass], yz_cut)
yz_cut1 = np.abs(data1.X-xchosen) < dim_cell1
Y1_yz, Z1_yz, dim1_yz, Rad1_yz, Den1_yz, mass1_yz, orb_en1_yz, ie1_yz, rad1_yz, IE1_onmass_yz = \
    sec.make_slices([data1.Y, data1.Z, dim_cell1, data1.Rad, data1.Den, data1.Mass, orb_en1, ie1, rad1, ie_onmass1], yz_cut1)

# YZ plane 
fig, ax = plt.subplots(1,2, figsize = (20,6))
img1 = ax[0].scatter(Y_yz/apo, Z_yz/apo, c = IE_onmass_yz,  cmap = 'viridis', s = 4, norm = colors.LogNorm(vmin = 7e12, vmax = 3.5e14))#np.max(Den_tra)))
cbar1 = plt.colorbar(img1)
# cbar1.set_label(r' Radiation energy density', fontsize = 18)
# ax[0].scatter(0, z_stream[idx], marker = 'x', s = 37, c = 'k', alpha = 1)
ax[0].set_title('Low Res', fontsize = 18)

img1 = ax[1].scatter(Y_yz/apo, Z_yz/apo, c = IE_onmass_yz,  cmap = 'viridis', s = 4, norm = colors.LogNorm(vmin = 7e12, vmax = 3.5e14))#np.max(Den_tra1)))
cbar1 = plt.colorbar(img1)
cbar1.set_label(r' Specific IE [erg/g]', fontsize = 18)
# ax[1].scatter(0, z_stream1[idx], marker = 'x', s = 37, c = 'k', alpha = 1)
ax[1].set_title('High Res', fontsize = 18)
for i in range(2):
    ax[i].set_xlim(-0.4,0.1)#xlim_neg, xlim)
    ax[i].set_ylim(-1, 1)#ylim_neg, ylim)
    ax[i].set_xlabel(r'Y/$R_a$', fontsize = 18)
    ax[i].set_ylabel(r'Z/$R_a$', fontsize = 18)
    ax[i].text(-0.4, -.9, f't = {np.round(tfb, 2)}' + r'$t_{fb}$', fontsize = 20)
plt.suptitle(f'YZ plane at X={xchosen}', fontsize = 20)
plt.tight_layout()