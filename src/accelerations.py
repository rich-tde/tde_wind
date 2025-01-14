""" Compute the accelerations due to centrifugal force and radial pathssure to understand the Eddingotn envelope."""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.coordinates as coord
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, J_cart_in_sphere

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
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 348

Rt = Rstar * (Mbh/mstar)**(1/3)
apo = orb.apocentre(Rstar, mstar, Mbh, beta) 
amin = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
solarR_to_au = 215

def dPdR(DpDx, DpDy, DpDz, lat, long):
    dP_radial = DpDx * np.sin(lat) * np.cos(long) + DpDy * np.sin(lat) * np.sin(long) + DpDz * np.cos(lat)
    return dP_radial

def v_phi(Vx, Vy, long):
    v_phi = - Vy * np.sin(long) + Vx * np.cos(long)
    return v_phi

def v_theta(Vx, Vy, Vz, lat, long):
    v_theta = Vx * np.cos(lat) * np.cos(long) + Vy * np.cos(lat) * np.sin(long) - Vz * np.sin(lat)
    return v_theta

# Load the data
path = f'{abspath}/TDE/{folder}/{snap}' #f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
data = make_tree(path, snap, energy = False)
X, Y, Z, Den, Vx, Vy, Vz, Vol = data.X, data.Y, data.Z, data.Den, data.VX, data.VY, data.VZ, data.Vol
R, lat, long = coord.cartesian_to_spherical(X, Y, Z) # r, latitude, longitude 
DpDx = np.load(f'{path}/DpDx_{snap}.npy')
DpDy = np.load(f'{path}/DpDy_{snap}.npy')
DpDz = np.load(f'{path}/DpDz_{snap}.npy')
dP_radial = dPdR(DpDx, DpDy, DpDz, lat, long)

# compute accelerations
a_P = np.abs(dP_radial) / Den
V_phi = v_phi(Vx, Vy, long)
a_centrifugal = V_phi**2 / R
# V_theta = v_theta(Vx, Vy, Vz, lat, long)
ratio = a_centrifugal / a_P

#%%
mid = np.logical_and(np.abs(Z) < Vol**(1/3), Den > 1e-19)
x_mid, y_mid, z_mid, den_mid, ratio_mid = X[mid], Y[mid], Z[mid], Den[mid], ratio[mid]

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20, 5))
img = ax1.scatter(x_mid/solarR_to_au, y_mid/solarR_to_au, c=den_mid, cmap='plasma', s = 1, norm=colors.LogNorm(vmin=1e-10, vmax=1e-6))
ax1.scatter(x_mid[ratio_mid<1]/solarR_to_au, y_mid[ratio_mid<1]/solarR_to_au, c='k',s = 1, norm=colors.LogNorm(vmin=1e-20, vmax=1e-19))
cbar = plt.colorbar(img)
cbar.set_label(r'$\rho$', fontsize = 20)
img = ax2.scatter(x_mid/solarR_to_au, y_mid/solarR_to_au, c=ratio_mid, cmap='viridis', s = 1, norm=colors.LogNorm(vmin=1e-2, vmax=1e2))
cbar = plt.colorbar(img)
cbar.set_label(r'$a_{\rm \omega}/a_{\rm P}$', fontsize = 20)
ax1.set_ylabel(r'$Y [AU]$', fontsize = 20)
for ax in [ax1, ax2]:
    ax.scatter(0, 0, c='k', s = 10)
    ax.set_xlabel(r'$X [AU]$', fontsize = 20)
    ax.set_xlim(-4, 2)
    ax.set_ylim(-2, 2)

# %%
