abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(f'{abspath}')
import numpy as np
import matplotlib.pyplot as plt
# import colorcet
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.sections import make_slices

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
snap = '348'

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

# Load data
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
X = np.load(f'{path}/CMx_{snap}.npy')
Y = np.load(f'{path}/CMy_{snap}.npy')
Z = np.load(f'{path}/CMz_{snap}.npy')
Den = np.load(f'{path}/Den_{snap}.npy')
Mass = np.load(f'{path}/Mass_{snap}.npy')
Temp = np.load(f'{path}/T_{snap}.npy')
cut = Den > 1e-19
X, Y, Z, Den, Mass, Temp = make_slices([X, Y, Z, Den, Mass, Temp], cut)
dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/rich_photo{snap}.txt')
xph, yph, zph, volph, denph, Tempph = dataph[0], dataph[1], dataph[2], dataph[3], dataph[4], dataph[5]
massph = denph * volph


#%% Make a 3D plot for T
xlimmin, xlimmax = -7, 5
ylimmin, ylimmax = -4, 4
zlimmin, zlimmax = -4, 4
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(xph/prel.solarR_to_au, yph/prel.solarR_to_au, zph/prel.solarR_to_au, c=Tempph, s = 30, marker = 's', cmap='hot', norm = colors.LogNorm(vmin = 4e3, vmax = 1e5))
cb = fig.colorbar(img)
cb.set_label(r'T [K]')
img1 = ax.scatter(X[::50]/prel.solarR_to_au, Y[::50]/prel.solarR_to_au, Z[::50]/prel.solarR_to_au, c=Den[::50]*prel.den_converter, s=1, cmap = 'viridis', norm = colors.LogNorm(vmin = 5e-12, vmax = 5e-8), alpha = 0.5)
cb1 = fig.colorbar(img1)
cb1.set_label(r'Density [g/cm$^3$]')
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_xlim(xlimmin, xlimmax)
ax.set_ylim(ylimmin, ylimmax)
ax.set_zlim(zlimmin, zlimmax)
plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/3Dproj{snap}T.png')

#%% Make a 3D plot for mass
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(xph/prel.solarR_to_au, yph/prel.solarR_to_au, zph/prel.solarR_to_au, c=massph, cmap='winter', marker= 's', s= 30,  norm = colors.LogNorm(vmin = 1e-9, vmax = 1e-7))
cb = fig.colorbar(img)
cb.set_label(r'Mass [$M_\odot$]')
# img1 = ax.scatter(X[::50]/prel.solarR_to_au, Y[::50]/prel.solarR_to_au, Z[::50]/prel.solarR_to_au, c=Den[::50]*prel.den_converter, s=1, cmap = 'viridis', norm = colors.LogNorm(vmin = 5e-12, vmax = 5e-8), alpha = 0.5)
img1 = ax.scatter(X[::50]/prel.solarR_to_au, Y[::50]/prel.solarR_to_au, Z[::50]/prel.solarR_to_au, c=Mass[::50], s=1, cmap = 'winter', norm = colors.LogNorm(vmin = 1e-9, vmax = 1e-7))
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_xlim(xlimmin, xlimmax)
ax.set_ylim(ylimmin, ylimmax)
ax.set_zlim(zlimmin, zlimmax)
plt.savefig(f'/Users/paolamartire/shocks/Figs/EddingtonEnvelope/3Dproj{snap}Mass.png')
# %%
