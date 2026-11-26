import sys

from shapely import snap

sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
import matplotlib.colors as colors
import Utilities.prelude as prel
from src import orbits as orb
from plotting.paperPol.P_moll import extract_poli

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'MG' 
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
NSIDE = 8 
snap = 320

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600
observers_xyz = hp.pix2vec(NSIDE, range(hp.nside2npix(NSIDE)))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
latitude_moll = np.arcsin(z / r)  
longitude_moll = np.arctan2(y, x)

# Regular grid in Mollweide coordinates
nlon = 720
nlat = 360

lon_grid = np.linspace(-np.pi, np.pi, nlon)
lat_grid = np.linspace(-np.pi/2, np.pi/2, nlat)

lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)


# healpy uses theta = colatitude, phi = longitude in [0, 2pi)
theta = np.pi/2 - lat_mesh
phi = np.mod(lon_mesh, 2*np.pi)

Rphoto = np.loadtxt(f'{abspath}/data/{folder}{check}/snap_{snap}/Rphoto_snap_{snap}.txt')
Eg = np.loadtxt(f'{abspath}/data/{folder}{check}/snap_{snap}/Eg_snap_{snap}.txt')
time = np.loadtxt(f'{abspath}/data/{folder}{check}/snap_{snap}/Time_snap_{snap}.txt')
time = time * prel.tsol_cgs / tfallback_cgs
Lumg = 4 * np.pi * Rphoto**2 * Eg * prel.c_cgs
opt_uv = np.sum(Lumg[:, 0:5], axis=1)
opt_X = np.sum(Lumg[:, 8:], axis=1)
ratio = opt_X / opt_uv

ratio_grid = hp.get_interp_val(
    ratio,
    theta,
    phi)

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1, projection='mollweide')# if moll_spl == '' else None)
ax.set_title(f't = {time:.2f} tfb', fontsize=20)
ax.grid()
img = ax.pcolormesh(lon_mesh, lat_mesh, np.log10(ratio_grid), cmap='rainbow',  vmin = -4, vmax = -2)
cbar = plt.colorbar(img)#, orientation='horizontal')
cbar.set_label(r'$\log_{10} (L_{\rm opt+UV}/L_{\rm X})$', fontsize=20)
fig.tight_layout()
plt.savefig(f'{abspath}/Figs/2.paperWind/ratioXopt_{snap}.png', dpi=300)

# MC
groups = np.arange(10)
path_file =  f'{abspath}/TDE/{folder}MG/Polarization/{snap}/tde_7500polarization{snap}.vtk'
data = extract_poli(path_file,
                fields=[f"forward_polarization_degree"])   
x = data["x"]
y = data["y"]
z = data["z"]
theta = data["theta"]
phi = data["phi"]

opt_uv = np.zeros(len(x))
xray = np.zeros(len(x))
for group in groups:
    data = extract_poli(
        path_file,
        fields=[f'forward_group_{group}_luminosity'])   
    if group < 5:
        opt_uv += data[f'forward_group_{group}_luminosity']
    if group == 8 or group == 9:
        xray += data[f'forward_group_{group}_luminosity']
ratio = xray / opt_uv

longitude_moll = phi 
latitude_moll = np.pi/2 - theta
nlon = 360
nlat = 180
lon_grid = np.linspace(longitude_moll.min(), longitude_moll.max(), nlon)
lat_grid = np.linspace(latitude_moll.min(), latitude_moll.max(), nlat)
lon_mc, lat_mc = np.meshgrid(lon_grid, lat_grid)
ratio_mc = griddata(
    points=(longitude_moll, latitude_moll),
    values=ratio,
    xi=(lon_mc, lat_mc),
    method='linear')

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1, projection='mollweide')# if moll_spl == '' else None)
ax.set_title(f't = {time:.2f} tfb', fontsize=20)
ax.grid()
img = ax.pcolormesh(lon_mc, lat_mc, np.log10(ratio_mc), cmap='rainbow',  vmin = -4, vmax = -2)
cbar = plt.colorbar(img)#, orientation='horizontal')
cbar.set_label(r'$\log_{10} (L_{\rm opt+UV}/L_{\rm X})$', fontsize=20)
fig.tight_layout()
plt.savefig(f'{abspath}/Figs/2.paperWind/ratioXopt_MC{snap}.png', dpi=300)
