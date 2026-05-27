""" Fit a geometrical shape to the photosphere"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import sort_list
from src.Wind.polarization import ellipsoid_surface

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 109
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

from scipy.linalg import svd, inv

def mvee_fit(points, tol=1e-6, max_iter=1000):
    """
    Minimum Volume Enclosing Ellipsoid (MVEE) for N points in d dimensions.

    Based on Khachiyan's algorithm.
    Returns:
        center: (d,)
        shape: (d,d) such that   (x - c) @ shape @ (x - c) <= 1
    """
    points = np.atleast_2d(points)
    d, N = points.shape[1], len(points)

    # Khachiyan algorithm
    Q = np.column_stack([points, np.ones(N)])  # (N, d+1)
    uu = np.ones(N) / N

    for it in range(max_iter):
        X = Q.T @ np.diag(uu) @ Q
        M = np.diag(Q @ inv(X) @ Q.T)  # leverage scores
        j = np.argmax(M)
        step_size = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        new_uu = (1 - step_size) * uu
        new_uu[j] += step_size
        if np.max(np.abs(new_uu - uu)) < tol:
            uu = new_uu
            break
        uu = new_uu

    # Center and shape matrix
    U = np.diag(uu)
    c = points.T @ uu            # center
    X = (points - c.reshape(1,-1)).T @ U @ (points - c.reshape(1,-1))
    Xinv = inv(X) / d
    return np.array(c), Xinv

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
snaps_lum = snaps_lum.astype(int)
time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
x_heal, y_heal, z_heal = observers_xyz[0], observers_xyz[1], observers_xyz[2]

photo = np.load(f'{abspath}/data/{folder}/photoPOL/{check}_photo{snap}.npz')
x, y, z = photo['x'], photo['y'], photo['z']
r = np.sqrt(x**2 + y**2 + z**2)
all_pts = np.column_stack([x, y, z])  # shape (N, 3)

centre, Xinv = mvee_fit(all_pts)

# To get radii and axes, diagonalize the shape matrix
w, v = np.linalg.eigh(Xinv)  # w = eigenvalues, v = eigenvectors
w = np.clip(w, 1e-15, None)   # avoid negative eigenvalues from rounding
abc = 1.0 / np.sqrt(w)


a_fit, b_fit, c_fit = abc
x0_fit, y0_fit, z0_fit = centre
print(f'Fitted ellipsoid parameters: a={a_fit}, b={b_fit}, c={c_fit}, center=({x0_fit}, {y0_fit}, {z0_fit})')
x_ell_fit, y_ell_fit, z_ell_fit = ellipsoid_surface(prel.NSIDE, a_fit, b_fit, c_fit, x0=x0_fit, y0=y0_fit, z0=z0_fit, healpix=True, stay_helpix=True)

fig = plt.figure(figsize=(35, 8))
ax1 = fig.add_subplot(111, projection = '3d')
ax1.scatter(x/330, y/330, z/330, s = 40)
ax2 = fig.add_subplot(121, projection = '3d') 
ax2.scatter(x_ell_fit/330, y_ell_fit/330, z_ell_fit/330, s = 40)
for ax in [ax1, ax2]:
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-10, 2.5); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
    ax.set_xlabel(r'x / r$_{\rm a}$', labelpad=15)
    ax.set_ylabel(r'y / r$_{\rm a}$', labelpad=15)
    ax.set_zlabel(r'z / r$_{\rm a}$', labelpad=15)
plt.tight_layout()
