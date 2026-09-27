abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
# import resource
import gc
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
from Utilities import prelude as prel 

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 151
params = [m, Rstar, mstar, beta, n, compton]
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre_saving = f'{abspath}/data/{folder}'

photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
alpha_scatter_ph, alpha_abs_ph = photo['alpha_scatter'], photo['alpha_abs']
alpha_ph_tot = alpha_scatter_ph + alpha_abs_ph

Rcol = np.load(f"{pre_saving}/spectra/{check}_Rcol{snap}.npz", allow_pickle=True)
alpha_scatter_col, alpha_abs_col = Rcol['alpha_scatter'], Rcol['alpha_abs']
alpha_col_tot = alpha_scatter_col + alpha_abs_col

Npix = hp.nside2npix(prel.NSIDE)
observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) # shape: (3, 192)
x_obs, y_obs, z_obs = observers_xyz

plt.figure(figsize=(7, 6))
plt.scatter(np.arange(192), alpha_abs_ph/alpha_ph_tot, label = r'$r_{ph}$')
plt.scatter(np.arange(192), alpha_abs_col/alpha_col_tot, label = r'$r_{col}$')
plt.ylabel(r'$\alpha_{\rm abs}/\alpha_{\rm tot}$')
plt.xlabel('Observer')
plt.legend(fontsize=15)
plt.grid()