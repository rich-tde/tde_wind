""" Plot photosphere and colorsphere."""

abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import Utilities.prelude as prel
import healpy as hp
import matplotlib.colors as colors
from src import orbits as orb
from Utilities.operators import choose_observers
mpl.rcParams['savefig.transparent'] = False  # Only figure patch
mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
mpl.rcParams['axes.facecolor'] = 'white' 

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
choice = 'split_stream'
snaps = [76, 109, 151]
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre_saving = f'{abspath}/data/{folder}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
apo = things['apo']
Rt = things['Rt']

observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_obs, label_obs, color_obs, lines_obs, _, _ = choose_observers(observers_xyz, choice)
fig, axes = plt.subplots(1, len(snaps), figsize=(6*len(snaps), 6))
for s, snap in enumerate(snaps):
    tfb = np.loadtxt(f'{abspath}/TDE/{folder}/{snap}/tfb_{snap}.txt')
    photosphere = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    x_ph, y_ph, z_ph = photosphere['x'], photosphere['y'], photosphere['z']
    alpha_rossland_ph, alpha_scatter_ph, alpha_abs_ph = photosphere['alpha_rossland'], photosphere['alpha_scatter'], photosphere['alpha_abs']

    colorsphere = np.load(f'{abspath}/data/{folder}/spectra/{check}_Rcol{snap}.npz')
    x_th, y_th, z_th, den_th = colorsphere['x'], colorsphere['y'], colorsphere['z'], colorsphere['den']
    r_th = np.sqrt(x_th**2 + y_th**2 + z_th**2)
    alpha_rossland_th, alpha_scatter_th, alpha_abs_th = colorsphere['alpha_rossland'], colorsphere['alpha_scatter'], colorsphere['alpha_abs']

    axes[s].set_title(f't = {tfb:.2f}' + r' $t_{\rm fb}$', fontsize=18)
    for idx, index in enumerate(indices_obs):
        index = index[r_th[index] > Rt]
        axes[s].scatter(r_th[index]/Rt, alpha_abs_th[index]/alpha_rossland_th[index], color = color_obs[idx]) #, label=r'$\alpha_{\rm abs}$' if idx == 0 else "")
        
for ax in axes:
    ax.set_xlabel(r'$r/r_t$')
    ax.tick_params(which='major', length = 7)
    ax.tick_params(which='minor', length = 5) 
    ax.loglog()
axes[0].legend()
axes[0].set_ylabel(r'$\alpha_{\rm abs, Pl}/\alpha_{\rm Ross}$')
plt.savefig(f'{abspath}/Figs/2.paperWind/referee/alphas.png', dpi=300, bbox_inches='tight')