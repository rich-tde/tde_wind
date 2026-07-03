""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
sys.path.append('/Users/paolamartire/shocks')
# import resource
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = True
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

import warnings
warnings.filterwarnings('ignore')
import healpy as hp
import numpy as np
from Utilities.operators import sort_list, choose_observers
import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
import src.orbits as orb

# Choose parameters -----------------------------------------------------------------
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
stat = 'PDF'
choice = 'left_right_z'

NPIX = hp.nside2npix(prel.NSIDE)
observers_xyz = hp.pix2vec(prel.NSIDE, range(NPIX))
observers_xyz = np.array(observers_xyz)
indeces_obs, label_obs, color_obs, _ = choose_observers(observers_xyz, choice)

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
wanted_snaps = [76, 109, 151]
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps_lum, tfb_lum, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps_lum, Lum, tfb_lum = sort_list([snaps_lum, Lum, tfb_lum], tfb_lum, unique=True) 
snaps_lum = snaps_lum.astype(int)

fig, ax = plt.subplots(1,3,figsize=(24, 8))
idx_dist = 0
fig_r, ax_r = plt.subplots(1,1,figsize=(10, 8))
fig_t, ax_t = plt.subplots(1,1,figsize=(10, 8))
ratios = []
median_ratios = np.zeros(len(snaps_lum))
for s, snap in enumerate(snaps_lum):
    try:
        photo = np.load(f'{abspath}/data/{folder}/photoNEW/{check}_photo{snap}.npz')
    except FileNotFoundError:
        continue
    los, los_scatt = photo['los'], photo['los_scatt']
    ratio_taus = los_scatt/los
    median_ratios[s] = np.median(ratio_taus)
    ratios.append(np.median(ratio_taus[indeces_obs], axis=1))

    if snap in wanted_snaps:
        time = tfb_lum[np.argmin(np.abs(snaps_lum - snap))]
        if stat == 'PDF':
            # ax[s].hist(los, bins=30, color='navy', alpha=0.7, label = r'$\tau$')
            ax[idx_dist].hist(los_scatt, bins=30, color='firebrick', alpha=0.7, label = r'$\tau_{\rm scatt}$')
            ax_r.hist(ratio_taus, bins=30,  alpha=0.7 - idx_dist*0.1, label = f't = {time:.2f}' + r't$_{\rm fb}$') 
        if stat == 'CDF':
            print('Better not to use CDF')
            taus = list(np.sort(los))
            bin_taus = list(np.arange(len(taus)) / len(taus))
            ax[idx_dist].plot(taus, bin_taus, linewidth = 2, label = r'$\tau$') 
            taus_scatt = list(np.sort(los_scatt))
            bin_taus_scatt = list(np.arange(len(taus_scatt)) / len(taus_scatt))
            ax[idx_dist].plot(taus_scatt, bin_taus_scatt, linewidth = 2, label = r'$\tau_{\rm scatt}$') 
            ratio_taus = list(np.sort(ratio_taus))
            bin_ratio_taus = list(np.arange(len(ratio_taus)) / len(ratio_taus))
            ax_r.plot(ratio_taus, bin_ratio_taus, linewidth = 2, label = f't = {time:.2f}' + r't$_{\rm fb}$')
        idx_dist += 1

ratios = np.transpose(np.array(ratios))
for i, obs in enumerate(indeces_obs):
    if label_obs[i] == 'South pole':
        continue
    ax_t.plot(tfb_lum[median_ratios!=0], ratios[i], c = color_obs[i],  linewidth = 3, label = label_obs[i])
ax_t.plot(tfb_lum[median_ratios!=0], median_ratios[median_ratios!=0], c = 'k', ls = '--', linewidth = 2)
ax_t.set_xlabel(r't / t$_{\rm fb}$', fontsize = 30)
ax_t.set_ylabel(r'$\tau_{\rm scatt}(r_{\rm ph})/\tau(r_{\rm ph})$', fontsize = 30)
ax_t.legend(fontsize = 22)
ax_t.grid()

if stat == 'CDF':
    ax[0].set_ylabel(r'CDF', fontsize = 30)
    ax_r.set_xlabel(r'$\tau_{\rm scatt}(r_{\rm ph})/\tau(r_{\rm ph})$', fontsize = 30)
    ax_r.set_ylabel(r'CDF', fontsize = 30)
if stat == 'PDF':
    ax[0].set_ylabel(r'N', fontsize = 30)
    ax_r.set_xlabel(r'$\tau_{\rm scatt}(r_{\rm ph})/\tau(r_{\rm ph})$', fontsize = 30)
    ax_r.set_ylabel(r'N', fontsize = 30)
ax_r.legend(fontsize = 22)
ax_r.grid()
for i in np.arange(3):
    ax[i].set_xlabel(r'$\tau$', fontsize = 30)
    ax[i].legend(fontsize = 22)
    ax[i].grid()


# fig_t.savefig(f'{abspath}/Figs/2.paperWind/ratioTaus_ev.pdf', dpi=300)
# fig_r.savefig(f'{abspath}/Figs/2.paperWind/ratioTausPDF.pdf', dpi=300)