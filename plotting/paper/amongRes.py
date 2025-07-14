""" Compare quantities with respect to the resolution of the simulation """
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
abspath = '/Users/paolamartire/shocks'
import csv
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
import matplotlib.colors as colors
import src.orbits as orb

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
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
last_time = 1.2157121487183973 #last time available from HighRes
Rt = Rstar * (Mbh/mstar)**(1/3)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, beta)
commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
checks = ['LowRes', '', 'HiRes']
labels_check = ['Low', 'Fid', 'Hi']
colors_checks = ['C1', 'yellowgreen', 'darkviolet']
Nres = 10**4 * np.sqrt(Mbh) * np.array([0.25,1,4])
observables_marker = ['o', 'x', 's', 'D']

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i, check in enumerate(checks):
    data = np.loadtxt(f'{abspath}/data/{commonfold}{check}/{check}_red.csv', delimiter=',', dtype=float)
    snap, Lum = data[:, 0], data[:, 2]   
    snap, Lum = snap[~np.isnan(Lum)], Lum[~np.isnan(Lum)]
    idx_maxLum = np.argmax(Lum)
    maxLum = Lum[idx_maxLum]
    snap_maxLum = int(snap[idx_maxLum])

    # Rph
    dataph = np.loadtxt(f'{abspath}/data/{commonfold}{check}/photo_stat{check}.txt')
    tph, medianph = dataph[0], dataph[1]
    last_medianph = medianph[np.argmin(np.abs(tph - last_time))]

    # ecc
    ecc2 = np.load(f'{abspath}/data/{commonfold}{check}/Ecc2_{check}.npy') 
    ecc = np.sqrt(ecc2)
    tfb_dataecc = np.loadtxt(f'{abspath}/data/{commonfold}{check}/Ecc_{check}_days.txt')
    snapecc, tfb_ecc = tfb_dataecc[0], tfb_dataecc[1]
    radii = np.load(f'{abspath}/data/{commonfold}{check}/radiiEcc_{check}.npy')
    ecc_mb = ecc[np.argmin(np.abs(radii - a_mb))][np.argmin(np.abs(tfb_ecc - last_time))]
    
    ax.scatter(Nres[i]/1e6, ecc_mb, color = colors_checks[i], label= r'$e(a_{\rm mb})/e_{\rm mb}$' if i == 1 else None, marker=observables_marker[0], s = 60)
    # ax.scatter(Nres[i]/1e6, dispersion/maxLum, color = colors_checks[i], label= r'$\sigma(L_{\rm obs})/L_{\rm peak}$' if i == 1 else None, marker=observables_marker[1], s = 60)
    ax.scatter(Nres[i]/1e6, last_medianph/apo, color = colors_checks[i], label= r'Med $R_{\rm ph}/R_{\rm a}$' if i == 1 else None, marker=observables_marker[2], s = 60)
    ax.scatter(Nres[i]/1e6, maxLum/Ledd, color = colors_checks[i], label= r'$L_{\rm peak}/L_{\rm Edd}$' if i == 1 else None, marker=observables_marker[3], s = 60)

ax.set_xlabel(r'$N_{\rm initial} [10^6$ cells]')
# ax.set_ylabel(r'simulation/theory')
ax.legend(fontsize = 18)
plt.tight_layout()
plt.savefig(f'{abspath}/Figs/paper/amongRes.pdf', bbox_inches='tight')

#%% find the dispersion of L
