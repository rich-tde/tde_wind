import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 109
x_axis = 'Temp'  # 'Freq' or 'Temp'

## Snapshots stuff
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre_saving = f'{abspath}/data/{folder}'

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)
time = tfb[snaps == snap][0]
freqs = np.loadtxt(f'{pre_saving}/spectra/freqs.txt')
Temp = freqs * prel.Hz_toK
F_photo = np.loadtxt(f'{pre_saving}/spectra/{check}_spectra{snap}.txt')
N_obs = F_photo.shape[0] 

observers_xyz = hp.pix2vec(prel.NSIDE, range(N_obs)) #shape: (3, 192)
observers_xyz = np.array(observers_xyz)

indices_sorted, label_obs, colors_obs, lines_obs = choose_observers(observers_xyz, choice = 'focus_axis')
# indices_sorted = np.concatenate(indices_sorted)

fig, ax = plt.subplots(1, 1, figsize=(10,6))
F_mean = []
for i, idx_list in enumerate(indices_sorted):
    F_mean.append(np.mean(F_photo[idx_list], axis=0))
print(np.shape(F_mean))


for idx, lab in enumerate(label_obs):
    if x_axis == 'Freq':
        ax.plot(freqs, freqs * F_mean[idx], c = colors_obs[idx], label = lab)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(1e14, 1e19)
    elif x_axis == 'Temp':  
        ax.plot(Temp, freqs * F_mean[idx], c = colors_obs[idx], label = lab)
        ax.set_xlabel('Temperature [K]')
        ax.set_xlim(1e3, 1e8)
ax.tick_params(axis='both', which='major', length=8, width=1.2)
ax.tick_params(axis='both', which='minor', length=5, width=1)
ax.loglog()
ax.set_ylabel(r'$\nu F_{\nu}$ [erg s$^{-1}$ cm$^{-2}$]')
ax.set_ylim(1e38, 1e42)
ax.legend()
ax.set_title(f't = {time:.2f}'+ r' t$_{\rm fb}$', fontsize=20)
plt.tight_layout()

