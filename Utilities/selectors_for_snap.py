import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import os
from Utilities.time_extractor import days_since_distruption

def select_prefix(m, check, mstar, rstar, beta, n, compton):
    Mbh = 10**m
    if m == 6:
        folder = f'R{rstar}M{mstar}BH1e+0{m}beta{beta}S60n{n}{compton}{check}'
    else:
        folder = f'R{rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    if alice:
        prepath = f'/home/martirep/data_pi-rossiem/TDE_data'
    else: 
        prepath = f'/Users/paolamartire/shocks/TDE'

    if check not in ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']:
        prepath = f'{prepath}/opacity_tests/{folder}'
    else:
        prepath = f'{prepath}/{folder}'

    return prepath

def select_snap(m, check, mstar, rstar, beta, n, compton = 'Compton',  time = False):
    pre = select_prefix(m, check, mstar, rstar, beta, n, compton)
    snapshots = np.arange(20, 348 + 1, step = 1) 
    # select just the ones that actually exist
    if alice:
        snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}/snap_{snap}/snap_{snap}.h5') or os.path.exists(f'{pre}/snap_{snap}/CMx_{snap}.npy')]
        # in case you've deleted some .h5 files
        # if not [os.path.exists(f'{pre}/snap_{snap}/snap_{snap}.h5') for snap in snapshots]:
        #     time = False
    else:
        snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}/{snap}/snap_{snap}.h5')]
    if time:
        days = np.zeros(len(snapshots))
        for i,snap in enumerate(snapshots):
            if alice:
                tfb = np.loadtxt(f'{pre}/snap_{snap}/tfb_{snap}.txt')
            else:
                tfb = np.loadtxt(f'{pre}/{snap}/tfb_{snap}.txt')
            days[i] = tfb
        return snapshots, days
    else:
        return snapshots
        