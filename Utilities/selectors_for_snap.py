import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import os
from Utilities.time_extractor import days_since_distruption

def select_prefix(m, check, mstar, rstar, beta, n, compton, step):
    Mbh = 10**m
    folder = f'R{rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}{step}'
    if alice:
        prepath = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}'
    else: 
        prepath = f'/Users/paolamartire/shocks/TDE/{folder}'
    return prepath

def select_snap(m, check, mstar, rstar, beta, n, compton = 'Compton', step = '', time = False):
    pre = select_prefix(m, check, mstar, rstar, beta, n, compton, step)
    if m == 4 :
        snapshots = np.arange(80, 348 + 1, step = 1)
        # select just the ones that actually exist
        if alice:
            snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}/snap_{snap}/snap_{snap}.h5')]
        else:
            snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}/{snap}/snap_{snap}.h5')]
    if time:
        days = np.zeros(len(snapshots))
        for i,snap in enumerate(snapshots):
            if alice:
                tfb = days_since_distruption(f'{pre}/snap_{snap}/snap_{snap}.h5', m, mstar, rstar, choose = 'tfb')
            else:
                tfb = days_since_distruption(f'{pre}/{snap}/snap_{snap}.h5', m, mstar, rstar, choose = 'tfb')
            days[i] = tfb
        return snapshots, days
    else:
        return snapshots
        