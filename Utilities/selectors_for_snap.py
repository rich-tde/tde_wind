import sys
sys.path.append('/Users/paolamartire/shocks')

import Utilities.prelude
from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import os
from Utilities.time_extractor import days_since_distruption

def select_prefix(m, check, mstar, rstar, beta, n, compton = 'Compton'):
    Mbh = 10**m
    if alice:
        folder = f'R{rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
        prepath = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}'
    else: 
        folder = f'R{rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{check}/'
        prepath = f'/Users/paolamartire/shocks/TDE/{folder}'
    return prepath

def select_snap(m, check, mstar, rstar, beta, n, compton = 'Compton', time = False):
    pre = select_prefix(m, check, mstar, rstar, beta, n, compton)
    if alice:
        if m == 4 :
            if check == 'Low' or check == 'HiRes':
                snapshots = np.arange(80, 82)#348 + 1, step = 1)
                # select just the ones that actually exist
                snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}/snap_{snap}/snap_{snap}.h5')]
    else:
        if m == 4:
            if check == 'Low' or check == 'HiRes':
                snapshots = [100, 115, 164, 199, 216]
            if check == 'Res20':
                snapshots = [101, 117, 169]
    if time:
        days = np.zeros(len(snapshots))
        for i,snap in enumerate(snapshots):
            if alice:
                tfb = days_since_distruption(f'{pre}/snap_{snap}/snap_{snap}.h5', m, mstar, rstar, choose = 'tfb')
            else:
                tfb = days_since_distruption(f'{pre}/{snap}{snap}.h5', m, mstar, rstar, choose = 'tfb')
            days[i] = tfb
        return snapshots, days
    else:
        return snapshots
        