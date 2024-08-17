import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import Utilities.prelude
from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import os
from Utilities.time_extractor import days_since_distruption

def select_prefix(m, check, mstar, rstar, beta, n):
    if alice:
        prepath = f'/data1/martirep/shocks/shock_capturing/'
    else: 
        Mbh = 10**m
        folder = f'R{rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{check}/'
        prepath = f'/Users/paolamartire/shocks/TDE/{folder}'
    return prepath

def select_snap(m, mstar, rstar, check, time = False):
    pre = select_prefix(m, check, mstar)
    days = []
    if alice:
        if m == 4 and check == 'fid':
            snapshots = np.arange(683, 1008 + 1, step = 1)
        if m == 4 and check == 'fid':
            snapshots = [293,322] #np.arange(110, 322 + 1) 
        if m == 4 and check == 'S60ComptonHires':
            snapshots = np.arange(210, 278 + 1)
        # select just the ones that actually exist
        snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}{snap}/snap_{snap}.h5')]
    else:
        if m == 4:
            if check == 'Low' or check == 'HiRes':
                snapshots = [100, 115, 164, 199, 216]
            if check == 'Res20':
                snapshots = [101, 117, 169]
    for snap in snapshots:
        if time:
            if not alice:
                tfb = days_since_distruption(f'{pre}/{snap}.h5', m, mstar, rstar, choose = 'tfb')
            days.append(tfb)
            return snapshots, days
        else:
            return snapshots

# Select opacity
def select_opacity(m):
    if m==6:
        return 'cloudy'
    else:
        return 'LTE'