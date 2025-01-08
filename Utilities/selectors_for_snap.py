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
        prepath = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}'
    else: 
        prepath = f'/Users/paolamartire/shocks/TDE/{folder}'
    return prepath

def select_snap(m, check, mstar, rstar, beta, n, compton = 'Compton',  time = False):
    pre = select_prefix(m, check, mstar, rstar, beta, n, compton)
    # select just the ones that actually exist
    if alice:
        if m ==4: 
            snapshots = np.arange(80, 365 + 1, step = 1)
        if m== 6: 
            snapshots = np.arange(444,445)#(180, 444, step = 1) # before 180 they are not "snap_full" on Drive
        snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}/snap_{snap}/snap_{snap}.h5')]
    else:
        snapshots = [80, 164]#[snap for snap in snapshots if os.path.exists(f'{pre}/{snap}/snap_{snap}.h5')]
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
        