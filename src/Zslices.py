#%%
import numpy as np
from Utilities.selectors_for_snap import select_snap
import pickle
from Utilities.isalice import isalice
alice, plot = isalice()
from Utilities.operators import make_tree

abspath = '/Users/paolamartire/shocks/'

##
# PARAMETERS
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
compton = 'Compton'
check = 'Low' # '' or 'HiRes' or 'Res20'
cutoff = 'cutden' # or '' or 'bound' or 'cutdenbound'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 

for snap in snaps:
    if alice:
        if check == 'Low':
            check = ''
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
    else:
        path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'

    data = make_tree(path, snap, energy = True)
    z_coord, dim_cell = data.Z, data.Vol**(1/3) 

    if cutoff == 'cutded':
        print('Cutting off low density elements')
        cut = data.Den > 1e-9 # throw fluff
        z_coord, dim_cell = z_coord[cut], dim_cell[cut]

    midplane = np.abs(z_coord) < dim_cell

    if alice:
        if check == '':
            check = 'Low'
        savepath = f'/data1/martirep/shocks/shock_capturing'
    else:
        savepath = abspath
    with open(f'{savepath}/data/{folder}/{check}/slices/midplaneCond_{snap}.pkl', 'wb') as filebool:
        pickle.dump(midplane, filebool)
