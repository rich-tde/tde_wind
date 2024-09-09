#%%
import numpy as np
from Utilities.selectors_for_snap import select_snap
import src.orbits as orb
import Utilities.sections as sec
from Utilities.isalice import isalice
alice, plot = isalice()
from Utilities.operators import make_tree

abspath = '/Users/paolamartire/shocks/'

#
## CONSTANTS
#
G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

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

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 

do = False


for snap in snaps:
    if do:
        if alice:
            if check == 'Low':
                check = ''
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'

        data = make_tree(path, snap, energy = True)
        Rsph = np.sqrt(np.power(data.X, 2) + np.power(data.Y, 2) + np.power(data.Z, 2))
        vel = np.sqrt(np.power(data.VX, 2) + np.power(data.VY, 2) + np.power(data.VZ, 2))
        mass, vol, ie_den, Rad_den = data.Mass, data.Vol, data.IE, data.Rad
        orb_en = orb.orbital_energy(Rsph, vel, mass, G, c, Mbh)
        dim_cell = (3/(4*np.pi) * vol)**(1/3)
        ie_onmass = ie_den / data.Den
        orb_en_onmass = orb_en / mass

        cut = data.Den > 1e-9 # throw fluff
        x_cut, y_cut, z_cut, dim_cut, ie_onmass_cut, orb_en_onmass_cut = \
            sec.make_slices([data.X, data.Y, data.Z, dim_cell, ie_onmass, orb_en_onmass], cut)
        midplane_cut = np.abs(z_cut) < dim_cut
        x_cut_mid, y_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid = \
            sec.make_slices([x_cut, y_cut, ie_onmass_cut, orb_en_onmass_cut], midplane_cut)

        midplane = np.abs(data.Z) < dim_cell
        x_mid, y_mid, Rad_den_mid = sec.make_slices([data.X, data.Y, Rad_den], midplane)
        
        if alice:
            if check == '':
                check = 'Low'
            savepath = f'/data1/martirep/shocks/shock_capturing'
        else:
            savepath = abspath

        np.save(f'{savepath}/data/{folder}/{check}/slices/midplaneIEorb_{snap}.npy',\
                 [x_cut_mid, y_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid])
        np.save(f'{savepath}/data/{folder}/{check}/slices/midplaneRad_{snap}.npy',\
                [x_mid, y_mid, Rad_den_mid])
        
        