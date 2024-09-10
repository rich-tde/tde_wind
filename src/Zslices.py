#%%
import sys
sys.path.append('/Users/paolamartire/shocks')
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

if alice:
    do = True
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
    with open(f'/data1/martirep/shocks/shock_capturing/data/{folder}/{check}/slices/midplane_time.txt','w') as file:
        file.write('#Snap \n') 
        file.write(' '.join(map(str, snaps)) + '\n')
        file.write('#Time \n')
        file.write(' '.join(map(str, tfb)) + '\n')
        file.close()
else:
    do = False
    time = np.loadtxt(f'{abspath}data/{folder}/{check}/slices/midplane_time.txt')
    snaps = time[0]
    snaps = [int(snap) for snap in snaps]
    tfb = time[1]

for idx, snap in enumerate(snaps):
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
        x_cut, y_cut, z_cut, dim_cut, den_cut, ie_onmass_cut, orb_en_onmass_cut = \
            sec.make_slices([data.X, data.Y, data.Z, dim_cell, data.Den, ie_onmass, orb_en_onmass], cut)
        midplane_cut = np.abs(z_cut) < dim_cut
        x_cut_mid, y_cut_mid, den_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid = \
            sec.make_slices([x_cut, y_cut, den_cut, ie_onmass_cut, orb_en_onmass_cut], midplane_cut)

        midplane = np.abs(data.Z) < dim_cell
        x_mid, y_mid, Rad_den_mid = sec.make_slices([data.X, data.Y, Rad_den], midplane)
        
        if alice:
            if check == '':
                check = 'Low'
            savepath = f'/data1/martirep/shocks/shock_capturing'
        else:
            savepath = abspath

        np.save(f'{savepath}/data/{folder}/{check}/slices/midplaneIEorb_{snap}.npy',\
                 [x_cut_mid, y_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid, den_cut_mid])
        np.save(f'{savepath}/data/{folder}/{check}/slices/midplaneRad_{snap}.npy',\
                [x_mid, y_mid, Rad_den_mid])
        
    else:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        choice = 'den'

        datacut = np.load(f'{abspath}data/{folder}/{check}/slices/midplaneIEorb_{snap}.npy')
        x_cut_mid, y_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid, den_cut_mid =\
            datacut[0], datacut[1], datacut[2], datacut[3], datacut[4]
        data = np.load(f'{abspath}data/{folder}/{check}/slices/midplaneRad_{snap}.npy')
        x_mid, y_mid, Rad_den_mid = data[0], data[1], data[2]
        
        if choice == 'rad':
            coloring = Rad_den_mid
            x_arr = x_mid
            y_arr = y_mid
        else:
            x_arr = x_cut_mid
            y_arr = y_cut_mid
            if choice == 'IE':
                coloring = ie_onmass_cut_mid
            elif choice == 'orb':
                coloring = np.abs(orb_en_onmass_cut_mid)
            elif choice == 'den':
                coloring = den_cut_mid

        fig, ax = plt.subplots(1,1, figsize = (14,4))
        img = ax.scatter(x_arr, y_arr, c = coloring, cmap = 'viridis', s= .1, \
                         norm = colors.LogNorm(vmin = np.percentile(coloring, 5), vmax = np.percentile(coloring, 95)))
        cb = plt.colorbar(img)
        ax.set_xlabel(r'$X [R_\odot]$', fontsize = 20)
        ax.set_ylabel(r'$Y [R_\odot]$', fontsize = 20)
        ax.set_xlim(-340,25)
        ax.set_ylim(-70,70)
        ax.text(-335, -65, f't = {np.round(tfb[idx], 2)}' + r'$t_{fb}$', fontsize = 20)
        plt.tight_layout()
        if choice == 'IE':
            cb.set_label(r'Specific IE', fontsize = 18)
            plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneIE_{snap}.png')
        elif choice == 'orb':
            cb.set_label(r'Absolute specific orbital energy', fontsize = 18)
            plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneorb_{snap}.png')
        elif choice == 'rad':
            cb.set_label(r'Radiation energy density', fontsize = 18)
            plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneRad_{snap}.png')
        elif choice == 'den':
            cb.set_label(r'Density', fontsize = 18)  
            ax.text(-335, -52, f'snap {int(snaps[idx])}', fontsize = 18)
            plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneDen_{snap}.png')
        plt.close()
        # plt.show()
        