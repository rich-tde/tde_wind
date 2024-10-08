""" Make slices of the orbital plane.
If alice: make the section (with density cut except for radiation) and save it.
If not alice: load the section and plot it.
"""
abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import numpy as np
from Utilities.selectors_for_snap import select_snap
import src.orbits as orb
import Utilities.prelude as prel
import Utilities.sections as sec
from Utilities.isalice import isalice
alice, plot = isalice()
from Utilities.operators import make_tree


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
compton = 'Compton'
check = 'HiRes' # 'Low' or 'HiRes' 
difference = False
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)
z_chosen = 0

if alice:
    # get ready to slice and save
    do = True
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, time = True) 
    with open(f'/data1/martirep/shocks/shock_capturing/data/{folder}/{check}/slices/z{z_chosen}_time.txt','w') as file:
        file.write('#Snap \n') 
        file.write(' '.join(map(str, snaps)) + '\n')
        file.write('#Time \n')
        file.write(' '.join(map(str, tfb)) + '\n')
        file.close()
else:
    # get ready to plot
    do = False
    time = np.loadtxt(f'{abspath}data/{folder}/{check}/slices/z{z_chosen}_time.txt')
    snaps = time[0]
    snaps = [int(snap) for snap in snaps]
    tfb = time[1]

for idx, snap in enumerate(snaps):
    if do:
        # you are in alice
        if check == 'Low':
            check = ''
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}{check}/snap_{snap}'

        data = make_tree(path, snap, energy = True)
        indices = np.arange(len(data.X))
        dim_cell = (data.Vol)**(1/3)
        
        # Z slice
        cut = np.abs(data.Z-z_chosen) < dim_cell
        indices_mid= indices[cut]
        
        # save
        if check == '':
            check = 'Low'
        savepath = f'/data1/martirep/shocks/shock_capturing'

        np.save(f'{savepath}/data/{folder}/{check}/slices/indicesZ{z_chosen}_{snap}.npy', indices_mid)
        
    else:
        # you are not in alice
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        if not difference:
            # choose what to plot
            choice = 'rad'

            # load the data
            datacut = np.load(f'{abspath}data/{folder}/{check}/slices/midplaneIEorb_{snap}.npy')
            x_cut_mid, y_cut_mid, ie_onmass_cut_mid, orb_en_onmass_cut_mid, den_cut_mid =\
                datacut[0], datacut[1], datacut[2], datacut[3], datacut[4]
            data = np.load(f'{abspath}data/{folder}/{check}/slices/midplaneRad_{snap}.npy')
            x_mid, y_mid, Rad_den_mid = data[0], data[1], data[2]
            
            if choice == 'rad':
                coloring = Rad_den_mid * prel.en_den_converter
                x_arr = x_mid
                y_arr = y_mid
            else:
                x_arr = x_cut_mid
                y_arr = y_cut_mid
                if choice == 'IE':
                    coloring = ie_onmass_cut_mid * prel.en_converter / prel.Msol_to_g
                elif choice == 'orb':
                    coloring = np.abs(orb_en_onmass_cut_mid) * prel.en_converter / prel.Msol_to_g
                elif choice == 'den':
                    coloring = den_cut_mid * prel.Msol_to_g / prel.Rsol_to_cm**3

            fig, ax = plt.subplots(1,1, figsize = (14,5))
            img = ax.scatter(x_arr/apo, y_arr/apo, c = coloring, cmap = 'viridis', s= .1, \
                            norm = colors.LogNorm(vmin = np.percentile(coloring, 5), vmax = np.percentile(coloring, 95)))
            cb = plt.colorbar(img)
            ax.set_xlabel(r'$X/R_a$', fontsize = 20)
            ax.set_ylabel(r'$Y/R_a$', fontsize = 20)
            ax.set_xlim(-1.2, 0.1)#(-340,25)
            ax.set_ylim(-0.3, 0.3)#(-70,70)
            ax.text(-400/apo, -80/apo, f't = {np.round(tfb[idx], 2)}' + r'$t_{fb}$', fontsize = 20)
            plt.tight_layout()

            if choice == 'IE':
                cb.set_label(r'Specific IE [erg/g]', fontsize = 16)
                plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneIE_{snap}.png')
            elif choice == 'orb':
                cb.set_label(r'Absolute specific orbital energy [erg/g]', fontsize = 16)
                plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneorb_{snap}.png')
            elif choice == 'rad':
                cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 16)
                plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneRad_{snap}.png')
            elif choice == 'den':
                cb.set_label(r'Density [g/$cm^3$]', fontsize = 16)  
                ax.text(-400/apo, -120/apo, f'snap {int(snaps[idx])}', fontsize = 16)
                plt.savefig(f'{abspath}Figs/{folder}/{check}/slices/midplaneDen_{snap}.png')
            
            # plt.show()
            plt.close()

        
        else:
            # load the data
            choice = 'nofluff'

            if choice == 'nofluff':
                dataL = np.load(f'{abspath}data/{folder}/Low/slices/midplaneIEorb_{snap}.npy')
                dataH = np.load(f'{abspath}data/{folder}/HiRes/slices/midplaneIEorb_{snap}.npy')
                x_mid, y_mid, Rad_den_mid = dataL[0], dataL[1], dataL[5]
                x_midH, y_midH, Rad_den_midH = dataH[0], dataH[1], dataH[5]
                coloring = Rad_den_mid - Rad_den_midH
            else:
                dataL = np.load(f'{abspath}data/{folder}/Low/slices/midplaneRad_{snap}.npy')
                dataH = np.load(f'{abspath}data/{folder}/HiRes/slices/midplaneRad_{snap}.npy')
                x_mid, y_mid, Rad_den_mid = data[0], data[1], data[2]
                x_midH, y_midH, Rad_den_midH = dataH[0], dataH[1], dataH[2]
            

            coloring *= prel.en_den_converter