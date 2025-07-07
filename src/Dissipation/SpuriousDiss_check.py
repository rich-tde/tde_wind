""" Checks for the initial high dissipation"""
# %%
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    prepath = '/data1/martirep/shocks/shock_capturing'
    compute = True
    from Utilities.selectors_for_snap import select_snap
else:
    prepath = abspath
    compute = False
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.colors as colors

import numpy as np
from Utilities.sections import make_slices
import Utilities.prelude as prel
from src import orbits as orb
from Utilities.operators import make_tree

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5 # 'n1.5'
gamma = 5/3
params = [Mbh, Rstar, mstar, beta]
check = 'NewAMR' # '' or 'HiRes' or 'LowRes'
compton = 'Compton'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
Rp =  Rt / beta
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    snaps, tfb = snaps[tfb<0.3], tfb[tfb<0.30] # select only the first 30% of the fall
    
    ie = np.zeros(len(snaps))
    orb_en_pos = np.zeros(len(snaps))
    orb_en_neg = np.zeros(len(snaps))
    diss_pos = np.zeros(len(snaps))

    for i,snap in enumerate(snaps):
        print(snap, flush = True)

        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Diss_den = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Diss
        Diss = Diss_den * vol
        ie = ie_den * vol
        # cut in density NOT in radiation
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        cut = np.logical_and(den > 1e-19, Rsph > 0.2*apo) # cut in density and in radius
        R_cut, VX_cut, VY_cut, VZ_cut, mass_cut, den_cut, ie_cut, diss_cut = \
            make_slices([Rsph, VX, VY, VZ, mass, den, ie, Diss], cut)
        vel_cut = np.sqrt(np.power(VX_cut, 2) + np.power(VY_cut, 2) + np.power(VZ_cut, 2))
        orb_en_cut = orb.orbital_energy(R_cut, vel_cut, mass_cut, prel.G, prel.csol_cgs, Mbh, R0)

        # total energies with only the cut in density (not in radiation)
        ie[i] = np.sum(ie_cut)
        orb_en_pos[i] = np.sum(orb_en_cut[orb_en_cut > 0])
        orb_en_neg[i] = np.sum(orb_en_cut[orb_en_cut < 0])
        diss_pos[i] = np.sum(diss_cut[diss_cut > 0])

    np.save(f'{abspath}/data/{folder}/spuriousDiss_{check}.npy', [ie, orb_en_pos, orb_en_neg])
    with open(f'{abspath}/data/{folder}/spuriousDiss_{check}_days.txt', 'w') as file:
            file.write(f'# In spuriousDiss_{check}, you will find internal. orbital+, orbital-, diss+ energy, all cut in density and R>0.2*apo.')
            file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()

else:
    snap = 106
    path = f'{abspath}/TDE/{folder}/{snap}' 
    data = make_tree(path, snap, energy = True)
    cut = data.Den > 1e-19
    X, Y, Z, mass, den, Vol, Mass, Press, Temp, vx, vy, vz, IE_den, Diss_den = \
        make_slices([data.X, data.Y, data.Z, data.Mass, data.Den, data.Vol, data.Mass, data.Press, data.Temp, data.VX, data.VY, data.VZ, data.IE, data.Diss], cut)
    dim_cell = Vol**(1/3) 
    Diss = Diss_den * Vol
    IE_spec = IE_den / den

    #%%
    ln_T = np.loadtxt(f'Tfile.txt') 
    ln_Rho = np.loadtxt(f'density.txt') 
    ln_U = np.loadtxt(f'Ufile.txt')
    print(np.shape(ln_T), np.shape(ln_Rho), np.shape(ln_U))

    plt.figure(figsize=(12, 12))
    img = plt.pcolormesh(ln_T, ln_Rho, ln_U.T, cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
    cbar = plt.colorbar(img, label=r'$\ln U$')



