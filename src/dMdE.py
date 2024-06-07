import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utilities.operators import make_tree
import Utilities.sections as sec
import Utilities.orbits as orb
from Utilities.time_extractor import days_since_distruption
matplotlib.rcParams['figure.dpi'] = 150

#
## Parameters
#

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'Compton' # 'Compton' or 'ComptonHiRes' or 'ComptonRes20'
snap = '115'
is_tde = True

#
## Constants
#

G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

Mbh = 10**m
Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = 2 * Rt * (Mbh/mstar)**(1/3)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'TDE/{folder}{check}/{snap}'
saving_path = f'Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

#
## MAIN
#

do = True
plot = True
save = False

compare_resol = False
compare_times = False

if do:
    data = make_tree(path, snap, is_tde, energy = False)
    dim_cell = data.Vol**(1/3) # according to Elad
    tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
    Rcyl = np.sqrt(data.X**2 + data.Y**2)
    Vcyl = np.sqrt(data.VX**2 + data.VY**2)
    orbital_enegy = orb.orbital_energy(Rcyl, Vcyl, G, Mbh)

    # Normalisation
    norm = Mbh/Rt * (Mbh/Rstar)**(-1/3)
    print('Normalization for energy:', norm)

    # Section at the midplane
    midplane = np.abs(data.Z) < dim_cell
    X_mid, Y_mid, VX_mid, VY_mid, Rcyl_mid, Mass_mid, Den_mid, orbital_enegy_mid = \
        sec.make_slices([data.X, data.Y, data.VX, data.VY, Rcyl, data.Mass, data.Den, orbital_enegy], midplane)

    # Energy bins 
    OE_mid = orbital_enegy_mid / norm
    bins = np.arange(-4, 4, .1)
    # compute mass per bin
    hist, bins = np.histogram(OE_mid, bins = bins)
    mass_sum = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        idx = np.where((OE_mid > bins[i]) & (OE_mid < bins[i+1]))
        mass_sum[i] = np.sum(Mass_mid[idx])
    # dM/dE
    dm_dE = mass_sum / (np.diff(bins) * norm) # multiply by norm because we normalised the energy

    if save:
        try:
            file = open(f'data/{folder}/{np.round(tfb,1)}.txt', 'r')
            # Perform operations on the file
            file.close()
        except FileNotFoundError:
            with open(f'data/{folder}/{np.round(tfb,1)}.txt','a') as fstart:
                # if file doesn'exist
                file.write(f'# Energy bins normalised (norm = {norm}) \n')
                file.write((' '.join(map(str, bins)) + '\n'))

        with open(f'data/{folder}/dMdE.txt','a') as file:
            file.write(f'# Check {check}, snap {snap} \n')
            file.write((' '.join(map(str, dm_dE)) + '\n'))
            file.close()

    if plot:
        # plot the mass distribution with respect to the energy
        fig, ax = plt.subplots(1,2, figsize = (8,4))
        ax[0].scatter(bins[:-1], dm_dE, c = 'k', s = 20)
        ax[0].set_xlabel(r'E/$\Delta E$', fontsize = 16)
        ax[0].set_ylabel('dM/dE', fontsize = 16)
        ax[0].set_xlim(-1.5,1.5)
        ax[0].set_ylim(1e-4, 1e-2)
        ax[0].set_yscale('log')

        img = ax[1].scatter(X_mid, Y_mid, c = np.log10(Den_mid), s = .1, cmap = 'jet', vmin = -11, vmax = -6)
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\log_{10}$ Density', fontsize = 16)
        ax[1].set_xlim(-500,100)
        ax[1].set_ylim(-200,50)
        ax[1].set_xlabel(r'X [$R_\odot$]', fontsize = 18)
        ax[1].set_ylabel(r'Y [$R_\odot$]', fontsize = 18)

        plt.suptitle(f'check: {check}, ' + r't/t$_{fb}$ = ' + str(np.round(tfb,3)), fontsize = 16)
        plt.tight_layout()
        if save:
            plt.savefig(f'{saving_path}/EnM_{snap}.png')
        plt.show()

if compare_times:
    data = np.loadtxt(f'data/{folder}/dMdE.txt')
    bin_plot = data[0]
    data2 = data[1]
    data3 = data[4]
    data4 = data[5]
    
    plt.figure()
    plt.scatter(bin_plot, data2/np.diff(bin_plot), c = 'b', s = 40, label = '0.2')
    plt.scatter(bin_plot, data3/np.diff(bin_plot), c = 'k', s = 30, label = '0.3')
    plt.scatter(bin_plot, data4/np.diff(bin_plot), c = 'r', s = 10, label = '0.7')
    plt.xlabel('Energy', fontsize = 16)
    plt.ylabel('dM/dE', fontsize = 16)
    plt.yscale('log')
    plt.legend()
    if save:
        plt.savefig(f'{saving_path}/dMdE_times.png')
    plt.show()

if compare_resol:
    time_chosen = np.round(tfb,1)
    data = np.loadtxt(f'data/{folder}/dMdE.txt')
    bin_plot = data[0]
    data = data[1]
    dataHiRes = data[2]
    dataRes20 = data[3]
    
    plt.figure()
    plt.scatter(bin_plot, data/np.diff(bin_plot), c = 'b', s = 30, label = 'Compton')
    plt.scatter(bin_plot, dataHiRes/np.diff(bin_plot), c = 'k', s = 25, label = 'HiRes0')
    plt.scatter(bin_plot, dataRes20/np.diff(bin_plot), c = 'r', s = 20, label = 'Res20')
    plt.xlabel('Energy', fontsize = 16)
    plt.ylabel('dM/dE', fontsize = 16)
    plt.yscale('log')
    plt.legend()
    plt.title(r't/t$_{fb}$ = ' + str(time_chosen), fontsize = 16)
    if save:
        plt.savefig(f'{saving_path}/dMdE_time{time_chosen}.png')
    plt.show()
    
