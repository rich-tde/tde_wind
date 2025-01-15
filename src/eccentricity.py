"""Eccentricity (squared) from angular momentum.
If alice: compute the time-space plot using tree.
If not: or plot the time-space plot 
        or plot the relative difference between in resolutions (both space-time plots ans line plots of the median of the errors)."""
import sys
sys.path.append('/Users/paolamartire/shocks')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'

import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude
import matplotlib.colors as colors
from Utilities.operators import make_tree, single_branch
import Utilities.sections as sec
import src.orbits as orb
from Utilities.selectors_for_snap import select_snap


#
## CONSTANTS
#
G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

#
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # '' or 'LowRes' or 'HiRes' 
save = True

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Rs = 2*G*Mbh / c**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

def specific_j(r, vel):
    """ (Magnitude of) specific angular momentum """
    j = np.cross(r, vel)
    magnitude_j = np.linalg.norm(j, axis = 1)
    return magnitude_j

def eccentricity_squared(r, vel, specOE, Mbh, G):
    j = specific_j(r, vel)
    ecc2 = 1 + 2 * specOE * j**2 / (G * Mbh)**2
    return ecc2

if alice: 
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    col_ecc2 = []
    # col_Rsph = []
    radii = np.logspace(np.log10(R0), np.log10(apo),
                    num=200) 
    for i,snap in enumerate(snaps):
        print(snap)
        if alice:
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
        data = make_tree(path, snap, energy = True)
        R_vec = np.transpose(np.array([data.X, data.Y, data.Z]))
        vel_vec = np.transpose(np.array([data.VX, data.VY, data.VZ]))
        Rsph = np.linalg.norm(R_vec, axis = 1)
        vel = np.linalg.norm(vel_vec, axis=1)
        orb_en = orb.orbital_energy(Rsph, vel, data.Mass, G, c, Mbh)
        spec_orb_en = orb_en / data.Mass
        ecc2 = eccentricity_squared(R_vec, vel_vec, spec_orb_en, Mbh, G)

        # throw fluff and unbound material
        cut = np.logical_and(data.Den > 1e-19, orb_en < 0)
        Rsph_cut, mass_cut, ecc_cut = sec.make_slices([Rsph, data.Mass, ecc2], cut)
        ecc_cast = single_branch(radii, Rsph_cut, ecc_cut, weights = mass_cut)

        col_ecc2.append(ecc_cast)

    if save:
        np.save(f'{abspath}/data/{folder}/Ecc2_{check}.npy', col_ecc2)
        with open(f'{abspath}/data/{folder}/Ecc_{check}_days.txt', 'w') as file:
            file.write(f'# {folder}_{check} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()
        np.save(f'{abspath}/data/{folder}/radiiEcc_{check}.npy', radii)

else:
    difference = False
    ecc_crit = orb.eccentricity(Rstar, mstar, Mbh, beta)

    if not difference:
        path = f'{abspath}/data/{folder}'
        ecc2 = np.load(f'{path}/Ecc2_{check}.npy') 
        ecc = np.sqrt(ecc2)
        ecc = np.sqrt(ecc2)
        tfb_data = np.loadtxt(f'{path}/Ecc_{check}_days.txt')
        snap_, tfb = tfb_data[0], tfb_data[1]
        radii = np.load(f'{path}/radiiEcc_{check}.npy')
        
        # Plot
        plt.figure(figsize=(10,8))
        img = plt.pcolormesh(radii/apo, tfb, ecc, vmin = 0.58, vmax = 1, cmap = 'viridis')#cmocean.cm.balance)
        cb = plt.colorbar(img)
        cb.ax.tick_params(labelsize=25)
        cb.set_label(r'Eccentricity', fontsize = 25, labelpad = 1)
        plt.axvline(x=Rt/apo, color = 'k', linestyle = 'dashed')
        plt.xscale('log')
        plt.xlabel(r'$R [R_{a}]$', fontsize = 25)
        plt.ylabel(r'$t [t_{fb}]$', fontsize = 25)
        plt.text(0.5, 1.6, r'e$_{mb}$ = ' + f'{np.round(ecc_crit,2)}', fontsize = 20, color = 'k')
        # Bigger ticks:
        # Get the existing ticks on the x-axis
        # original_ticks = plt.yticks()[0]
        # Calculate midpoints between each pair of ticks
        # midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        # Combine the original ticks and midpoints
        # new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        # Set tick labels: empty labels for midpoints
        # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
        # plt.yticks(new_ticks, labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks])
        plt.tick_params(axis='x', which='major', width=1.2, length=8, color = 'white', labelsize=25)
        plt.tick_params(axis='y', which='major', width=1, length=8, color = 'white', labelsize=25)
        plt.tick_params(axis='x', which='minor', width=1, length=5, color = 'white', labelsize=25)
        plt.ylim(np.min(tfb), np.max(tfb))
        plt.savefig(f'{abspath}/Figs/{folder}/ecc.png')
    
    else:
        import matplotlib.gridspec as gridspec
        folderL = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes'
        pathL = f'{abspath}/data/{folderL}'
        ecc2L = np.load(f'{pathL}/Ecc2_LowRes.npy') 
        eccL = np.sqrt(ecc2L)
        tfb_dataL = np.loadtxt(f'{pathL}/Ecc_LowRes_days.txt')
        snapL, tfbL = tfb_dataL[0], tfb_dataL[1]
        radii = np.load(f'{pathL}/radiiEcc_LowRes.npy')

        folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
        path = f'{abspath}/data/{folder}'
        ecc2 = np.load(f'{path}/Ecc2_.npy') 
        ecc = np.sqrt(ecc2)
        tfb_data = np.loadtxt(f'{path}/Ecc__days.txt')
        snap, tfb = tfb_data[0], tfb_data[1]

        folderH = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes'
        pathH = f'{abspath}/data/{folderH}'
        ecc2H = np.load(f'{pathH}/Ecc2_HiRes.npy') 
        eccH = np.sqrt(ecc2H)
        tfb_dataH = np.loadtxt(f'{pathH}/Ecc_HiRes_days.txt')
        snapH, tfbH = tfb_dataH[0], tfb_dataH[1]

        # relative difference L and middle
        rel_diffL = []
        medianL = np.zeros(len(eccL))
        for i in range(len(eccL)):
            time = tfbL[i]
            idx = np.argmin(np.abs(tfb - time))
            rel_diff_time_i = 2*np.abs(eccL[i] - ecc[idx]) / (eccL[i]+ecc[idx])
            rel_diffL.append(rel_diff_time_i)
            medianL[i] = np.median(rel_diff_time_i)
        rel_diffH = []
        medianH = np.zeros(len(eccH))
        for i in range(len(eccH)):
            time = tfbH[i]
            idx = np.argmin(np.abs(tfb - time))
            rel_diff_time_i = 2*np.abs(eccH[i] - ecc[idx]) / (eccH[i]+ecc[idx])
            rel_diffH.append(rel_diff_time_i)
            medianH[i] = np.median(rel_diff_time_i)

        #%% Plot
        fig = plt.figure(figsize=(25, 9))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1.5], height_ratios=[3, 0.5], hspace=0.5, wspace = 0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # First plot
        ax2 = fig.add_subplot(gs[0, 1])  # Second plot
        ax3 = fig.add_subplot(gs[0, 2])  # Third plot

        img = ax1.pcolormesh(radii/apo, tfbL, rel_diffL, cmap = 'inferno', norm = colors.LogNorm(vmin = 1e-3, vmax=1e-1))
        ax1.set_xscale('log')
        ax1.text(0.35, 0.9*np.max(tfbL), 'Fid-Low', fontsize = 22, color = 'white')
        ax1.set_ylabel(r'$t [t_{fb}]$', fontsize = 25)

        img = ax2.pcolormesh(radii/apo, tfbH, rel_diffH, cmap = 'inferno', norm = colors.LogNorm(vmin = 1e-3, vmax=1e-1))
        ax2.set_xscale('log')
        ax2.text(0.35, 0.9*np.max(tfbH), 'Fid-High', fontsize = 22, color = 'white')

        # Create a colorbar that spans the first two subplots
        cbar_ax = fig.add_subplot(gs[1, 0:2])  # Colorbar subplot below the first two
        cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cb.set_label(r'$\Delta_{\rm rel}$ eccentricity', fontsize = 25)
        cb.ax.tick_params(labelsize=25)

        ax3.plot(tfbL, medianL, c = 'orange', linestyle = '--', label = 'Low and Middle')
        ax3.plot(tfbH, medianH, c = 'darkviolet', label = 'Middle and High')
        ax3.set_yscale('log')
        ax3.legend(fontsize = 20)
        ax3.set_ylabel(r'$\Delta_{\rm rel}$ eccentricity', fontsize = 25)
        ax3.set_xlim(0.2, tfbL[-1])
        ax3.set_ylim(3e-4, 0.1)
        ax3.grid()

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel(r'$\rm t [t_{fb}]$', fontsize = 22)
            ax.tick_params(labelsize=25)
            if ax!=ax3:
                ax.axvline(x=Rt/apo, color = 'k', linestyle = 'dashed')
                ax.tick_params(axis='x', which='major', width=1.4, length=10, color = 'white',)
                ax.tick_params(axis='x', which='minor', width=1.2, length=5, color = 'white',)
                ax.tick_params(axis='y', which='both', width=1.2, length=8, color = 'k')
        
        plt.savefig(f'{abspath}/Figs/multiple/ecc_diff.pdf')
        plt.savefig(f'{abspath}/Figs/multiple/ecc_diff.png')

# %%
