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
import colorcet
import pickle
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
step = ''
check = '' # '' or 'LowRes' or 'HiRes'
save = True

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}{step}'
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, step, time = True) #[100,115,164,199,216]

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

if __name__ == '__main__':
    if alice: 
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
            ecc_cast = single_branch(radii,'radii', Rsph_cut, ecc_cut, weights = mass_cut)

            col_ecc2.append(ecc_cast)

        if save:
            np.save(f'{abspath}/data/{folder}/Ecc2_{check}{step}.npy', col_ecc2)
            with open(f'{abspath}/data/{folder}/Ecc_{check}{step}_days.txt', 'w') as file:
                file.write(f'# {folder}_{check}{step} \n' + ' '.join(map(str, snaps)) + '\n')
                file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
                file.close()
            np.save(f'{abspath}/data/{folder}/radiiEcc_{check}{step}.npy', radii)
    
    else:
        difference = False

        if not difference:
            path = f'{abspath}/data/{folder}'
            ecc2 = np.load(f'{path}/Ecc2_{check}{step}.npy') 
            ecc = np.sqrt(ecc2)
            ecc = np.sqrt(ecc2)
            tfb_data = np.loadtxt(f'{path}/Ecc_{check}{step}_days.txt')
            snap_, tfb = tfb_data[0], tfb_data[1]
            radii = np.load(f'{path}/radiiEcc2_{check}{step}.npy')
            
            # Plot
            img = plt.pcolormesh(radii/apo, tfb, ecc, vmin = 0.6, vmax = 1, cmap = 'cet_rainbow4')
            cb = plt.colorbar(img)
            plt.xscale('log')
            cb.set_label(r'Eccentricity', fontsize = 20, labelpad = 2)
            plt.xlabel(r'$R/R_{a}$', fontsize = 20)
            plt.ylabel(r'$t/t_{fb}$', fontsize = 20)
            # Bigger ticks:
            # Get the existing ticks on the x-axis
            original_ticks = plt.yticks()[0]
            # Calculate midpoints between each pair of ticks
            midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
            # Combine the original ticks and midpoints
            new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
            # Set tick labels: empty labels for midpoints
            labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
            plt.yticks(new_ticks, labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks])
            plt.tick_params(axis='x', which='major', width=1, length=8)
            plt.tick_params(axis='y', which='major', width=1, length=8, color = 'white')
            plt.tick_params(axis='x', which='minor', width=1, length=5)
            plt.tick_params(axis='y', which='minor', width=1, length=5, color = 'white')
            plt.ylim(np.min(tfb), np.max(tfb))
            plt.savefig(f'{abspath}/Figs/{folder}/ecc.pdf')
       
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
            fig = plt.figure(figsize=(20, 6))
            gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1.5], height_ratios=[3, 0.5], hspace=0.4, wspace = 0.3)
            ax1 = fig.add_subplot(gs[0, 0])  # First plot
            ax2 = fig.add_subplot(gs[0, 1])  # Second plot
            ax3 = fig.add_subplot(gs[0, 2])  # Third plot

            img = ax1.pcolormesh(radii/apo, tfbL, rel_diffL, cmap = 'plasma', norm = colors.LogNorm(vmin = 1e-3, vmax=1e-1))
            ax1.set_xscale('log')
            ax1.set_title('Low and Middle', fontsize = 20)

            img = ax2.pcolormesh(radii/apo, tfbH, rel_diffH, cmap = 'plasma', norm = colors.LogNorm(vmin = 1e-3, vmax=1e-1))
            ax2.set_xscale('log')
            ax2.set_title('Middle and High', fontsize = 20)

            # Create a colorbar that spans the first two subplots
            cbar_ax = fig.add_subplot(gs[1, 0:2])  # Colorbar subplot below the first two
            cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
            cb.set_label(r'Relative difference', fontsize = 20)

            ax3.plot(tfbL, medianL, c = 'dodgerblue', label = 'Low and Middle')
            ax3.plot(tfbH, medianH, c = 'coral', label = 'Middle and High')
            ax3.set_yscale('log')
            ax3.legend(fontsize = 20)
            ax3.set_ylabel('Relative difference', fontsize = 20)
            ax3.set_xlim(0.2, tfbL[-1])
            ax3.set_ylim(1e-3, 0.1)
            ax3.grid()

            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel(r'$R/R_{a}$', fontsize = 20)
                if ax!=ax3:
                    ax.set_ylabel(r'$t/t_{fb}$', fontsize = 20)
            
            plt.savefig(f'{abspath}/Figs/multiple/ecc_diff.pdf')

# %%
