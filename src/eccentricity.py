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
from Utilities.operators import make_tree, single_branch, find_ratio
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
check = 'OpacityNew' # '' or 'LowRes' or 'HiRes' 
save = False
which_cut = '' #if 'high' cut density at 1e-12, if '' cut density at 1e-19

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

Rs = 2*G*Mbh / c**2
Rg = Rs/2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
Rstart = 0.4 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

if alice: 
    snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) #[100,115,164,199,216]
    col_ecc2 = []
    # col_Rsph = []
    radii = np.logspace(np.log10(Rstart), np.log10(apo),
                    num=200) 
    tfb = np.zeros(len(snaps))
    for i,snap in enumerate(snaps):
        print(snap)
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        tfb[i] = np.loadtxt(f'{path}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        R_vec = np.transpose(np.array([data.X, data.Y, data.Z]))
        vel_vec = np.transpose(np.array([data.VX, data.VY, data.VZ]))
        Rsph = np.linalg.norm(R_vec, axis = 1)
        vel = np.linalg.norm(vel_vec, axis=1)
        orb_en = orb.orbital_energy(Rsph, vel, data.Mass, G, c, Mbh, R0)
        spec_orb_en = orb_en / data.Mass
        ecc2 = orb.eccentricity_squared(R_vec, vel_vec, spec_orb_en, Mbh, G)

        # throw fluff and unbound material
        if which_cut == 'high':
            cut = np.logical_and(data.Den > 1e-12, orb_en < 0)
        else:
            cut = np.logical_and(data.Den > 1e-19, orb_en < 0)
        Rsph_cut, mass_cut, ecc_cut = sec.make_slices([Rsph, data.Mass, ecc2], cut)
        ecc_cast = single_branch(radii, Rsph_cut, ecc_cut, weights = mass_cut)

        col_ecc2.append(ecc_cast)

    np.save(f'{abspath}/data/{folder}/Ecc2_{which_cut}_{check}.npy', col_ecc2)
    with open(f'{abspath}/data/{folder}/Ecc_{which_cut}_{check}_days.txt', 'w') as file:
        file.write(f'# {folder}_{check} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()
    np.save(f'{abspath}/data/{folder}/radiiEcc_{which_cut}_{check}.npy', radii)

else:
    ecc_slice = False
    space_time = False
    error = True
    ecc_crit = orb.e_mb(Rstar, mstar, Mbh, beta)

    if ecc_slice:
        radii_grid = [R0, Rt] 
        styles = ['dotted', 'dashed']
        xcfr_grid, ycfr_grid, cfr_grid = [], [], []
        for i, radius_grid in enumerate(radii_grid):
            xcr, ycr, cr = orb.make_cfr(radius_grid)
            xcfr_grid.append(xcr)
            ycfr_grid.append(ycr)
            cfr_grid.append(cr)
        snap = 348
        path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
        data = make_tree(path, snap, energy = True)
        X, Y, Z, VX, VY, VZ, Den, Mass, Vol = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Den, data.Mass, data.Vol
        mid = np.abs(Z) < Vol**(1/3)
        X, Y, Z, VX, VY, VZ, Den, Mass = \
            sec.make_slices([X, Y, Z, VX, VY, VZ, Den, Mass], mid)
        R_vec = np.transpose(np.array([X, Y, Z]))
        vel_vec = np.transpose(np.array([VX, VY, VZ]))
        Rsph = np.linalg.norm(R_vec, axis = 1)
        vel = np.linalg.norm(vel_vec, axis=1)
        orb_en = orb.orbital_energy(Rsph, vel, Mass, G, c, Mbh, R0)
        spec_orb_en = orb_en / Mass
        ecc2 = orb.eccentricity_squared(R_vec, vel_vec, spec_orb_en, Mbh, G)
        cut = np.logical_and(Den > 1e-19, orb_en < 0)
        X, Y, ecc2 = sec.make_slices([X, Y, ecc2], cut)
        ecc = np.sqrt(ecc2)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        img = ax.scatter(X/Rt, Y/Rt, c = ecc, s = 4, vmin = 0.2, vmax = .95, cmap = 'viridis', rasterized = True)
        cb = plt.colorbar(img)
        for i in range(len(radii_grid)):
            ax.contour(xcfr_grid[i]/Rt, ycfr_grid[i]/Rt, cfr_grid[i]/Rt, levels = [0], color = 'k', linestyle = styles[i], linewidth = 2)
        cb.set_label(r'Eccentricity')
        ax.set_xlabel(r'$X [R_{\rm t}]$')
        ax.set_ylabel(r'$Y [R_{\rm t}]$')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        plt.tight_layout()
        plt.savefig(f'{abspath}/Figs/{folder}/ecc_slice{snap}.png', bbox_inches='tight')

    if space_time:
        path = f'{abspath}/data/{folder}'
        ecc2 = np.load(f'{path}/Ecc2_{which_cut}_{check}.npy') 
        ecc = np.sqrt(ecc2)
        tfb_data= np.loadtxt(f'{path}/Ecc_{which_cut}_{check}_days.txt')
        snap_, tfb = tfb_data[0], tfb_data[1]
        tfb_dataFid = np.loadtxt(f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n1.5{compton}/Ecc_{which_cut}__days.txt')
        tfbFid = tfb_dataFid[1]
        radii = np.load(f'{path}/radiiEcc_{which_cut}_{check}.npy')
        
        # Plot
        plt.figure(figsize=(8,7))
        # set to white the 0 values so they are white in the plot
        ecc[ecc == 0] = np.nan
        img = plt.pcolormesh(radii/apo, tfb, ecc, vmin = 0.5, vmax = .95, cmap = 'viridis', rasterized = True)#cmocean.cm.balance)
        cb = plt.colorbar(img)
        cb.ax.tick_params(labelsize=25)
        cb.set_label(r'Eccentricity' , fontsize = 25, labelpad = 1)
        # line thicks 2
        plt.axvline(x=Rt/apo, color = 'k', linestyle = 'dashed', linewidth = 2)
        plt.text(1.05*Rt/apo, 0.9*np.max(tfb), r'$R_{\rm t}$', fontsize = 25, color = 'k')
        plt.xscale('log')
        plt.xlabel(r'$R [R_{\rm a}]$')#, fontsize = 25)
        plt.ylabel(r'$t [t_{fb}]$')#, fontsize = 25)
        plt.tick_params(axis='x', which='major', width=1.2, length=8, color = 'k', labelsize=25)
        plt.tick_params(axis='y', which='major', width=1, length=8, color = 'white', labelsize=25)
        plt.tick_params(axis='x', which='minor', width=1, length=6, color = 'k', labelsize=25)
        plt.ylim(np.min(tfbFid), np.max(tfbFid))
        plt.savefig(f'{abspath}/Figs/paper/ecc{which_cut}{check}.pdf', bbox_inches='tight')

        folderFid = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n1.5{compton}'
        pathFid = f'{abspath}/data/{folderFid}'
        ecc2Fid = np.load(f'{pathFid}/Ecc2_.npy') 
        eccFid = np.sqrt(ecc2Fid)

        rel_diff = []
        median = np.zeros(len(ecc))
        for i in range(len(ecc)):
            time = tfbFid[i]
            idx = np.argmin(np.abs(tfbFid - time))
            rel_diff_time_i =  find_ratio(eccFid[idx], ecc[i]) #
            rel_diff.append(rel_diff_time_i)
            median[i] = np.median(rel_diff_time_i)

        where_are_nans = np.isnan(rel_diff)
        where_infs = np.isinf(rel_diff)
        rel_diff = np.array(rel_diff)
        print('max error', np.max(rel_diff[np.logical_and(~where_are_nans, ~where_infs)]))
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,10))    
        img = ax1.pcolormesh(radii/apo, tfb, rel_diff, cmap = 'inferno', vmin = 0.9, vmax = 1.1, rasterized = True)
        ax1.set_xscale('log')
        ax1.set_xlabel(r'$R [R_{\rm a}]$')#, fontsize = 25)
        ax1.set_ylabel(r'$t [t_{\rm fb}]$')#, fontsize = 25)
        ax1.set_ylim(np.min(tfb), np.max(tfbFid))
        cb = plt.colorbar(img,  orientation='horizontal')
        cb.set_label(r'$\mathcal{R}$ eccentricity')#, fontsize = 25)
        cb.ax.tick_params(labelsize=25)
        cb.ax.tick_params(width=1, length=11, color = 'k',)

        ax2.plot(tfb, median, c = 'yellowgreen', linewidth = 4)
        ax2.set_ylabel(r'$\mathcal{R}$ median eccentricity')
        # ax2.set_ylim(0.98, 1.08)
        ax2.grid()
        ax2.set_xlabel(r'$t [t_{\rm fb}]$')

    if error: # compare resolutions
        import matplotlib.gridspec as gridspec
        check1 = 'OpacityNewNewAMR' #LowResNewAMR'
        path1 = f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check1}'
        ecc_quad1 = np.load(f'{path1}/Ecc2_{which_cut}_{check1}.npy') 
        ecc1 = np.sqrt(ecc_quad1)
        tfb_data1 = np.loadtxt(f'{path1}/Ecc_{which_cut}_{check1}_days.txt')
        snap1, tfb1 = tfb_data1[0], tfb_data1[1]
        radii = np.load(f'{path1}/radiiEcc_{which_cut}_{check1}.npy')

        check = 'NewAMRRemoveCenter' #'LowResNewAMRRemoveCenter'
        path = f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
        ecc_quad = np.load(f'{path}/Ecc2_{which_cut}_{check}.npy') 
        ecc = np.sqrt(ecc_quad)
        tfb_data = np.loadtxt(f'{path}/Ecc_{which_cut}_{check}_days.txt')
        snap, tfb = tfb_data[0], tfb_data[1]

        print('comparing', check1, 'and', check)

        # folderH = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes'
        # pathH = f'{abspath}/data/{folderH}'
        # ecc2H = np.load(f'{pathH}/Ecc2_HiRes.npy') 
        # eccH = np.sqrt(ecc2H)
        # tfb_dataH = np.loadtxt(f'{pathH}/Ecc_HiRes_days.txt')
        # snapH, tfbH = tfb_dataH[0], tfb_dataH[1]

        # relative difference L and middle
        rel_diffL = []
        medianL = np.zeros(len(ecc1))
        percentile16 = np.zeros(len(ecc1))
        percentile84 = np.zeros(len(ecc1))
        for i in range(len(ecc1)):
            time = tfb1[i]
            idx = np.argmin(np.abs(tfb - time))
            rel_diff_time_i = ecc[idx]/ecc1[i] #find_ratio(ecc1[i], ecc[idx]) 
            rel_diffL.append(rel_diff_time_i)
            medianL[i] = np.median(rel_diff_time_i)
            percentile16[i] = np.percentile(rel_diff_time_i, 16)
            percentile84[i] = np.percentile(rel_diff_time_i, 84)
        # rel_diffH = []
        # medianH = np.zeros(len(eccH))
        # medianFoverH = np.zeros(len(eccH))
        # for i in range(len(eccH)):
        #     time = tfbH[i]
        #     idx = np.argmin(np.abs(tfb - time))
        #     rel_diff_time_i =  find_ratio(ecc[idx], eccH[i]) #
        #     rel_diffH.append(rel_diff_time_i)
        #     medianH[i] = np.median(rel_diff_time_i)
        #     medianFoverH[i] = np.median(ecc[idx]/eccH[i])

        #%% Plot
        # fig = plt.figure(figsize=(25, 9))
        # gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1.5], height_ratios=[3, 0.5], hspace=0.5, wspace = 0.3)
        # ax1 = fig.add_subplot(gs[0, 0])  # First plot
        # ax2 = fig.add_subplot(gs[0, 1])  # Second plot
        # ax3 = fig.add_subplot(gs[0, 2])  # Third plot

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 7))
        img = ax1.pcolormesh(radii/apo, tfb1, rel_diffL, cmap = 'inferno', vmin = .9, vmax = 2, rasterized = True)
        ax1.set_xscale('log')
        # ax1.text(0.22, .88*np.max(tfb1), 'Low vs Fid', fontsize = 28, color = 'k')
        ax1.set_xlabel(r'$R [R_{\rm a}]$')
        ax1.set_ylabel(r'$t [t_{\rm fb}]$')

        # img = ax2.pcolormesh(radii/apo, tfbH, rel_diffH, cmap = 'inferno', vmin = 0.95, vmax = 1.05, rasterized = True)
        # ax2.set_xscale('log')
        # ax2.text(0.21, 0.88*np.max(tfbH), 'Fid vs High', fontsize = 28, color = 'k')
        # ax2.set_xlabel(r'$R [R_{\rm a}]$')

        # Create a colorbar that spans the first two subplots
        # cbar_ax = fig.add_subplot(gs[1, 0:2])  # Colorbar subplot below the first two
        # cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cb = plt.colorbar(img)
        cb.set_label(r'sink/no sink')#r'$\mathcal{R}$ eccentricity')
        cb.ax.tick_params(labelsize=25)
        # label with 3 decimals in the colorbar
        # cb.set_ticks([0.98, 1, 1.02])
        # cb.set_ticklabels(['0.98', '1', '1.02']) 
        cb.ax.tick_params(width=1, length=11, color = 'k',)

        ax3.plot(tfb1, medianL, linewidth = 4, c = 'darkorange')
        # ax3.plot(tfb1, medianL, c = 'yellowgreen', linewidth = 4, linestyle = (0, (5, 10)), label = 'Low and Middle')
        ax3.plot(tfb1, percentile16, c = 'grey')
        ax3.plot(tfb1, percentile84, c = 'grey')
        # ax3.plot(tfbH, medianH, c = 'yellowgreen', linewidth = 4)
        # ax3.plot(tfbH, medianH, c = 'darkviolet', linewidth = 4, linestyle = (0, (5, 10)), label = 'Middle and High')
        # ax3.text(0.4, 1, 'Fid vs High', fontsize = 27, color = 'k')
        ax3.set_ylabel(r'median sink/no sink')#$\mathcal{R}$ median eccentricity')
        original_ticks = ax3.get_xticks()
        mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
        all_ticks = np.concatenate(([original_ticks, mid_ticks]))
        ax3.set_xticks(all_ticks)
        ax3.set_xlim(0.11, tfb1[-1])
        ax3.set_ylim(0.99, 1.02)
        ax3.grid()
        ax3.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 25)

        for ax in [ax1, ax3]: #, ax2]:
            # ax.tick_params(labelsize=26)
            if ax!=ax3:
                ax.axvline(x=Rt/apo, color = 'white', linestyle = 'dashed', linewidth = 2)
                ax.axvline(x=R0/apo, color = 'white', linestyle = ':', linewidth = 2)
                ax.tick_params(axis='x', which='major', width=1.4, length=11, color = 'k',)
                ax.tick_params(axis='y', which='major', width=1.4, length=9, color = 'k',)
                ax.tick_params(axis='x', which='minor', width=1.2, length=7, color = 'k',)
        
        # plt.savefig(f'{abspath}/Figs/paper/ecc_diff.pdf', bbox_inches='tight')
        plt.tight_layout
        plt.savefig(f'{abspath}/Figs/Test/MazeOfRuns/sink/ecc_diff_Fid.png', bbox_inches='tight')

# %%
