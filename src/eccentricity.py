"""Eccentricity (squared) from angular momentum.
If alice: compute the time-space plot using tree.
If not: If time-space, plot the time-space plot of the eccentricity.
        If ecc_slice, plot the eccentricity slice at a given time.
        If error, compute the relative difference between resolutions and plot it.
"""
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
import Utilities.prelude as prel

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
check = 'HiResNewAMR' # '' or 'LowRes' or 'HiRes' 
save = False
which_cut = 'high' #if 'high' cut density at 1e-12, if '' cut density at 1e-19

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
ecc_mb = things['ecc_mb']
Rstart = 0.4 * Rt

if alice: 
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    col_ecc2 = []
    radii = np.logspace(np.log10(Rstart), np.log10(apo), num=200)  

    for i,snap in enumerate(snaps):
        print(snap, flush=True)
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        Mass = data.Mass
        R_vec = np.transpose(np.array([data.X, data.Y, data.Z]))
        vel_vec = np.transpose(np.array([data.VX, data.VY, data.VZ]))
        Rsph = np.linalg.norm(R_vec, axis = 1)
        vel = np.linalg.norm(vel_vec, axis=1)
        orb_en = orb.orbital_energy(Rsph, vel, Mass, params, prel.G)
        spec_orb_en = orb_en / Mass
        ecc2 = orb.eccentricity_squared(R_vec, vel_vec, spec_orb_en, Mbh, prel.G)

        # throw fluff and unbound material
        if which_cut == 'high':
            cut = np.logical_and(data.Den > 1e-12, orb_en < 0)
        else:
            cut = np.logical_and(data.Den > 1e-19, orb_en < 0)
        Rsph_cut, mass_cut, ecc_cut = sec.make_slices([Rsph, Mass, ecc2], cut)
        ecc_cast = single_branch(radii, Rsph_cut, ecc_cut, weights = mass_cut)

        col_ecc2.append(ecc_cast)

    np.save(f'{abspath}/data/{folder}/Ecc2_{which_cut}_{check}.npy', col_ecc2)
    np.save(f'{abspath}/data/{folder}/EccRadii_{which_cut}_{check}.npy', radii)
    with open(f'{abspath}/data/{folder}/Ecc_{which_cut}_{check}_days.txt', 'w') as file:
        file.write(f'# {folder}_{check} \n' + ' '.join(map(str, snaps)) + '\n')
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.close()

else:
    ecc_slice = False
    space_time = True
    error = True

    if ecc_slice:
        radii_grid = [R0, Rt] 
        styles = ['dotted', 'dashed']
        xcfr_grid, ycfr_grid, cfr_grid = [], [], []
        for i, radius_grid in enumerate(radii_grid):
            xcr, ycr, cr = orb.make_cfr(radius_grid)
            xcfr_grid.append(xcr)
            ycfr_grid.append(ycr)
            cfr_grid.append(cr)
        snap = 318
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
        snap, tfb = tfb_data[0], tfb_data[1]
        radii = np.load(f'{path}/EccRadii_{which_cut}_{check}.npy')
        
        # Plot
        plt.figure(figsize=(8,7))
        # set to white the 0 values so they are white in the plot
        ecc[ecc == 0] = np.nan
        img = plt.pcolormesh(radii/apo, tfb, ecc, vmin = 0.5, vmax = .95, cmap = 'viridis', rasterized = True)#cmocean.cm.balance)
        cb = plt.colorbar(img)
        cb.ax.tick_params(labelsize=25)
        cb.set_label(r'Eccentricity' , fontsize = 25, labelpad = 1)
        plt.axvline(x=R0/apo, color = 'k', linestyle = 'dotted', linewidth = 2)
        plt.axvline(x=Rt/apo, color = 'k', linestyle = 'dashed', linewidth = 2)
        plt.text(1.05*R0/apo, 0.9*np.max(tfb), r'$R_{\rm 0}$', fontsize = 24, color = 'k')
        plt.text(1.05*Rt/apo, 0.9*np.max(tfb), r'$R_{\rm t}$', fontsize = 24, color = 'k')
        plt.xscale('log')
        plt.xlabel(r'$R [R_{\rm a}]$')#, fontsize = 25)
        plt.ylabel(r'$t [t_{\rm fb}]$')#, fontsize = 25)
        plt.tick_params(axis='both', which='major', width=1.2, length=10, color = 'k', labelsize=25)
        plt.tick_params(axis='x', which='minor', width=.9, length=6, color = 'k', labelsize=25)
        plt.xlim(0.8*R0/apo, np.max(radii)/apo)
        # plt.ylim(0.054, 1.7)
        plt.savefig(f'{abspath}/Figs/paper/ecc{which_cut}{check}.pdf', bbox_inches='tight')
        plt.show()

    if error: # compare resolutions
        import matplotlib.gridspec as gridspec
        commonpath = f'{abspath}/data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
        ecc_quadL = np.load(f'{commonpath}LowResNewAMR/Ecc2_{which_cut}_LowResNewAMR.npy') 
        eccL = np.sqrt(ecc_quadL)
        tfb_dataL = np.loadtxt(f'{commonpath}LowResNewAMR/Ecc_{which_cut}_LowResNewAMR_days.txt')
        snapL, tfbL = tfb_dataL[0], tfb_dataL[1]
        radii = np.load(f'{commonpath}LowResNewAMR/EccRadii_{which_cut}_LowResNewAMR.npy')

        ecc_quad = np.load(f'{commonpath}NewAMR/Ecc2_{which_cut}_NewAMR.npy') 
        ecc = np.sqrt(ecc_quad)
        tfb_data = np.loadtxt(f'{commonpath}NewAMR/Ecc_{which_cut}_NewAMR_days.txt')
        snap, tfb = tfb_data[0], tfb_data[1]

        ecc_quadH = np.load(f'{commonpath}HiResNewAMR/Ecc2_{which_cut}_HiResNewAMR.npy') 
        eccH = np.sqrt(ecc_quadH)
        tfb_dataH = np.loadtxt(f'{commonpath}HiResNewAMR/Ecc_{which_cut}_HiResNewAMR_days.txt')
        snapH, tfbH = tfb_dataH[0], tfb_dataH[1]

        # relative difference L and middle
        rel_diffL = []
        medianL = np.zeros(len(ecc))
        for i, time in enumerate(tfb):
            idx = np.argmin(np.abs(tfbL - time))
            rel_diff_time_i = find_ratio(eccL[idx], ecc[i]) 
            rel_diff_time_i[eccL[idx] == 0] = 0 
            rel_diffL.append(rel_diff_time_i)
            medianL[i] = np.median(rel_diff_time_i)

        rel_diffH = []
        medianH = np.zeros(len(ecc))
        for i, time in enumerate(tfb):
            idx = np.argmin(np.abs(tfbH - time))
            rel_diff_time_i =  find_ratio(eccH[idx], ecc[i])
            rel_diff_time_i[ecc[i] == 0] = 0
            rel_diffH.append(rel_diff_time_i)
            medianH[i] =  np.median(rel_diff_time_i)

        #%% Plot
        vmin = 0.95
        vmax = 1.05
        fig = plt.figure(figsize=(20, 9))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,.05], hspace=0.5)
        ax1 = fig.add_subplot(gs[0, 0])  # First plot
        ax2 = fig.add_subplot(gs[0, 1])  # Second plot

        img = ax1.pcolormesh(radii/apo, tfb, rel_diffL, cmap = 'Greens', vmin = vmin, vmax = vmax, rasterized = True)
        ax1.set_xscale('log')
        ax1.text(0.35, .88*np.max(tfb), r'$e_{\rm Fid}/e_{\rm Low}$', fontsize = 30, color = 'k')
        ax1.set_xlabel(r'$R [R_{\rm a}]$')
        ax1.set_ylabel(r'$t [t_{\rm fb}]$')

        img = ax2.pcolormesh(radii/apo, tfb, rel_diffH, cmap = 'Greens', vmin = vmin, vmax = vmax, rasterized = True)
        ax2.set_xscale('log')
        ax2.text(0.35, 0.88*np.max(tfb), r'$e_{\rm High}/e_{\rm Fid}$', fontsize = 30, color = 'k')
        ax2.set_xlabel(r'$R [R_{\rm a}]$')

        # Create a colorbar that spans the first two subplots
        cbar_ax = fig.add_subplot(gs[0, 2])  # Colorbar subplot below the first two
        cb = fig.colorbar(img, cax=cbar_ax)
        cb.ax.tick_params(labelsize=25)
        cb.set_label(r'ratio eccentricity')
        # label with 3 decimals in the colorbar
        # cb.set_ticks([0.98, 1, 1.02])
        # cb.set_ticklabels(['0.98', '1', '1.02']) 
        cb.ax.tick_params(width=1, length=11, color = 'k',)

        for ax in [ax1, ax2]:
            ax.axvline(x=Rt/apo, color = 'white', linestyle = 'dashed', linewidth = 2)
            ax.axvline(x=R0/apo, color = 'white', linestyle = ':', linewidth = 2)
            ax.tick_params(axis='both', which='major', width=1.4, length=11, color = 'k',)
            ax.tick_params(axis='x', which='minor', width=1.2, length=7, color = 'k',)
            ax.set_ylim(np.min(tfb), np.max(tfb))
            # ax.set_xlim(R0/apo, np.max(radii)/apo)
        
        plt.savefig(f'{abspath}/Figs/paper/ecc_diff.pdf', bbox_inches='tight')
        plt.tight_layout

        fig, ax3 = plt.subplots(1,1,figsize=(9, 5))
        ax3.plot(tfb, np.abs(1-medianL), c = 'darkorange', label = 'Fid/Low')
        ax3.plot(tfb, np.abs(1-medianL), c = 'yellowgreen', linestyle = (0, (5, 10)))
        ax3.plot(tfb, np.abs(1-medianH), c = 'darkviolet', label = 'High/Fid')
        ax3.plot(tfb, np.abs(1-medianH), c = 'yellowgreen',linestyle = (0, (5, 10)),)
        # ax3.text(0.4, 1, 'Fid vs High', fontsize = 27, color = 'k')
        original_ticks = ax3.get_xticks()
        mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
        all_ticks = np.concatenate(([original_ticks, mid_ticks]))
        ax3.set_xticks(all_ticks)
        labels = [f'{tick:.1f}' if tick in original_ticks else '' for tick in all_ticks]
        ax3.set_xticks(all_ticks)
        ax3.set_xticklabels(labels, fontsize=25)
        ax3.set_xlim(0.08, 1.8)
        ax3.set_ylim(-0.01, 0.08)
        ax3.set_ylabel(r'$|$1 - ratio eccentricity$|$')
        ax3.grid()
        ax3.set_xlabel(r'$t [t_{\rm fb}]$')#, fontsize = 25)
        ax3.legend(fontsize = 25)


# %%
