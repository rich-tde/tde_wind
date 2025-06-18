""" If alice: Compute and save the mass enclosed and the total diss rate in a sphere of radius R0, Rt, a_mb, apo for all the snapshots.
If local: plots"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import make_tree, sort_list, calc_deriv
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import src.orbits as orb

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'

##
# MAIN
##
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds

if alice:
    Rcheck = np.array([R0, Rt, a_mb, apo])
    check = 'NewAMRRemoveCenter'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
    
    Mass_encl = np.zeros((len(snaps), len(Rcheck)))
    Diss_pos_encl = np.zeros((len(snaps), len(Rcheck)))
    Diss_neg_encl = np.zeros((len(snaps), len(Rcheck)))
    tfb = np.zeros(len(snaps))
    for i,snap in enumerate(snaps):
        print(snap,  flush = True)
        sys.stdout.flush()
        # Load data
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        tfb[i] = np.loadtxt(f'{path}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        mass, den, vol, diss_den = data.Mass, data.Den, data.Vol, data.Diss
        Rsph = np.sqrt(data.X**2 + data.Y**2 + data.Z**2)
        cut = den > 1e-19
        mass, Rsph, diss_den, vol = \
            make_slices([mass, Rsph, diss_den, vol], cut)
        diss_den_pos = diss_den >= 0
        diss_den_neg = diss_den < 0
        for j, R in enumerate(Rcheck):
            enclosed = Rsph < R
            Mass_encl[i,j] = np.sum(mass[enclosed])
            Diss_pos_encl[i,j] = np.sum(diss_den[np.logical_and(diss_den_pos, enclosed)] * vol[np.logical_and(diss_den_pos, enclosed)])
            Diss_neg_encl[i,j] = np.sum(diss_den[np.logical_and(diss_den_neg, enclosed)] * vol[np.logical_and(diss_den_neg, enclosed)])
    # save in code units
    np.savetxt(f'{abspath}/data/{folder}/{check}R_Mass_encl.txt', Rcheck)
    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl_time.txt', tfb)
    np.savetxt(f'{abspath}/data/{folder}/{check}Mass_encl.txt', Mass_encl)
    np.savetxt(f'{abspath}/data/{folder}/{check}Diss_pos_encl.txt', Diss_pos_encl)
    np.savetxt(f'{abspath}/data/{folder}/{check}Diss_neg_encl.txt',Diss_neg_encl)

else:
    what_to_plot = 'Maccr'
    
    if what_to_plot == 'sink':
        checks = ['LowResNewAMR', 'LowResNewAMRRemoveCenter', 'OpacityNewNewAMR', 'NewAMRRemoveCenter']
        lines = ['solid', '--', 'solid', '--']
        checklab = ['Final Extr', 'Final Extr + sink', 'Fid, New Extr', 'Fid, Final Extr + sink']
        colorcheck = ['plum', 'maroon', 'royalblue', 'k'] #['C1', 'yellowgreen', 'darkviolet']
        commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

        # Plot mass and dissipation rate enclosed in a sphere of radius Rcheck: 
        # LowResNewAMR and LowResNewAMRRemoveCenter in first column (ax1, axDiss1)
        # OpacityNewNewAMR and NewAMRRemoveCenter in second column (ax2, axDiss2)
        fig, ((ax1, ax2), (axDiss1, axDiss2)) = plt.subplots(2,2, figsize = (16,14)) #len(checks), 1, figsize = (5, 8))
        for i, check in enumerate(checks):
            # load and covert
            tfb = np.loadtxt(f'{abspath}/data/{commonfolder}{check}/{check}Mass_encl_time.txt')
            Mass_encl = np.loadtxt(f'{abspath}/data/{commonfolder}{check}/{check}Mass_encl.txt', comments="#")
            Mass_encl = np.transpose(Mass_encl)
            Diss_encl = np.loadtxt(f'{abspath}/data/{commonfolder}{check}/{check}Diss_pos_encl.txt')
            Diss_encl = (np.transpose(Diss_encl)) * prel.en_converter/prel.tsol_cgs
            data_totDiss = np.loadtxt(f'{abspath}/data/{commonfolder}{check}/Rdiss_{check}cutDen.txt')
            tfbdissAMR, LDissAMR = data_totDiss[0], data_totDiss[2] * prel.en_converter/prel.tsol_cgs
            if i < 2:
                ax = ax1
                axDiss = axDiss1
            else:
                ax = ax2
                axDiss = axDiss2
            ax.plot(tfb, Mass_encl[0]/mstar, c = 'deepskyblue', ls = lines[i], linewidth = 2, label = {r'R $< R_0$' if i in [0,2]  else 'with sink term'})
            ax.plot(tfb, Mass_encl[1]/mstar, c = 'coral', ls = lines[i], linewidth = 2, label = {r'R $< R_{t}$' if i in [0,2] else ''})
            ax.plot(tfb, Mass_encl[2]/mstar, c = 'mediumseagreen', ls = lines[i], linewidth = 2, label =  {r'R $< a_{\rm mb}$' if i in [0,2] else ''})
            # ax.plot(tfb, Mass_encl[3]/mstar, c = 'm', ls = lines[i], linewidth = 2, label = {r'R $< R_{\rm apo}$' if i in [0,2] else ''})
            # just for the label
            ax.plot(tfb, Mass_encl[0]/mstar, c = 'deepskyblue', ls = lines[i], linewidth = 2, label = {r'no sink term' if i in [0,2]  else ''})
            
            axDiss.plot(tfbdissAMR, LDissAMR, c = 'k', ls = lines[i], linewidth = 2, label = {r'total Diss' if i in [0,2]  else ''})
            axDiss.plot(tfb, Diss_encl[0], c = 'deepskyblue', ls = lines[i], linewidth = 2, label = {r'R $< R_0$' if i in [0,2]  else 'with sink term'})
            axDiss.plot(tfb, Diss_encl[1], c = 'coral', ls = lines[i], linewidth = 2, label = {r'R $< R_{t}$' if i in [0,2] else ''})
            axDiss.plot(tfb, Diss_encl[2], c = 'mediumseagreen', ls = lines[i], linewidth = 2, label =  {r'R $< a_{\rm mb}$' if i in [0,2] else ''})
            # axDiss.plot(tfb, Diss_encl[3], c = 'm', ls = lines[i], linewidth = 2, label = {r'R $< R_{\rm apo}$' if i in [0,2] else ''})
            # just for the label
            axDiss.plot(tfb, Diss_encl[0], c = 'deepskyblue', ls = lines[i], linewidth = 2, label = {r'no sink term' if i in [0,2]  else ''})
            
            # axDiss.scatter(tfb[137], Diss_encl[0][136], c = 'k', marker = 'x', s = 100, label = r'$t_{\rm fb}$')
            # axDiss.scatter(tfb[137], Diss_encl[0][137], c = 'k', marker = 'x', s = 100, label = r'$t_{\rm fb}$')
            # filename = f'{abspath}/data/{commonfolder}{check}/{check}_red.csv'
            # dataLum = np.loadtxt(filename, delimiter=',', dtype=float)
            # t_Lum = dataLum[:, 1]
            # Lum = dataLum[:, 2]
            # Lum, t_Lum = sort_list([Lum, t_Lum], t_Lum, unique=True)

            # axLum.plot(t_Lum, Lum, c = 'b', ls = lines[i], linewidth = 2, label = {r'L$_{\rm FLD}$' if i in [0,2] else ''})
            # axLum.plot(tfbdissAMR, LDissAMR, c = 'k', ls = lines[i], linewidth = 2, label = {r'no sink term' if i in [0,2]  else ''})
            # axLum.set_xlabel(r'$\rm t [t_{fb}]$')
            # axLum.set_ylim(5e37, 1e43)
        
        ax1.set_ylabel(r'Mass enclosed $[M_\star]$')
        axDiss1.set_ylabel(r'Dissipation rate enclosed [erg/s]')
        ax2.legend(fontsize = 20)
        axDiss2.legend(fontsize = 20)
        ax1.set_title(f'Low Res', fontsize = 20)
        ax2.set_title(f'Fid Res', fontsize = 20)
        original_ticks = ax1.get_xticks()
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        # y 
        original_ticks_y = ax1.get_yticks()
        midpoints_y = (original_ticks_y[:-1] + original_ticks_y[1:]) / 2
        new_ticks_y = np.sort(np.concatenate((original_ticks_y, midpoints_y)))
        for a in [ax1, ax2, axDiss1, axDiss2]:
            a.set_xticks(new_ticks)
            a.tick_params(axis='both', which='major', length=7, width=1)
            a.tick_params(axis='both', which='minor', length=4, width=1)
            a.grid()
            # a.axvline(tfb[136], c = 'k', ls = '--', linewidth = 2, label = r'$t_{\rm fb}$')
            # print(tfb[136], tfb[137])
            a.set_xlim(0.01, 1.8)
            if a in [ax1, ax2]:
                a.set_yticks(new_ticks_y)
                a.set_ylim(1e-9, 9e-2)
            else:
                a.set_ylim(2e37, 1e43)
                ax.set_xlabel(r'$\rm t [t_{fb}]$')
            a.set_yscale('log')

        fig.suptitle(f'New AMR runs', fontsize = 25)
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/Test/MazeOfRuns/sink/massDiss_encl.png', bbox_inches='tight')

    #%%
    if what_to_plot == 'Maccr':
        checks = ['', 'OpacityNewNewAMR', 'NewAMRRemoveCenter']
        checklab = ['Fid, Old', 'Fid, lin Extr, New AMR', 'Fid, final Extr, New AMR + sink']
        fig, ax = plt.subplots(1,3, figsize = (22, 5))
        for i, check in enumerate(checks):
            folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
            tfb = np.loadtxt(f'{abspath}/data/{folder}/{check}Mass_encl_time.txt')
            tfb_cgs = tfb * tfallback_cgs
            tfbDiss, _, Ldisstot_pos, _, _ = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}cutDen.txt', comments = '#')
            Mass_encl = np.loadtxt(f'{abspath}/data/{folder}/{check}Mass_encl.txt')
            Mass_encl = np.transpose(Mass_encl)
            Diss_encl = np.loadtxt(f'{abspath}/data/{folder}/{check}Diss_pos_encl.txt')
            Diss_encl = (np.transpose(Diss_encl)) * prel.en_converter/prel.tsol_cgs
            Diss_encl0 = Diss_encl[0]

            Mdot0 = calc_deriv(tfb_cgs, Mass_encl[0]) * prel.Msol_cgs 
            Lacc0 = 0.05 * Mdot0 * prel.c_cgs**2
            # delete nan
            nan = np.isnan(Diss_encl0)
            Diss_encl0 = Diss_encl0[~nan]
            tfbDis = tfb[~nan]
            nan = np.isnan(Lacc0)
            Lacc0 = Lacc0[~nan]
            tfbL = tfb[~nan]
            # exclude the 127 because old data for LowRes
            if check == 'LowRes':
                Diss_encl0 = np.delete(Diss_encl0, 127)
                tfbDis = np.delete(tfbDis, 127)
            # find where Lacc0 give nan and remove it from Lacc0 and tfb
            ax[i].plot(tfbDis, np.abs(Diss_encl0), c = 'cornflowerblue', label = r'L$_{\rm diss}$')
            ax[i].plot(tfbL, np.abs(Lacc0), c = 'chocolate', label = r'L$_{\rm acc}=0.05\dot{M}c^2$')
            ax[i].set_xlabel(r'$\rm t [t_{fb}]$')
            ax[i].set_yscale('log')
            ax[i].set_ylim(1e37, 1e44)
            # ax[i].text(np.max(tfbL)-0.4, 2e37, f'{checklab[i]} res', fontsize = 25)
            ax[i].set_title(f'{checklab[i]}', fontsize = 20)
            ax[i].grid()
            original_ticks = ax[i].get_xticks()
            midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
            new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
            ax[i].set_xticks(new_ticks)
            labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
            ax[i].set_xticklabels(labels)
            ax[i].set_xlim(0, np.max(tfbL))


        ax[0].set_ylabel(r'Luminosity [erg/s] from $R<R_0$')#, fontsize = 25)
        ax[0].legend(fontsize = 18)
        # plt.savefig(f'{abspath}/Figs/paper/Maccr_encl.pdf', bbox_inches='tight')

# # %%

# %%
