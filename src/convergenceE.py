""" Total energies (orbital, internal and radiation) at each snapshot, both with cut in coordinates and not.
Cut in density (at 1e-19 code units) is done in both the cases, but not for radiation.
Written to be run on alice"""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'

import numpy as np
from Utilities.selectors_for_snap import select_snap
from Utilities.operators import make_tree
import Utilities.sections as sec
import src.orbits as orb
import Utilities.prelude as prel
import csv
import os
import gc

#
## PARAMETERS STAR AND BH
#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
# thresh = '' # '' or 'cutCoord'

#%%
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Mbh = 10**m
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
t_fall = things['t_fb_days']
t_fall_cgs = t_fall * 24 * 3600

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]

    for i,snap in enumerate(snaps):
        print(snap, flush=False)
        sys.stdout.flush()

        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        data = make_tree(path, snap, energy = True)
        X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Rad
        Rad = Rad_den * vol
        # cut in density NOT in radiation
        cut = den > 1e-19 
        X_cut, Y_cut, Z_cut, VX_cut, VY_cut, VZ_cut, mass_cut, vol_cut, den_cut, ie_den_cut = \
            sec.make_slices([X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den], cut)
        Rsph_cut = np.sqrt(np.power(X_cut, 2) + np.power(Y_cut, 2) + np.power(Z_cut, 2))
        vel_cut = np.sqrt(np.power(VX_cut, 2) + np.power(VY_cut, 2) + np.power(VZ_cut, 2))
        orb_en_cut = orb.orbital_energy(Rsph_cut, vel_cut, mass_cut, params, prel.G)
        ie_cut = ie_den_cut * vol_cut

        # total energies with only the cut in density (not in radiation)
        tot_ie = np.sum(ie_cut)
        tot_orb_en_pos = np.sum(orb_en_cut[orb_en_cut > 0])
        tot_orb_en_neg = np.sum(orb_en_cut[orb_en_cut < 0])
        tot_Rad = np.sum(Rad)

        data_E = [snap, tfb[i], tot_ie, tot_orb_en_pos, tot_orb_en_neg, tot_Rad]
        csv_path = f'{abspath}/data/{folder}/convE_{check}.csv'
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                header = ['snap', ' tfb', ' tot_ie', ' tot_orb_en_pos', ' tot_orb_en_neg', ' tot_Rad']
                writer.writerow(header)
            writer.writerow(data_E)
        file.close()

        del X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den
        del X_cut, Y_cut, Z_cut, VX_cut, VY_cut, VZ_cut, mass_cut, vol_cut, den_cut, ie_den_cut
        del Rsph_cut, vel_cut, orb_en_cut, ie_cut, Rad
        gc.collect()    

        # consider the small box for the cut in coordinates
        # if thresh == 'cutCoord':
        #     box = np.load(f'{path}/box_{snap}.npy')
        #     if np.logical_and(int(snap)>80, int(snap) <= 317):
        #         boxL = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/snap_{snap}/box_{snap}.npy')
        #         if int(snap) <= 267:
        #             boxH = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/snap_{snap}/box_{snap}.npy')
        #         else:
        #             boxH = box
        #     else: 
        #         boxH = box
        #         boxL = box
            # xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0], -0.75*apo]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
            # print("ratio xmin/0.75apo: ", -xmin/(0.75*apo), flush=False)
            # sys.stdout.flush()
            # xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0]]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
            # xmax, ymax, zmax = np.min([box[3], boxL[3], boxH[3]]), np.min([box[4], boxL[4], boxH[4]]), np.min([box[5], boxL[5], boxH[5]])
            # cut_coord = (X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax) & (Z > zmin) & (Z < zmax) 
            # Rad_thresh = Rad[cut_coord]
            # cut_den_coord = (X_cut > xmin) & (X_cut < xmax) & (Y_cut > ymin) & (Y_cut < ymax) & (Z_cut > zmin) & (Z_cut < zmax)
            # den_thresh, orb_en_thresh, ie_thresh, mass_thresh = \
            #     sec.make_slices([den_cut, orb_en_cut, ie_cut, mass_cut], cut_den_coord)

            # # total energies with the cut in density (not in radiation) and coordinates
            # tot_ie_thres[i] = np.sum(ie_thresh)
            # tot_orb_en_thres_pos[i] = np.sum(orb_en_thresh[orb_en_thresh > 0])
            # tot_orb_en_thres_neg[i] = np.sum(orb_en_thresh[orb_en_thresh < 0])
            # tot_Rad_thres[i] = np.sum(Rad_thresh)
            # missingMass[i] = np.sum(mass_thresh)/np.sum(mass_cut)

    # if thresh == 'cutCoord':
    #     print("ratio mass_box/mass_all: ", missingMass, flush = True)
    #     np.save(f'{abspath}/data/{folder}/convE_{check}_thresh.npy', [tot_ie_thres, tot_orb_en_thres_pos, tot_orb_en_thres_neg, tot_Rad_thres])

else:
    import matplotlib.pyplot as plt
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    data = np.loadtxt(f'{abspath}/data/{folder}/convE_{check}.csv', delimiter=',', dtype=float, skiprows=1)
    snapsH, tfbH, IEH, OEHpos, OEHneg, Rad = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
    dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
    tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3] *  prel.en_converter/prel.tsol_cgs

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,7))
    figL, axL = plt.subplots(1,1, figsize = (10,7))
    ax1.plot(tfbH, prel.en_converter * OEHpos, c = 'maroon', label = 'Unbound gas')
    ax1.plot(tfbH, np.abs(prel.en_converter * OEHneg), c = 'maroon', ls = ':', label = 'Bound gas (abs value) ')
    ax1.set_title(r'OE [erg]', fontsize = 24) 
    # ax1.set_ylim(1.16e49, 1.2e49)
    # ax1.set_yscale('log')

    ax2.plot(tfbH, prel.en_converter * IEH, c = 'C1', label = 'Thermal energy')
    ax2.plot(tfbH, prel.en_converter * Rad, c = 'r', label = 'Radiation energy')
    ax2.set_title(r'Thermal and radiation [erg]', fontsize = 24) 

    # compute rates 
    dtH = np.diff(tfbH * t_fall_cgs)
    dOEHpos = np.diff(OEHpos * prel.en_converter)
    dOEHneg = np.diff(OEHneg * prel.en_converter)
    dIEH = np.diff(IEH * prel.en_converter)
    dRad = np.diff(Rad * prel.en_converter)
    axL.plot(tfbdiss, LDiss, c = 'gray', label = r'$\dot{E}_{\rm irr}$', ls = '--')
    axL.plot(tfbH[:-1], np.abs(dOEHpos)/dtH, c = 'maroon', label = 'Unbound gas')
    axL.plot(tfbH[:-1], np.abs(dOEHneg)/dtH, c = 'maroon', ls = ':', label = 'Bound gas')
    axL.plot(tfbH[:-1], np.abs(dIEH)/dtH, c = 'C1', label = 'Thermal energy')
    axL.plot(tfbH[:-1], np.abs(dRad)/dtH, c = 'r', label = 'Radiation energy')
    axL.set_ylabel(r'$|$Energy rates$|$ [erg/s]') 
    axL.set_ylim(1e38, 3e43)

    orginal_ticks = axL.get_xticks()
    middle_ticks = (orginal_ticks[:-1] + orginal_ticks[1:]) /2
    new_ticks = np.sort(np.concatenate((orginal_ticks, middle_ticks)))
    labels = [str(np.round(tick,2)) if tick in orginal_ticks else "" for tick in new_ticks]       
    for ax in (ax1, ax2, axL):
        ax.tick_params(axis='both', which='major', width=1.2, length=7)
        ax.tick_params(axis='both', which='minor', width=0.9, length=5)
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        if ax != ax1:
            ax.set_yscale('log')
        ax.legend(fontsize = 20)
        ax.grid()
        ax.set_xlim(0, np.max(tfbH))
    plt.tight_layout()

# %%
