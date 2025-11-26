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
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
t_fall = things['t_fb_days']
t_fall_cgs = t_fall * 24 * 3600

#%%
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
        kin_en_cut = 0.5 * mass_cut *vel_cut**2
        orb_en_cut = orb.orbital_energy(Rsph_cut, vel_cut, mass_cut, params, prel.G)
        ie_cut = ie_den_cut * vol_cut

        # total energies with only the cut in density (not in radiation)
        tot_ie = np.sum(ie_cut)
        tot_orb_en_pos = np.sum(orb_en_cut[orb_en_cut > 0])
        tot_orb_en_neg = np.sum(orb_en_cut[orb_en_cut < 0])
        tot_Rad = np.sum(Rad)
        tot_kin_en_pos = np.sum(kin_en_cut[orb_en_cut >= 0])
        tot_kin_en_neg = np.sum(kin_en_cut[orb_en_cut < 0])

        data_E = [snap, tfb[i], tot_ie, tot_orb_en_pos, tot_orb_en_neg, tot_Rad, tot_kin_en_pos, tot_kin_en_neg]
        csv_path = f'{abspath}/data/{folder}/convE_{check}.csv'
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                header = ['snap', ' tfb', ' tot_ie', ' tot_orb_en_pos', ' tot_orb_en_neg', ' tot_Rad', ' tot_kin_en_pos', ' tot_kin_en_neg']
                writer.writerow(header)
            writer.writerow(data_E)
        file.close()

        del X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den
        del X_cut, Y_cut, Z_cut, VX_cut, VY_cut, VZ_cut, mass_cut, vol_cut, den_cut, ie_den_cut
        del Rsph_cut, vel_cut, orb_en_cut, ie_cut, Rad
        gc.collect()    

else:
    import matplotlib.pyplot as plt
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    data = np.loadtxt(f'{abspath}/data/{folder}/convE_{check}.csv', delimiter=',', dtype=float, skiprows=1)
    snaps, tfb, IE, OEpos, OEEneg, Rad, Kinpos, Kinneg = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
    tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3] *  prel.en_converter/prel.tsol_cgs
    totalK = Kinneg + Kinpos

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,7))
    figL, axL = plt.subplots(1,1, figsize = (10,7))
    ax1.plot(tfb, prel.en_converter * OEpos, c = 'plum', label = 'Orbital energy unbound gas')
    ax1.plot(tfb, np.abs(prel.en_converter * OEEneg), c = 'plum', ls = ':', label = 'Orbital energy bound gas (abs value) ')
    ax1.set_title(r'OE [erg]', fontsize = 24) 
    ax2.set_ylim(1e43, 6e49)
    # ax1.set_yscale('log')

    ax2.plot(tfb, prel.en_converter * IE, c = 'magenta', label = 'Thermal energy')
    ax2.plot(tfb, prel.en_converter * Rad, c = 'darkviolet', label = 'Radiation energy')
    ax2.plot(tfb, prel.en_converter * Kinpos, c = 'plum', label = 'Kinetic energy unbound gas')
    ax2.plot(tfb, np.abs(prel.en_converter * Kinneg), c = 'plum', ls = ':', label = 'Kinetic energy bound gas (abs value)')
    ax2.set_title(r'Thermal and radiation [erg]', fontsize = 24) 

    # compute rates 
    dtH = np.diff(tfb * t_fall_cgs)
    dOEpos = np.diff(OEpos * prel.en_converter)
    dOEEneg = np.diff(OEEneg * prel.en_converter)
    dIE = np.diff(IE * prel.en_converter)
    dRad = np.diff(Rad * prel.en_converter)
    dKinpos = np.diff(Kinpos * prel.en_converter)
    dKinneg = np.diff(Kinneg * prel.en_converter)
    dTotalK = np.diff(totalK * prel.en_converter)
    axL.plot(tfb[:-1], np.abs(dOEpos)/dtH, c = 'plum', label = 'Orb. en. unbound gas')
    axL.plot(tfb[:-1], np.abs(dOEEneg)/dtH, c = 'plum', ls = ':', label = 'Orb. en. bound gas')
    axL.plot(tfb[:-1], np.abs(dIE)/dtH, c = 'magenta', label = 'Thermal energy')
    axL.plot(tfb[:-1], np.abs(dRad)/dtH, c = 'darkviolet', label = 'Radiation energy')
    axL.plot(tfbdiss, LDiss, c = 'gray', label = r'$\dot{E}_{\rm irr}$', ls = '--')
    # axL.plot(tfb[:-1], np.abs(dKinpos)/dtH, c = 'plum', label = 'Kinetic energy unbound gas')
    # axL.plot(tfb[:-1], np.abs(dKinneg)/dtH, c = 'plum', ls = ':', label = 'Kinetic energy bound gas (abs value)')
    # axL.plot(tfb[:-1], np.abs(dTotalK)/dtH, c = 'brown', label = 'Total Kinetic energy')
    axL.set_ylabel(r'Luminosity [erg/s]') 
    axL.set_ylim(1e39, 1e44)

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
        ax.legend(fontsize = 18)
        ax.grid()
        ax.set_xlim(0, np.max(tfb))
    fig.tight_layout()
    fig.savefig(f'{abspath}/Figs/paper/Ebudget_{check}.png', dpi = 300)
    figL.tight_layout()
    figL.savefig(f'{abspath}/Figs/paper/Ebudget_absrates_{check}.pdf', dpi = 300)


    


# %%
