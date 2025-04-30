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

#
## PARAMETERS STAR AND BH
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
Rp = orb.pericentre(Rstar, mstar, Mbh, beta)
R0 = 0.6*Rp
compton = 'Compton'
check = 'HiRes' 

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
Mbh = 10**m
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]

    col_ie = np.zeros(len(snaps))
    col_orb_en_pos = np.zeros(len(snaps))
    col_orb_en_neg = np.zeros(len(snaps))
    col_orb_en = np.zeros(len(snaps))
    col_Rad = np.zeros(len(snaps))
    col_ie_thres = np.zeros(len(snaps))
    col_orb_en_thres_pos = np.zeros(len(snaps))
    col_orb_en_thres_neg = np.zeros(len(snaps))
    col_orb_en_thres = np.zeros(len(snaps))
    col_Rad_thres = np.zeros(len(snaps))
    missingMass = np.zeros(len(snaps))

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
        orb_en_cut = orb.orbital_energy(Rsph_cut, vel_cut, mass_cut, prel.G, prel.csol_cgs, Mbh, R0)
        ie_cut = ie_den_cut * vol_cut

        # total energies with only the cut in density (not in radiation)
        col_ie[i] = np.sum(ie_cut)
        col_orb_en_pos[i] = np.sum(orb_en_cut[orb_en_cut > 0])
        col_orb_en_neg[i] = np.sum(orb_en_cut[orb_en_cut < 0])
        col_orb_en[i] = np.sum(orb_en_cut)
        col_Rad[i] = np.sum(Rad)

        # consider the small box for the cut in coordinates
        box = np.load(f'{path}/box_{snap}.npy')
        if np.logical_and(int(snap)>80, int(snap) <= 317):
            boxL = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/snap_{snap}/box_{snap}.npy')
            if int(snap) <= 267:
                boxH = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/snap_{snap}/box_{snap}.npy')
            else:
                boxH = box
        else: 
            boxH = box
            boxL = box
        # xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0], -0.75*apo]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
        # print("ratio xmin/0.75apo: ", -xmin/(0.75*apo), flush=False)
        # sys.stdout.flush()
        xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0]]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
        xmax, ymax, zmax = np.min([box[3], boxL[3], boxH[3]]), np.min([box[4], boxL[4], boxH[4]]), np.min([box[5], boxL[5], boxH[5]])
        cut_coord = (X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax) & (Z > zmin) & (Z < zmax) 
        Rad_thresh = Rad[cut_coord]
        cut_den_coord = (X_cut > xmin) & (X_cut < xmax) & (Y_cut > ymin) & (Y_cut < ymax) & (Z_cut > zmin) & (Z_cut < zmax)
        den_thresh, orb_en_thresh, ie_thresh, mass_thresh = \
            sec.make_slices([den_cut, orb_en_cut, ie_cut, mass_cut], cut_den_coord)

        # total energies with the cut in density (not in radiation) and coordinates
        col_ie_thres[i] = np.sum(ie_thresh)
        col_orb_en_thres_pos[i] = np.sum(orb_en_thresh[orb_en_thresh > 0])
        col_orb_en_thres_neg[i] = np.sum(orb_en_thresh[orb_en_thresh < 0])
        col_orb_en_thres[i] = np.sum(orb_en_thresh)
        col_Rad_thres[i] = np.sum(Rad_thresh)

        missingMass[i] = np.sum(mass_thresh)/np.sum(mass_cut)
    print("ratio mass_box/mass_all: ", missingMass, flush=False)
    sys.stdout.flush()

    np.save(f'{abspath}/data/{folder}/convE_{check}Pot.npy', [col_ie, col_orb_en_pos, col_orb_en_neg, col_orb_en, col_Rad])
    np.save(f'{abspath}/data/{folder}/convE_{check}_threshPot.npy', [col_ie_thres, col_orb_en_thres_pos, col_orb_en_thres_neg, col_orb_en_thres, col_Rad_thres])
    with open(f'{abspath}/data/{folder}/convE_{check}_days.txt', 'w') as file:
            file.write(f'# In convE_{check}_thresh you find internal, orbital (pos and neg) and radiation energy [NO denisty/specific] inside the biggest box enclosed in the three simulation volumes and cut in density.\n')
            file.write(f'# In convE_{check} only cut in density.')
            file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()

else:
    import matplotlib.pyplot as plt
    from Utilities.operators import find_ratio
    commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    _, tfb = np.loadtxt(f'{abspath}/data/{commonfolder}/convE__days.txt')
    _, tfbL = np.loadtxt(f'{abspath}/data/{commonfolder}LowRes/convE_LowRes_days.txt')
    _, tfbH = np.loadtxt(f'{abspath}/data/{commonfolder}HiRes/convE_HiRes_days.txt')
   
    threshes = ['', 
                '_thresh75',
                '_thresh5',  
                '_thresh2', 
                '_thresh20Dencut', 
                '_thresh18Dencut',
                'Pot',
                '_threshPot',]
    titles = [r'$\rho>10^{19} [M_\odot/R_\odot^3]$ ', 
                r'cut $x>-0.75R_{a}, \rho>10^{19} [M_\odot/R_\odot^3]$', 
                r'cut $x>-5R_{a}, \rho>10^{19} [M_\odot/R_\odot^3]$', 
                r'cut $x>-2R_{a}, \rho>10^{19} [M_\odot/R_\odot^3]$',  
                r'$\rho>10^{20} [M_\odot/R_\odot^3]$, box', 
                r'$\rho>10^{18} [M_\odot/R_\odot^3]$, box',
                r'$\rho>10^{19} [M_\odot/R_\odot^3]$, box, correct potential',
                r'$\rho>10^{19} [M_\odot/R_\odot^3]$, box, correct potential, cut in density']
    for i, thresh in enumerate(threshes):
        IEL, OELpos, OELneg, _,  _ = np.load(f'{abspath}/data/{commonfolder}LowRes/convE_LowRes{thresh}.npy')
        IE, OEpos, OEneg, _,  _ = np.load(f'{abspath}/data/{commonfolder}/convE_{thresh}.npy')
        IEH, OEHpos, OEHneg, _,  _ = np.load(f'{abspath}/data/{commonfolder}HiRes/convE_HiRes{thresh}.npy')
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (14,5))
        ax1.plot(tfbL, prel.en_converter * OELpos, c = 'C1', label = 'Low')
        ax1.plot(tfb, prel.en_converter * OEpos, c = 'yellowgreen', label = 'Fid')
        ax1.plot(tfbH, prel.en_converter * OEHpos, c = 'darkviolet', label = 'High')
        # ax1.plot(tfbL, prel.en_converter * OEL_posnocut, '--', c = 'maroon')
        ax1.set_ylabel(r'$|$OE$|$ [$10^{48}$ erg/s]')
        ax1.set_title(r'Unbound gas')
        # ax1.set_ylim(11.5,12)
        ax1.legend(fontsize = 15)
        ax1.set_yscale('log')

        ax2.plot(tfbL, prel.en_converter * np.abs(OELneg), c = 'C1', label = 'Low')
        ax2.plot(tfb, prel.en_converter * np.abs(OEneg), c = 'yellowgreen', label = 'Fid')
        ax2.plot(tfbH, prel.en_converter * np.abs(OEHneg), c = 'darkviolet', label = 'High')
        ax2.set_title(r'Bound gas')
        for ax in (ax1, ax2):
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
        # ax2.set_ylim(12.2, 13)
        ax2.set_yscale('log')
        plt.suptitle(f'{titles[i]}', fontsize = 15)
        plt.tight_layout()

        if i == 0:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))
            ax1.plot(tfbL,find_ratio(OELpos, OEpos[1:len(OELpos)+1]), c = 'C1')
            ax1.plot(tfbL, find_ratio(OELpos, OEpos[1:len(OELpos)+1]), '--', c = 'yellowgreen')
            ax1.plot(tfbH, find_ratio(OEHpos, OEpos[:len(OEHpos)+1]), c = 'darkviolet')
            ax1.plot(tfbH, find_ratio(OEHpos, OEpos[:len(OEHpos)+1]), '--', c = 'yellowgreen')
            ax1.set_title(r'Unbound gas')
            ax1.legend(fontsize = 15)
            ax1.set_ylabel(r'ratio OE')
            ax2.plot(tfbL,find_ratio(OELneg, OEneg[1:len(OELneg)+1]), c = 'C1')
            ax2.plot(tfbL, find_ratio(OELneg, OEneg[1:len(OELneg)+1]), '--', c = 'yellowgreen')
            ax2.plot(tfbH, find_ratio(OEHneg, OEneg[:len(OEHneg)+1]), c = 'darkviolet')
            ax2.plot(tfbH, find_ratio(OEHneg, OEneg[:len(OEHneg)+1]), '--', c = 'yellowgreen')
            ax2.set_title(r'Bound gas')
            plt.suptitle(f'{titles[i]}', fontsize = 15)

            fig, ax1 = plt.subplots(1,1, figsize = (7,5))
            ax1.plot(tfbL,find_ratio(IEL, IE[1:len(OELpos)+1]), c = 'C1')
            ax1.plot(tfbL, find_ratio(IEL, IE[1:len(IEL)+1]), '--', c = 'yellowgreen')
            ax1.plot(tfbH, find_ratio(IEH, IE[:len(IEH)+1]), c = 'darkviolet')
            ax1.plot(tfbH, find_ratio(IEH, IE[:len(IEH)+1]), '--', c = 'yellowgreen')
            ax1.legend(fontsize = 15)
            ax1.set_ylabel(r'ratio IE')
            plt.suptitle(f'{titles[i]}', fontsize = 15)

        fig, (ax1) = plt.subplots(1,1, figsize = (7,5))
        ax1.plot(tfbL, 1e-46*prel.en_converter * IEL, c = 'C1', label = 'Low')
        ax1.plot(tfb, 1e-46*prel.en_converter * IE, c = 'yellowgreen', label = 'Fid')
        ax1.plot(tfbH, 1e-46*prel.en_converter * IEH, c = 'darkviolet', label = 'High')
        ax1.set_ylabel(r'IE [$10^{46}$ erg/s]')
        plt.suptitle(f'{titles[i]}', fontsize = 15)
        ax1.legend(fontsize = 15)
        ax1.set_xlabel(r'$t [t_{\rm fb}]$')
        ax1.set_yscale('log')

    _, OEpos19, OEneg19, _,  _ = np.load(f'{abspath}/data/{commonfolder}/convE_.npy')
    _, OEpos20, OEneg20, _,  _ = np.load(f'{abspath}/data/{commonfolder}/convE__thresh18Dencut.npy')
    print(OEpos19-OEpos20)
    print(OEneg19-OEneg20)