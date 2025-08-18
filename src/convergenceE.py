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
#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'NewAMR' 
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

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]

    tot_ie = np.zeros(len(snaps))
    tot_orb_en_pos = np.zeros(len(snaps))
    tot_orb_en_neg = np.zeros(len(snaps))
    tot_Rad = np.zeros(len(snaps))
    # if thresh == 'cutCoord':
    #     tot_ie_thres = np.zeros(len(snaps))
    #     tot_orb_en_thres_pos = np.zeros(len(snaps))
    #     tot_orb_en_thres_neg = np.zeros(len(snaps))
    #     tot_Rad_thres = np.zeros(len(snaps))
    #     missingMass = np.zeros(len(snaps))

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
        tot_ie[i] = np.sum(ie_cut)
        tot_orb_en_pos[i] = np.sum(orb_en_cut[orb_en_cut > 0])
        tot_orb_en_neg[i] = np.sum(orb_en_cut[orb_en_cut < 0])
        tot_Rad[i] = np.sum(Rad)

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

    np.save(f'{abspath}/data/{folder}/convE_{check}.npy', [tot_ie, tot_orb_en_pos, tot_orb_en_neg, tot_Rad])
    with open(f'{abspath}/data/{folder}/convE_{check}_days.txt', 'w') as file:
            # file.write(f'# In convE_{check}_thresh you find internal, orbital (pos and neg) and radiation energy [NO denisty/specific] inside the biggest box enclosed in the three simulation volumes and cut in density.\n')
            file.write(f'# In convE_{check} only cut in density.')
            file.write(f'# {folder} \n' + ' '.join(map(str, snaps)) + '\n')
            file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
            file.close()

else:
    import matplotlib.pyplot as plt
    commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    _, tfbL = np.loadtxt(f'{abspath}/data/{commonfolder}LowResNewAMR/convE_LowResNewAMR_days.txt')
    IEL, OELpos, OELneg,  _ = np.load(f'{abspath}/data/{commonfolder}LowResNewAMR/convE_LowResNewAMR.npy')
    _, tfb = np.loadtxt(f'{abspath}/data/{commonfolder}NewAMR/convE_NewAMR_days.txt')
    IE, OEpos, OEneg,   _ = np.load(f'{abspath}/data/{commonfolder}NewAMR/convE_NewAMR.npy')
    _, tfbH = np.loadtxt(f'{abspath}/data/{commonfolder}HiResNewAMR/convE_HiResNewAMR_days.txt')
    IEH, OEHpos, OEHneg, _ = np.load(f'{abspath}/data/{commonfolder}HiResNewAMR/convE_HiResNewAMR.npy')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (22,6))
    ax1.plot(tfbL, prel.en_converter * (OELpos + OELneg), c = 'C1', label = 'Low')
    ax1.plot(tfb, prel.en_converter * (OEpos + OEneg), c = 'yellowgreen', label = 'Fid')
    ax1.plot(tfbH, prel.en_converter * (OEHpos + OEHneg), c = 'darkviolet', label = 'High')
    ax1.set_ylabel(r'$|$OE$|$ [erg/s]')
    ax1.set_title(r'Sum')
    ax1.legend(fontsize = 15) 

    ax2.plot(tfbL, prel.en_converter * OELpos, c = 'C1', label = 'Low')
    ax2.plot(tfb, prel.en_converter * OEpos, c = 'yellowgreen', label = 'Fid')
    ax2.plot(tfbH, prel.en_converter * OEHpos, c = 'darkviolet', label = 'High')
    ax2.set_ylabel(r'$|$OE$|$ [erg/s]')
    ax2.set_title(r'Unbound gas')

    ax3.plot(tfbL, prel.en_converter * np.abs(OELneg), c = 'C1', label = 'Low')
    ax3.plot(tfb, prel.en_converter * np.abs(OEneg), c = 'yellowgreen', label = 'Fid')
    ax3.plot(tfbH, prel.en_converter * np.abs(OEHneg), c = 'darkviolet', label = 'High')
    ax3.set_title(r'Bound gas')
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        # ax.set_yscale('log')
        # ax.set_ylim(1e49, 1.5e49)
        ax.grid()
    plt.tight_layout()


# %%
