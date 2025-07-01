""" Compute Rdissipation. Written to be run on alice."""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
import csv
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree
from Utilities.sections import make_slices
from Utilities.selectors_for_snap import select_snap
import Utilities.sections as sec

##
# PARAMETERS
## 
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'NewAMR'
do_cut = 'cutDen' # '' or 'cutDen'

Rt = Rstar * (Mbh/mstar)**(1/3)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

if alice:
    snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 

    Ldisstot_pos = np.zeros(len(snaps))
    Rdiss_pos = np.zeros(len(snaps))
    Ldisstot_neg = np.zeros(len(snaps))
    Rdiss_neg = np.zeros(len(snaps))
    tfb = np.zeros(len(snaps))
    for i, snap in enumerate(snaps):
        print(snap, flush=False) 
        sys.stdout.flush()
        
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        tfb[i] = np.loadtxt(f'{path}/tfb_{snap}.txt')
        data = make_tree(path, snap, energy = True)
        X, Y, Z, vol, den, Rad_den, Ediss_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Rad, data.Diss
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)

        # cut low density
        if do_cut == 'cutDen':
            cut = den > 1e-19
            Rsph, vol, Rad_den, Ediss_den = \
                make_slices([Rsph, vol, Rad_den, Ediss_den], cut)
        Ediss = Ediss_den * vol # energy dissipation rate [energy/time] in code units
        Ldisstot_pos[i] = np.sum(Ediss[Ediss_den >= 0])
        Rdiss_pos[i] = np.sum(Rsph[Ediss_den >= 0] * Ediss[Ediss_den >= 0]) / np.sum(Ediss[Ediss_den >= 0])
        Ldisstot_neg[i] = np.sum(Ediss[Ediss_den < 0])
        Rdiss_neg[i] = np.sum(Rsph[Ediss_den < 0] * Ediss[Ediss_den < 0]) / np.sum(Ediss[Ediss_den < 0])

    with open(f'{abspath}/data/{folder}/Rdiss_{check}{do_cut}.txt','w') as file:
        if do_cut == 'cutDen':
            file.write('# just cells with den > 1e-19 \n')
        file.write('# Separeted positive and negative values for Diss. You should use the positive (since it is positive at X=Rp) \n')
        file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.write(f'# Rdiss [R_\odot] from positive Ediss  \n' + ' '.join(map(str, Rdiss_pos)) + '\n')
        file.write(f'# Total dissipation luminosity [energy/time in code units] from positive Ediss \n' + ' '.join(map(str, Ldisstot_pos)) + '\n')
        file.write(f'# Rdiss [R_\odot] from negative Ediss  \n' + ' '.join(map(str, Rdiss_neg)) + '\n')
        file.write(f'# Total dissipation luminosity [energy/time in code units] from negative Ediss \n' + ' '.join(map(str, Ldisstot_neg)) + '\n')
        file.close()

else:
    from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall
    commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    dataL = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/Rdiss_LowResNewAMRcutDen.txt', comments = '#')
    tfbL, Rdiss_posL, Ldisstot_posL, Rdiss_negL, Ldisstot_negL = dataL[0], dataL[1], dataL[2], dataL[3], dataL[4]
    data = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/Rdiss_NewAMRcutDen.txt', comments = '#')
    tfb, Rdiss_pos, Ldisstot_pos, Rdiss_neg, Ldisstot_neg = data[0], data[1], data[2], data[3], data[4]
    dataH = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/Rdiss_HiResNewAMRcutDen.txt', comments = '#')
    tfbH, Rdiss_posH, Ldisstot_posH, Rdiss_negH, Ldisstot_negH = dataH[0], dataH[1], dataH[2], dataH[3], dataH[4]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    ax1.plot(tfbL, Ldisstot_posL * prel.en_converter/prel.tsol_cgs, color = 'C1', label = 'Low')
    ax1.plot(tfb, Ldisstot_pos * prel.en_converter/prel.tsol_cgs, color = 'yellowgreen', label = 'Fid')
    ax1.plot(tfbH, Ldisstot_posH * prel.en_converter/prel.tsol_cgs, color = 'darkviolet', label = 'High')
    # ax1.plot(tfb, np.abs(Ldisstot_neg) * prel.en_converter/prel.tsol_cgs, label = r'$|$L negative, cut den$|$', color = 'grey')
    ax1.set_ylabel('Dissipation rate [erg/s]')
    ax1.legend(fontsize = 16)

    ax2.plot(tfbL, Rdiss_posL/apo, c = 'C1')
    ax2.plot(tfb, Rdiss_pos/apo, c = 'yellowgreen')
    ax2.plot(tfbH, Rdiss_posH/apo, c = 'darkviolet')
    ax2.axhline(Rt/apo, color = 'k', linestyle = '--', label = r'$R_{\rm t}$')


    ax2.set_ylabel(r'$R_{\rm diss}$ [R$_{\rm a}$]')
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', length = 7, width = 1.2)
        ax.tick_params(axis='both', which='minor', length = 3, width = 1)
    ax2.set_xlabel(r't [t$_{\rm fb}$]')

    # with ratios
    tfb_ratioDiss_L, ratio_DissL = ratio_BigOverSmall(tfb, Ldisstot_pos, tfbL, Ldisstot_posL)
    tfb_ratioDiss_H, ratio_DissH = ratio_BigOverSmall(tfb, Ldisstot_pos, tfbH, Ldisstot_posH)
    tfb_ratioRdiss_L, ratio_RdissL = ratio_BigOverSmall(tfb, Rdiss_pos, tfbL, Rdiss_posL)
    tfb_ratioRdiss_H, ratio_RdissH = ratio_BigOverSmall(tfb, Rdiss_pos, tfbH, Rdiss_posH)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 16))
    ax1.plot(tfbL, Ldisstot_posL * prel.en_converter/prel.tsol_cgs, color = 'C1', label = 'Low')
    ax1.plot(tfb, Ldisstot_pos * prel.en_converter/prel.tsol_cgs, color = 'yellowgreen', label = 'Fid')
    ax1.plot(tfbH, Ldisstot_posH * prel.en_converter/prel.tsol_cgs, color = 'darkviolet', label = 'High')
    ax1.set_ylabel('Dissipation rate [erg/s]')
    ax1.legend(fontsize = 16)

    ax3.plot(tfb_ratioDiss_L, ratio_DissL, color = 'C1')
    ax3.plot(tfb_ratioDiss_L, ratio_DissL, '--', color = 'yellowgreen')
    ax3.plot(tfb_ratioDiss_H, ratio_DissH, color = 'yellowgreen')
    ax3.plot(tfb_ratioDiss_H, ratio_DissH, '--', color = 'darkviolet')

    ax2.plot(tfbL, Rdiss_posL/apo, c = 'C1')
    ax2.plot(tfb, Rdiss_pos/apo, c = 'yellowgreen')
    ax2.plot(tfbH, Rdiss_posH/apo, c = 'darkviolet')
    ax2.set_ylabel(r'$R_{\rm diss}$ [R$_{\rm a}$]')
    ax2.axhline(Rt/apo, color = 'k', linestyle = '--', label = r'$R_{\rm t}$')
    ax3.set_ylabel('Ratio of dissipation rate')

    ax4.plot(tfb_ratioRdiss_L, ratio_RdissL, color = 'C1')
    ax4.plot(tfb_ratioRdiss_L, ratio_RdissL, '--', color = 'yellowgreen')
    ax4.plot(tfb_ratioRdiss_H, ratio_RdissH, color = 'yellowgreen')
    ax4.plot(tfb_ratioRdiss_H, ratio_RdissH, '--', color = 'darkviolet')
    ax4.set_ylabel(r'ratio $R_{\rm diss}$')

    for ax in [ax1, ax2, ax3, ax4]:
        if ax != ax4:
            ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', length = 7, width = 1.2)
        ax.tick_params(axis='both', which='minor', length = 3, width = 1)
        ax.set_xlim(0, 1.7)
    ax3.set_xlabel(r't [t$_{\rm fb}$]')
    ax4.set_xlabel(r't [t$_{\rm fb}$]')


### AS IT WAS BEFORE

# Rdiss = np.zeros(len(snaps))
# Eradtot = np.zeros(len(snaps))
# Ldisstot = np.zeros(len(snaps))
# # for the cut in density 
# Rdiss_cut = np.zeros(len(snaps))
# Eradtot_cut = np.zeros(len(snaps))
# Ldisstot_cut = np.zeros(len(snaps))
# for i,snap in enumerate(snaps):
#     print(snap, flush=False) 
#     sys.stdout.flush()
#     path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
#     data = make_tree(path, snap, energy = True)
#     X, Y, Z, vol, den, Rad_den, Ediss_den = \
#         data.X, data.Y, data.Z, data.Vol, data.Den, data.Rad, data.Diss
    # box = np.load(f'{path}/box_{snap}.npy')
    # if int(snap) <= 317:
    #     boxL = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}LowRes/snap_{snap}/box_{snap}.npy')
    #     if int(snap) <= 267:
    #         boxH = np.load(f'/home/martirep/data_pi-rossiem/TDE_data/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiRes/snap_{snap}/box_{snap}.npy')
    #     else:
    #         boxH = box
    # else: 
    #     boxH = box
    #     boxL = box
    # xmin, ymin, zmin = np.max([box[0], boxL[0], boxH[0], -0.75*apo]), np.max([box[1], boxL[1], boxH[1]]), np.max([box[2], boxL[2], boxH[2]])
    # xmax, ymax, zmax = np.min([box[3], boxL[3], boxH[3]]), np.min([box[4], boxL[4], boxH[4]]), np.min([box[5], boxL[5], boxH[5]])
    # cutx = (X > xmin) & (X < xmax)
    # cuty = (Y > ymin) & (Y < ymax)
    # cutz = (Z > zmin) & (Z < zmax)
    # cut_coord = cutx & cuty & cutz
    # X, Y, Z, vol, den, Rad_den, Ediss_den = \
    #     sec.make_slices([X, Y, Z, vol, den, Rad_den, Ediss_den], cut_coord)
        
    # Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    # Ediss = np.abs(Ediss_den) * vol # energy dissipation rate [energy/time] in code units
    # Rdiss[i] = np.sum(Rsph * Ediss) / np.sum(Ediss)
    # Eradtot[i] = np.sum(Rad_den * vol)
    # Ldisstot[i] = np.sum(Ediss)

    # cut = den > 1e-19
    # Rsph_cut, vol_cut, Rad_den_cut, Ediss_den_cut = Rsph[cut], vol[cut], Rad_den[cut], Ediss_den[cut]
    # Ediss_cut = np.abs(Ediss_den_cut) * vol_cut # energy dissipation rate [energy/time] in code units
    # Rdiss_cut[i] = np.sum(Rsph_cut * Ediss_cut) / np.sum(Ediss_cut)
    # Eradtot_cut[i] = np.sum(Rad_den_cut * vol_cut)
    # Ldisstot_cut[i] = np.sum(Ediss_cut)

# with open(f'{abspath}/data/{folder}/Rdiss_{check}cutCoord.txt','a') as file:
#     file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
#     file.write(f'# Rdiss [R_\odot] \n' + ' '.join(map(str, Rdiss)) + '\n')
#     file.write(f'# Total radiation energy [energy in code units] \n' + ' '.join(map(str, Eradtot)) + '\n')
#     file.write(f'# Total dissipation luminosity [energy/time in code units] \n' + ' '.join(map(str, Ldisstot)) + '\n')
#     file.close()

# with open(f'{abspath}/data/{folder}/Rdiss_{check}cutDenCoord.txt','a') as file:
    # file.write(f'# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
    # file.write(f'# Rdiss [R_\odot] \n' + ' '.join(map(str, Rdiss_cut)) + '\n')
    # file.write(f'# Total radiation energy [energy in code units] \n' + ' '.join(map(str, Eradtot_cut)) + '\n')
    # file.write(f'# Total dissipation luminosity [energy/time in code units] \n' + ' '.join(map(str, Ldisstot_cut)) + '\n')
    # file.close()

