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
check = ''
do_cut = 'cutDen' # '' or 'cutDen'

Rt = Rstar * (Mbh/mstar)**(1/3)
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 

    Ldisstot_pos = np.zeros(len(snaps))
    Rdiss_pos = np.zeros(len(snaps))
    Ldisstot_neg = np.zeros(len(snaps))
    Rdiss_neg = np.zeros(len(snaps))
    for i,snap in enumerate(snaps):
        print(snap, flush=False) 
        sys.stdout.flush()
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
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

    with open(f'{abspath}/data/{folder}/Rdiss_{check}{do_cut}.txt','a') as file:
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
    data = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}cutDen.txt', comments = '#')
    tfb, Rdiss_pos, Ldisstot_pos, Rdiss_neg, Ldisstot_neg = data[0], data[1], data[2], data[3], data[4]
    datanocut = np.loadtxt(f'{abspath}/data/{folder}/convergence/Rdiss_{check}.txt')
    timenocut, Rnocut, Eradtotnocut, LDissnocut = datanocut[0], datanocut[1], datanocut[2], datanocut[3] 
    datacutCoord = np.loadtxt(f'{abspath}/data/{folder}/convergence/Rdiss_{check}cutCoord.txt')
    timecutCoord, RcutCoord, EradtotcutCoord, LDisscutCoord = datacutCoord[0], datacutCoord[1], datacutCoord[2], datacutCoord[3] 
    datacutDenCoord = np.loadtxt(f'{abspath}/data/{folder}/convergence/Rdiss_{check}cutDenCoord.txt')
    timecutDenCoord, RcutDenCoord, EradtotcutDenCoord, LDisscutDenCoord = datacutDenCoord[0], datacutDenCoord[1], datacutDenCoord[2], datacutCoord[3] 
    
    plt.figure()
    plt.plot(tfb, Ldisstot_pos, label = 'L positive, cut den', color = 'blue')
    plt.plot(tfb, np.abs(Ldisstot_neg), label = r'$|$L negative, cut den$|$', color = 'purple')
    plt.plot(timenocut, LDissnocut, label = 'L tot, no cut', color = 'red')
    plt.plot(timecutCoord, LDisscutCoord, label = 'L tot, cutCoord ', color = 'green')
    plt.plot(timecutDenCoord, LDisscutDenCoord, label = 'L tot, cutDenCoord', color = 'orange', ls = '--')
    plt.yscale('log')
    plt.xlabel(r't [t$_{\rm fb}$]')
    plt.ylabel('Dissipation rate [energy/time]')
    plt.legend()
    plt.savefig(f'{abspath}/Figs/{folder}/TestCuts_LDiss.png')

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

