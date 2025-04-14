""" Search for bound mass"""

import sys
sys.path.append('/Users/paolamartire/shocks')
from Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.sections import make_slices
from Utilities.selectors_for_snap import select_snap
import src.orbits as orb
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' 
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
color_snap = ['darkviolet', 'dodgerblue']
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

# plt.figure(figsize = (8, 6))
OE_pos_tot = np.zeros(len(snaps))
OE_neg_tot = np.zeros(len(snaps))
OE_spec_pos_tot = np.zeros(len(snaps))
OE_spec_neg_tot = np.zeros(len(snaps))
for i, snap in enumerate(snaps):
    path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
    data = make_tree(path, snap, energy = True)
    X, Y, Z, P, Mass, Den, VX, VY, VZ, IE = \
        data.X, data.Y, data.Z, data.Press, data.Mass, data.Den, data.VX, data.VY, data.VZ, data.IE
    cut = Den > 1e-19
    X, Y, Z, P, Mass, Den, VX, VY, VZ, IE = \
        make_slices([X, Y, Z, P, Mass, Den, VX, VY, VZ, IE], cut)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    orbital_enegy = orb.orbital_energy(R, V, Mass, prel.G, prel.csol_cgs, Mbh)
    OE_pos_tot[i] = np.sum(orbital_enegy[orbital_enegy>0])
    OE_neg_tot[i] = np.sum(orbital_enegy[orbital_enegy<0])
    OE_spec = orbital_enegy/Mass
    OE_spec_pos_tot[i] = np.sum(OE_spec[OE_spec>0])
    OE_spec_neg_tot[i] = np.sum(OE_spec[OE_spec<0])

with open(f'{abspath}/data/{folder}/OE_tot.txt', 'a') as file:
    file.write(f'# t/tfb \n')
    file.write(f' '.join(map(str, tfb)) + '\n')
    file.write(f'# total specific POSITIVE orbital energy [code units] \n')
    file.write(f' '.join(map(str, OE_spec_pos_tot)) + '\n')
    file.write(f'# total specific NEGATIVE orbital energy [code units] \n')
    file.write(f' '.join(map(str, OE_spec_neg_tot)) + '\n')
    file.write(f'# total POSITIVE orbital energy [code units] \n')
    file.write(f' '.join(map(str, OE_pos_tot)) + '\n')
    file.write(f'# total NEGATIVE orbital energy [code units] \n')
    file.write(f' '.join(map(str, OE_neg_tot)) + '\n')
    file.close()


# plt.figure(figsize = (8, 6))
# OE_tot = np.zeros(len(snaps))
# inner_tot = np.zeros(len(snaps))
# B_tot = np.zeros(len(snaps))
# Pden_inner_mean = np.zeros(len(snaps))
# IE_inner_mean = np.zeros(len(snaps))
# for i, snap in enumerate(snaps):
#     path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
#     data = make_tree(path, snap, energy = True)
#     X, Y, Z, P, Mass, Den, VX, VY, VZ, IE = \
#         data.X, data.Y, data.Z, data.Press, data.Mass, data.Den, data.VX, data.VY, data.VZ, data.IE
#     cut = Den > 1e-19
#     X, Y, Z, P, Mass, Den, VX, VY, VZ, IE = \
#         make_slices([X, Y, Z, P, Mass, Den, VX, VY, VZ, IE], cut)
#     R = np.sqrt(X**2 + Y**2 + Z**2)
#     V = np.sqrt(VX**2 + VY**2 + VZ**2)
#     orbital_enegy = orb.orbital_energy(R, V, Mass, prel.G, prel.csol_cgs, Mbh)
#     OE_spec = orbital_enegy/Mass
#     B = orbital_enegy + IE + P/Den
#     inner_cond = np.logical_and(OE_spec<0, IE+P/Den>np.abs(OE_spec))
#     OE_tot[i] = np.sum(Mass[OE_spec>0])
#     inner_tot[i] = np.sum(Mass[inner_cond])
#     Pden_inner_mean[i] = np.mean(P[inner_cond]/Den[inner_cond])
#     IE_inner_mean[i] = np.mean(IE[inner_cond])

#     B_tot[i] = np.sum(Mass[B>0])
# #%%
# plt.figure()
# plt.plot(snaps, OE_tot/mstar, label = r'OE$>$0')
# plt.plot(snaps, inner_tot/mstar, label = r'OE$<0$, IE+P$\rho^{-1}>|OE|$')
# # plt.plot(snaps, B_tot/mstar, label = r'B$>$0')
# plt.yscale('log')
# plt.xlabel('Snap')
# plt.ylabel(r'$M [M_\star$]')
# plt.legend(fontsize = 16)

# # %%
# plt.figure()
# plt.plot(snaps, Pden_inner_mean, label = r'$\langle P/\rho \rangle$')
# plt.plot(snaps, IE_inner_mean, label = r'$\langle IE \rangle$')
# plt.yscale('log')
# plt.xlabel('Snap')
# plt.ylabel(r'$\langle \cdots \rangle$')
# plt.legend(fontsize = 16)
# # %%
