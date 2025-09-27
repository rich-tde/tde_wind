""" If alice: Compute and save the unbound mass and orbital energy. Compute also the dMdE.
If local: plots"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    path = '/home/martirep/data_pi-rossiem/TDE_data'
else:
    abspath = '/Users/paolamartire/shocks'
    path = f'{abspath}/TDE'
import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities.operators import make_tree
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices
import src.orbits as orb
import csv
import os

#
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
check = 'NewAMR'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

#%%
# MAIN
##
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
norm = things['E_mb']
amin = things['a_mb'] # semimajor axis of the bound orbit

if alice:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    bins = np.arange(0, 2, .01) #np.concatenate((bins1, bins2, bins3))
    with open(f'{abspath}/data/{folder}/wind/unbounddMdE_{check}_days.txt','w') as filedays:
        filedays.write(f'# {folder}_{check} \n# Snaps \n' + ' '.join(map(str, snaps)) + '\n')
        filedays.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        filedays.close()
    with open(f'{abspath}/data/{folder}/wind/unbounddMdE_{check}_bins.txt','w') as file:
        file.write(f'# Energy bins normalised (by DeltaE = {norm}) \n')
        file.write((' '.join(map(str, bins)) + '\n'))
        file.close()

    # compute the unbound mass for all the snapshots
    for i, snap in enumerate(snaps):
        print(snap, flush = True)
        pathfold = f'{path}/{folder}/snap_{snap}'
        data = make_tree(pathfold, snap, energy = True)
        X, Y, Z, mass, den, Press, vx, vy, vz, IE_den, Rad_den = \
            data.X, data.Y, data.Z, data.Mass, data.Den, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        cut = den > 1e-19
        X, Y, Z, mass, den, Press, vx, vy, vz, IE_den, Rad_den = \
            make_slices([X, Y, Z, mass, den, Press, vx, vy, vz, IE_den, Rad_den], cut)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        vel = np.sqrt(vx**2 + vy**2 + vz**2)
        orb_en = orb.orbital_energy(Rsph, vel, mass, params, prel.G)
        Mass_dynunboundOE = np.sum(mass[orb_en > 0]) #- Mass_dynunboundOE
        OE_dynunboundOE = np.sum(orb_en[orb_en > 0])
        # Mass_dynunboundOE_frombound = mstar - np.sum(mass[orb_en < 0]) #- Mass_dynunboundOE_frombound
        bern = orb.bern_coeff(Rsph, vel, den, mass, Press, IE_den, Rad_den, params)
        Mass_unbound = np.sum(mass[bern > 0]) #- Mass_dynunboundbern
        OE_unbound = np.sum(orb_en[bern > 0])
        # Mass_unbound_frombound = mstar - np.sum(mass[bern < 0]) #- Mass_dynunboundbern_frombound
        Mass_unbound_exclAmin = np.sum(mass[np.logical_and(bern > 0, X>-amin)]) #- Mass_dynunboundbern
        OE_unbound_exclAmin = np.sum(orb_en[np.logical_and(bern > 0, X>-amin)])

        csv_path = f'{abspath}/data/{folder}/wind/Mass_unbound{check}.csv'
        with open(csv_path,'a', newline='') as file:
            writer = csv.writer(file)
            if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                writer.writerow(['snap', ' tfb', ' Mass unbound (bern > 0)', ' OE unbound (bern > 0)', ' Mass dynunbound (OE > 0)', ' OE dynunbound (OE > 0)', ' Mass unbound (bern > 0, X>-amin)', ' OE unbound (bern > 0, X>-amin)'])
            writer.writerow([snap, tfb[i], Mass_unbound, OE_unbound, Mass_dynunboundOE, OE_dynunboundOE, Mass_unbound_exclAmin, OE_unbound_exclAmin])
            file.close()
        
        # compute dMdE
        # specific_OE = orb_en / mass
        # specOE_norm = specific_OE/norm 
        # mass_binned, bins_edges = np.histogram(specOE_norm, bins = bins, weights=mass) # sum the mass in each bin (bins done on specOE_norm)
        # dm_dE = mass_binned / (np.diff(bins_edges)*norm)

        # with open(f'{abspath}/data/{folder}/wind/unbounddMdE_{check}.txt','a') as file:
        #     file.write(f'# dM/dE [code units] snap {snap} \n')
        #     file.write((' '.join(map(str, dm_dE)) + '\n'))
        #     file.close()

#%%
if plot:
    # among resolutions
    commonfold = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    _, tfbL, M_bernL, OE_bernL, M_OEL, OE_OEL, M_bernL_exclAmin, OE_bernL_exclAmin = np.loadtxt(f'{abspath}/data/{commonfold}LowResNewAMR/wind/Mass_unboundLowResNewAMR.csv', delimiter = ',', skiprows = 1, unpack=True)
    _, tfb, M_bern, OE_bern, M_OE, OE_OE, M_bern_exclAmin, OE_bern_exclAmin  = np.loadtxt(f'{abspath}/data/{commonfold}NewAMR/wind/Mass_unboundNewAMR.csv', delimiter = ',', skiprows = 1, unpack=True)
    _, tfbH, M_bernH, OE_bernH, M_OEH, OE_OEH, M_bernH_exclAmin, OE_bernH_exclAmin = np.loadtxt(f'{abspath}/data/{commonfold}HiResNewAMR/wind/Mass_unboundHiResNewAMR.csv', delimiter = ',', skiprows = 1, unpack=True)
    
    # compare dyn with B
    # fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    # ax1.plot(tfbL, M_bernL/mstar, c = 'C1', label = r'Low with B')
    # ax1.plot(tfb, M_bern/mstar,  c = 'yellowgreen', label = r'Fid with B')
    # ax1.plot(tfbH, M_bernH/mstar, c = 'darkviolet', label = r'High with B')
    # ax1.plot(tfbL, M_OEL/mstar, ls = '--', c = 'r', label = r'Low with B')
    # ax1.plot(tfb, M_OE/mstar,  ls = '--', c = 'forestgreen', label = r'Fid with B')
    # ax1.plot(tfbH, M_OEH/mstar, ls = '--', c = 'orchid', label = r'High with B')
    # ax1.set_ylabel(r'Unbound mass [$M_\star$]')
    # ax1.legend(fontsize = 15)
    # ax1.set_yscale('log')
    # ax1.set_xlabel(r'$t [t_{\rm fb}]$')
    # ax1.grid()
    # ax1.tick_params(axis='both', which='major', width=1, length=7)
    # ax1.tick_params(axis='both', which='minor', width=.8, length=4)
    # plt.tight_layout()
    # plt.savefig(f'{abspath}/Figs/Test/Mass_unbound_cond.png', dpi = 300, bbox_inches='tight')
    print('Relative error Low-Fid', np.max((M_bernL[:len(M_bern)])/(M_bern)))
    print('Relative error Fid-High', np.max((M_bernH)/(M_bern[:len(M_bernH)])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    # ax1.plot(tfbL, (M_bernL-M_OEL[0])/mstar, c = 'C1', label = r'Low')
    # ax1.plot(tfb, (M_bern-M_OE[0])/mstar,  c = 'yellowgreen', label = r'Fid')
    # ax1.plot(tfbH, (M_bernH-M_OEH[0])/mstar, c = 'darkviolet', label = r'High')
    # ax1.set_ylabel(r'Mass [$M_\star$] after disruption')
    ax1.plot(tfbL, M_bernL/mstar, c = 'C1', label = r'Low')
    ax1.plot(tfb, M_bern/mstar,  c = 'yellowgreen', label = r'Fid')
    ax1.plot(tfbH, M_bernH/mstar, c = 'darkviolet', label = r'High')
    ax1.set_ylabel(r'Unbound mass [$M_\star$]')
    ax1.legend(fontsize = 15) 
    ax2.set_title(r'Material in all the simulation volume') 
    ax1.set_title(r'All volume', fontsize = 16) 
    
    ax2.plot(tfbL, M_bernL_exclAmin/mstar, c = 'C1', label = r'Low')
    ax2.plot(tfb, M_bern_exclAmin/mstar,  c = 'yellowgreen', label = r'Fid')
    ax2.plot(tfbH, M_bernH_exclAmin/mstar, c = 'darkviolet', label = r'High')  
    ax2.set_title(r'Material with $X>-a_{\rm min}$', fontsize = 16) 
    ax2.set_yscale('log')
    original_xticks = ax1.get_xticks()
    middle_ticks = (original_xticks[1:]+original_xticks[:-1])/2
    ticks = np.concatenate((original_xticks, middle_ticks))
    for ax in [ax1, ax2]:
        ax.set_xticks(ticks)
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.grid()
        ax.tick_params(axis='both', which='major', width=1, length=7)
        ax.tick_params(axis='both', which='minor', width=.8, length=4)
        ax.set_xlim(0,1.8)
    # plt.savefig(f'{abspath}/Figs/multiple/Mass_unbound.png', dpi = 300, bbox_inches='tight')
    plt.suptitle(r'Unbound material', fontsize = 16)
    plt.tight_layout()
    plt.show()


    datadays = np.loadtxt(f'{abspath}/data/{folder}/wind/unbounddMdE_{check}_days.txt')
    snaps, tfb= datadays[0], datadays[1]
    bins = np.loadtxt(f'{abspath}/data/{folder}/wind/unbounddMdE_{check}_bins.txt')
    data = np.loadtxt(f'{abspath}/data/{folder}/wind/unbounddMdE_{check}.txt')
    mid_points = (bins[:-1]+bins[1:])/2
    plt.figure()
    for i in range(len((data))):
        if i!=0 and i !=len(data)-1:
            continue
        plt.plot(mid_points, data[i], alpha = 0.8, label = f't/tfb = {np.round(tfb[i],2)}')
    plt.xlabel(r'$\log_{10}E/\Delta E$')
    plt.ylabel('dM/dE')
    plt.yscale('log')
    plt.legend(fontsize = 16)
    plt.xlim(0,2.5)
    plt.title(f'{check}', fontsize = 16)
    plt.tight_layout()
    plt.show()


    
# %%
