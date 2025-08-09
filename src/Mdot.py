""" Compute Mdot fallback and wind"""
import sys
sys.path.append('/Users/paolamartire/shocks/')

from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    compute = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices

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
check = 'NewAMR'
cond_selection = 'B' # if 'B' you put the extra condition on the Bernouilli coeff to select cells

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb'] # semimajor axis of the bound orbit
norm_dMdE = things['E_mb']

radii = np.array([Rt, 0.5*amin, amin])
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Medd = Ledd/prel.c_cgs**2
v_esc = np.sqrt(2*prel.G*Mbh/Rt)
convers_kms = prel.Rsol_cgs * 1e-5/prel.tsol_cgs # it's aorund 400
print(f'escape velocity at Rt: {v_esc*convers_kms} km/s')

#
## FUNCTIONS
#
def f_out_LodatoRossi(M_fb, M_edd):
    f = 2/np.pi * np.arctan(1/7.5 * (M_fb/M_edd-1))
    return f

# MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}_bins.txt')
    max_bin_negative = np.abs(np.min(bins))
    mid_points = (bins[:-1]+bins[1:]) * norm_dMdE/2  # get rid of the normalization
    dMdE_distr = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
   
    mfall = np.zeros(len(tfb_cgs))
    mwind_pos = []
    Vwind_pos = []
    mwind_neg = []
    Vwind_neg = []
    for i, snap in enumerate(snaps):
        print(snap, flush=True)
        if alice:
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
        t = tfb_cgs[i] 
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol) # it'll give it positive
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        if energy-np.abs(bins_tokeep[i_bin]) > 0:
            print(f'You overcome the maximum negative bin ({max_bin_negative}). You required {energy}')
            continue
        
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mfall[i] = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)

        data = make_tree(path, snap, energy = True)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE
        dim_cell = Vol**(1/3)
        cut = Den > 1e-19
        X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den= \
            make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den], cut)
        IE_spec = IE_den/Den
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        orb_en = orb.orbital_energy(Rsph, V, Mass, params, prel.G) 
        bern = orb_en/Mass + IE_spec + Press/Den
        long = np.arctan2(Y, X)          # Azimuthal angle in radians
        lat = np.arccos(Z / Rsph)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, lat, long)
        # Positive velocity (and unbound)
        if cond_selection == 'B':
            cond = np.logical_and(v_rad >= 0, np.logical_and(bern > 0, X > amin))
        elif cond_selection == '':
            cond = v_rad >= 0  
        X_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([X, Den, Rsph, v_rad, dim_cell], cond)
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            mwind_pos.append(0)
            Vwind_pos.append(0)
        else:
            Mdot_pos = dim_cell_pos**2 * Den_pos * v_rad_pos # there should be a pi factor here, but you put it later
            Mdot_pos_casted = np.zeros(len(radii))
            v_rad_pos_casted = np.zeros(len(radii))
            # print('Mdot_pos: ')
            for j, r in enumerate(radii):
                selected_pos = np.abs(Rsph_pos - r) < dim_cell_pos
                if Mdot_pos[selected_pos].size == 0:
                    Mdot_pos_casted[j] = 0
                    v_rad_pos_casted[j] = 0
                else:
                    Mdot_pos_casted[j] = np.sum(Mdot_pos[selected_pos]) * np.pi 
                    v_rad_pos_casted[j] = np.mean(v_rad_pos[selected_pos])
                    # print('sum of circles/sphere you want: ', np.pi*np.sum(dim_cell_pos[selected_pos]**2)/(4*np.pi*r**2))
            mwind_pos.append(Mdot_pos_casted)
            Vwind_pos.append(v_rad_pos_casted)
        # Negative velocity (and bound)
        if cond_selection == 'B':
            cond = np.logical_and(v_rad < 0, bern <= 0)
        elif cond_selection == '':
            cond = v_rad < 0
        X_neg, Den_neg, Rsph_neg, v_rad_neg, dim_cell_neg = \
            make_slices([X, Den, Rsph, v_rad, dim_cell], cond)
        if Den_neg.size == 0:
            print(f'no bern negative: {bern}', flush=True)
            mwind_pos.append(0)
            Vwind_pos.append(0)
        else:
            Mdot_neg = dim_cell_neg**2 * Den_neg * v_rad_neg # there should be a pi factor here, but you put it later
            Mdot_neg_casted = np.zeros(len(radii))
            v_rad_neg_casted = np.zeros(len(radii))
            # print('Mdot_neg: ')
            for j, r in enumerate(radii):
                selected_neg = np.abs(Rsph_neg - r) < dim_cell_neg
                if Mdot_neg[selected_neg].size == 0:
                    Mdot_neg_casted[j] = 0
                    v_rad_neg_casted[j] = 0
                else:
                    Mdot_neg_casted[j] = np.sum(Mdot_neg[selected_neg]) * np.pi #4 *  * radii**2
                    v_rad_neg_casted[j] = np.mean(v_rad_neg[selected_neg])
                    # print('sum of circles/sphere you want: ', np.pi*np.sum(dim_cell_neg[selected_neg]**2)/(4*np.pi*r**2))
            mwind_neg.append(Mdot_neg_casted)
            Vwind_neg.append(v_rad_neg_casted)

    mwind_pos = np.transpose(np.array(mwind_pos)) # shape pass from len(snap) x len(radii) to len(radii) x len(snap)
    mwind_neg = np.transpose(np.array(mwind_neg))
    Vwind_pos = np.transpose(np.array(Vwind_pos))
    Vwind_neg = np.transpose(np.array(Vwind_neg))

    with open(f'{abspath}/data/{folder}/wind/Mdot_{check}_{cond_selection}pos_equal.txt','w') as file:
        if cond_selection == 'B':
            file.write(f'# Distinguish using Bernouilli criterion \n#t/tfb \n')
        else:
            file.write(f'# t/tfb \n')
        file.write(f' '.join(map(str, tfb)) + '\n')
        file.write(f'# Mdot_f \n')
        file.write(f' '.join(map(str, mfall)) + '\n')
        file.write(f'# Mdot_wind at Rt\n')
        file.write(f' '.join(map(str, mwind_pos[0])) + '\n')
        file.write(f'# Mdot_wind at 0.5 amin\n')
        file.write(f' '.join(map(str, mwind_pos[1])) + '\n')
        file.write(f'# Mdot_wind at amin\n')
        file.write(f' '.join(map(str, mwind_pos[2])) + '\n')
        file.write(f'# v_wind at Rt\n')
        file.write(f' '.join(map(str, Vwind_pos[0])) + '\n')
        file.write(f'# v_wind at 0.5 amin\n')
        file.write(f' '.join(map(str, Vwind_pos[1])) + '\n')
        file.write(f'# v_wind at amin\n')
        file.write(f' '.join(map(str, Vwind_pos[2])) + '\n')
        file.close()
    
    with open(f'{abspath}/data/{folder}/wind/Mdot_{check}_{cond_selection}neg_equal.txt','w') as file:
        if cond_selection == 'B':
            file.write(f'# Distinguish using Bernouilli criterion and X > a_mb \n#t/tfb \n')
        else:
            file.write(f'# t/tfb \n')
        file.write(f' '.join(map(str, tfb)) + '\n')
        file.write(f'# Mdot_wind at Rt\n')
        file.write(f' '.join(map(str, mwind_neg[0])) + '\n')
        file.write(f'# Mdot_wind at 0.5 amin\n')
        file.write(f' '.join(map(str, mwind_neg[1])) + '\n')
        file.write(f'# Mdot_wind at amin\n')
        file.write(f' '.join(map(str, mwind_neg[2])) + '\n')
        file.write(f'# v_wind at Rt\n')
        file.write(f' '.join(map(str, Vwind_neg[0])) + '\n')
        file.write(f'# v_wind at 0.5 amin\n')
        file.write(f' '.join(map(str, Vwind_neg[1])) + '\n')
        file.write(f'# v_wind at amin\n')
        file.write(f' '.join(map(str, Vwind_neg[2])) + '\n')
        file.close()

if plot:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    Medd_code = Medd * prel.tsol_cgs / prel.Msol_cgs  # [g/s]
    tfb, mfall, mwind_pos, mwind_pos1, mwind_pos2, mwind_pos3, Vwind_pos, Vwind_pos1, Vwind_pos2, Vwind_pos3 = \
        np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}_pos.txt')
    _, mwind_neg, mwind_neg1, mwind_neg2, mwind_neg3, Vwind_neg, Vwind_neg1, Vwind_neg2, Vwind_neg3 = \
        np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}_neg.txt')
    tfbB, mfallB, mwind_posB, mwind_posB1, mwind_posB2, mwind_posB3, Vwind_posB, Vwind_posB1, Vwind_posB2, Vwind_posB3 = \
        np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}_Bpos.txt')
    _, mwind_negB, mwind_negB1, mwind_negB2, mwind_negB3, Vwind_negB, Vwind_negB1, Vwind_negB2, Vwind_negB3 = \
        np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}_Bneg.txt')
    f_out_th = f_out_LodatoRossi(mfall, Medd_code)
    # load the data splitted in x>0 or <x0
    splitpos = np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}_pos_splitX.txt')
    pos_xpos1, pos_xneg1 = splitpos[3], splitpos[13]
    splitneg = np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}_neg_splitX.txt')
    neg_xpos1, neg_xneg1 = splitneg[3], splitneg[13]

    fig, ax1 = plt.subplots(1, 1, figsize = (8,7))
    fig2, ax2 = plt.subplots(1, 1, figsize = (8,7))
    ax1.plot(tfb[10:], np.abs(mwind_pos3[10:])/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm out}$')
    ax1.plot(tfb, np.abs(mwind_neg1)/Medd_code, c = 'forestgreen', label = r'$\dot{M}_{\rm in}$')
    ax1.plot(tfb, np.abs(mwind_posB1)/Medd_code, ls = '--', c = 'dodgerblue')#, label = r'$\dot{M}_{\rm out} [B>0]$')
    ax1.plot(tfb, np.abs(mwind_negB1)/Medd_code, ls = '--', c = 'forestgreen')#, label = r'$\dot{M}_{\rm in} [B>0]$')
    ax1.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm fb}$', c = 'k')
    ax1.plot(tfb[-40:], 6e4*np.array(tfb[-40:])**(-5/3), ls = 'dotted', c = 'k')
    ax1.text(1.28, 1.5e4, r'$\propto t^{-5/3}$', fontsize = 18)
    # ax1.axvline(tfb[np.argmax(np.abs(mfall)/Medd_code)], c = 'k', linestyle = 'dotted')
    # ax1.text(tfb[np.argmax(np.abs(mfall)/Medd_code)]+0.01, 0.1, r'$t_{\dot{M}_{\rm peak}}$', fontsize = 20, rotation = 90)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 6e5)
    ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')    
    ax2.plot(tfb, Vwind_pos1/v_esc, c = 'dodgerblue', label = r'$v_{\rm out}$')
    ax2.plot(tfb, Vwind_neg1/v_esc, c = 'forestgreen', label = r'$v_{\rm in}$')
    ax2.plot(tfb, Vwind_posB1/v_esc, '--', c = 'dodgerblue')#, label = r'$v_{\rm out} [B>0]$')
    ax2.plot(tfb, Vwind_negB1/v_esc, '--', c = 'forestgreen')#, label = r'$v_{\rm in} [B>0]$')
    ax2.set_ylabel(r'$v_{\rm out}/v_{\rm esc}(0.5 a_{\rm mb})$')
    original_ticks = ax2.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.legend(fontsize = 18)
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', width=0.7, length=7)
        ax.tick_params(axis='x', which='minor', width=0.5, length=5)
        ax.set_xlim(0, 1.8)
        ax.grid()
    plt.tight_layout()
    fig.savefig(f'{abspath}/Figs/outflow/Mdot_{check}.pdf', bbox_inches = 'tight')

    fig, ax1 = plt.subplots(1, 1, figsize = (8,6))
    ax1.plot(tfb[10:], np.abs(mwind_pos1[10:])/Medd_code, c = 'dodgerblue')
    ax1.plot(tfb, np.abs(pos_xpos1)/Medd_code, '--', c = 'b', label = r'$x>0$')
    ax1.plot(tfb, np.abs(pos_xneg1)/Medd_code, ls = '--', c = 'r', label = r'$x<0$')
    ax1.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm fb}$', c = 'k')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 6e5)
    ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')    
    original_ticks = ax2.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    ax1.set_xlabel(r'$t [t_{\rm fb}]$')
    ax1.legend(fontsize = 18)
    ax1.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='x', which='major', width=0.7, length=7)
    ax1.tick_params(axis='x', which='minor', width=0.5, length=5)
    ax1.set_xlim(0, 1.8)
    ax1.grid()
    plt.title(r'$\dot{M}_{\rm out}$')
    plt.tight_layout()

    # reproduce LodatoRossi11 Fig.6
    plt.figure(figsize = (8,6))
    plt.plot(np.abs(mfall/Medd_code), np.abs(f_out_th), c = 'k')
    # plt.plot(np.abs(mwind_pos1/Medd_code), np.abs(mwind_pos1/mfall), c = 'orange', label = r'f$_{\rm out}$ (0.5$a_{\rm min})$')
    plt.xlim(0, 100)
    plt.legend(fontsize = 14)
    plt.xlabel(r'$\dot{M}_{\rm f} [\dot{M}_{\rm Edd}]$')
    plt.ylabel(r'$f_{\rm out}$')

    plt.figure(figsize = (8,6))
    plt.plot(tfb, np.abs(mwind_pos1/mfall), c = 'orange', label = r'f$_{\rm out}$ (0.5$a_{\rm min})$')
    plt.plot(tfb, np.abs(mwind_pos2/mfall), c = 'purple', label = r'f$_{\rm out}$ (0.7$a_{\rm min})$')
    plt.plot(tfb, np.abs(mwind_pos3/mfall), c = 'green', label = r'f$_{\rm out}$ (1$a_{\rm min})$')
    plt.plot(tfb, np.abs(f_out_th), c = 'k', label = 'LodatoRossi11')
    plt.legend(fontsize = 14)
    plt.xlabel(r't $[t_{\rm fb}]$')
    plt.ylabel(r'$f_{\rm out}\equiv \dot{M}_{\rm wind}/\dot{M}_{\rm fb}$')
    plt.yscale('log')
    plt.ylim(5e-3, 80)

    ## Check convergence
    dataposLow = np.loadtxt(f'{abspath}/data/{folder}LowRes/Mdot_LowRes_pos.txt')
    tfbL,  mwind_pos1L = dataposLow[0], dataposLow[3]
    datanegLow = np.loadtxt(f'{abspath}/data/{folder}LowRes/Mdot_LowRes_neg.txt')
    mwind_neg1L =  datanegLow[2]
    dataposHi = np.loadtxt(f'{abspath}/data/{folder}HiRes/Mdot_HiRes_pos.txt')
    tfbH,  mwind_pos1H = dataposHi[0], dataposHi[3]
    datanegHi = np.loadtxt(f'{abspath}/data/{folder}HiRes/Mdot_HiRes_neg.txt')
    mwind_neg1H = datanegHi[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))
    ax1.plot(tfbL[10:], np.abs(mwind_pos1L[10:])/Medd_code, c = 'C1', label = r'Low')
    ax1.plot(tfb[10:], np.abs(mwind_pos1[10:])/Medd_code, c = 'yellowgreen', label = r'Fid')
    ax1.plot(tfbH[10:], np.abs(mwind_pos1H[10:])/Medd_code, c = 'darkviolet', label = r'High')
    ax1.set_ylabel(r'$|\dot{M}_{\rm out}| [\dot{M}_{\rm Edd}]$')  
    ax2.plot(tfbL, np.abs(mwind_neg1L)/Medd_code, c = 'C1')
    ax2.plot(tfb, np.abs(mwind_neg1)/Medd_code, c = 'yellowgreen')
    ax2.plot(tfbH, np.abs(mwind_neg1H)/Medd_code, c = 'darkviolet')
    ax2.set_ylabel(r'$|\dot{M}_{\rm in}| [\dot{M}_{\rm Edd}]$')  
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.legend(fontsize = 18)
        ax.set_xlim(0, 1.8)
        # ax.set_xticks(new_ticks)
        # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        # ax.set_xticklabels(labels)
        # ax.tick_params(axis='x', which='major', width=0.7, length=7)
        # ax.tick_params(axis='x', which='minor', width=0.5, length=5)
        ax.set_yscale('log')
        ax.set_ylim(1e-1, 6e5)
        ax.grid()
    ax1.legend(fontsize = 18)
    plt.tight_layout()
    fig.savefig(f'{abspath}/Figs/outflow/Mdot_convergence.pdf', bbox_inches = 'tight')
    


    

# %%
