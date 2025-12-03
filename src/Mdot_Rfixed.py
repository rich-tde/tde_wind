""" Compute Mdot fallback and wind across a symmetrical (eventually fixed) surface"""
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
import csv
import os
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, to_spherical_components
from Utilities.selectors_for_snap import select_snap
from Utilities.sections import make_slices

##
# PARAMETERS
#%%
m = 4
Mbh = 10**4
Mbh_cgs = Mbh * prel.Msol_cgs
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
statist = 'mean' # '' for mean, 'median' for median

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb'] # semimajor axis of the bound orbit
# print(0.5*amin/Rt)
norm_dMdE = things['E_mb']
max_Mdot = mstar*prel.Msol_cgs/(3*tfallback_cgs) # in code units

Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
convers_kms = prel.Rsol_cgs * 1e-5/prel.tsol_cgs # it's aorund 400

#%%
# MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
    r_chosen = 0.5*amin # 'Rtr' for radius of the trap, value that you want for a fixed value
    which_r_title = '05amin'

    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}_bins.txt')
    max_bin_negative = np.abs(np.min(bins))
    mid_points = (bins[:-1]+bins[1:]) * norm_dMdE/2  # get rid of the normalization
    dMdE_distr = np.loadtxt(f'{abspath}/data/{folder}/wind/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
   
    for i, snap in enumerate(snaps):
        print(snap, flush=True)
        if alice:
            path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        else:
            path = f'/Users/paolamartire/shocks/TDE/{folder}/{snap}'
        t = tfb_cgs[i] 

        # Compute dot{M}_fb
        # convert to code units
        tsol = t / prel.tsol_cgs
        # Find the energy of the element at time t
        energy = orb.keplerian_energy(Mbh, prel.G, tsol) # it'll give it positive
        i_bin = np.argmin(np.abs(energy-np.abs(bins_tokeep))) # just to be sure that you match the data
        if energy-max_bin_negative*norm_dMdE > 0:
            print(f'You overcome the maximum negative bin ({max_bin_negative*norm_dMdE}). You required {energy}')
            continue
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mfall = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)

        # Compute \dot{M}_w
        # Load data and pick the ones unbound and with positive velocity
        data = make_tree(path, snap, energy = True)
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Press, data.VX, data.VY, data.VZ, data.IE, data.Rad
        cut = Den > 1e-19
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den = \
            make_slices([X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den, Rad_den], cut)
        dim_cell = Vol**(1/3)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        bern = orb.bern_coeff(Rsph, V, Den, Mass, Press, IE_den, Rad_den, params)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
        IE = IE_den * Vol
        Rad = Rad_den * Vol
        cond_wind = np.logical_and(v_rad >= 0, bern > 0)
        X_pos, Y_pos, Z_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos, IE_pos, Rad_pos = \
            make_slices([X, Y, Z, Den, Rsph, v_rad, dim_cell, IE, Rad], cond_wind)
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            data = [snap, tfb[i], mfall, 0, 0, 0, 0, 0, 0, 0]

        else: 
            tot_IE = np.sum(IE_pos)
            tot_Rad = np.sum(Rad_pos)
            Mdot_dimCell = np.pi * dim_cell_pos**2 * Den_pos * v_rad_pos  
            Mdot_R = 4 * np.pi * r_chosen**2 * Den_pos * v_rad_pos 
            condRtr = np.abs(Rsph_pos-r_chosen) < dim_cell_pos
            # Pick the cells at r_chosen
            Mdot_dimCell_casted = Mdot_dimCell[condRtr] 
            Mdot_R_casted = Mdot_R[condRtr]
            v_rad_pos_casted = v_rad_pos[condRtr] 

            mwind_dimCell = 4 * r_chosen**2 * np.sum(Mdot_dimCell_casted) / np.sum(dim_cell_pos[condRtr]**2)
            if statist == 'mean':
                mwind_R = np.mean(Mdot_R_casted) # NB: this is an overestimate since you're doing the mean already on the positive ones, not all the cells at radius R
                mwind_R_nonzero = np.mean(Mdot_R_casted[Mdot_R_casted!=0]) # NB: if the radius is fixed, it's the same as the mean on all. 
                Vwind = np.mean(v_rad_pos_casted)
                Vwind_nonzero = np.mean(v_rad_pos_casted[v_rad_pos_casted!=0])
            elif statist == 'median':
                mwind_R = np.median(Mdot_R_casted) 
                mwind_R_nonzero = np.median(Mdot_R_casted[Mdot_R_casted!=0])
                Vwind = np.median(v_rad_pos_casted)
                Vwind_nonzero = np.median(v_rad_pos_casted[v_rad_pos_casted!=0])
        
            data = [snap, tfb[i], mfall, mwind_dimCell, mwind_R, mwind_R_nonzero, Vwind, Vwind_nonzero, tot_IE, tot_Rad]
    
        csv_path = f'{abspath}/data/{folder}/wind/Mdot_{check}{which_r_title}{statist}.csv'
        if alice:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    writer.writerow(['snap', ' tfb', ' Mdot_fb', 'normalized Mw with dimCell', f'Mw with {which_r_title}', f'Mw with {which_r_title} (nonzero)', 'Vwind', 'Vwind (nonzero)', 'tot_IE', 'tot_Rad'])
                writer.writerow(data)
            file.close()

if plot:
    # from scipy.integrate import cumulative_trapezoid
    from plotting.paper.IHopeIsTheLast import ratio_BigOverSmall
    from Utilities.operators import sort_list
    import matplotlib.colors as mcolors
    which_r_title = '05amin'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    checks = ['LowResNewAMR', 'NewAMR', 'HiResNewAMR']
    checks_label = ['Low', 'Middle', 'High']    

    fig, ax1 = plt.subplots(1, 1, figsize = (10, 6))
    figCon, (axCon, axerr) = plt.subplots(2, 1, figsize = (9, 9), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)

    dataL = np.loadtxt(f'{abspath}/data/{folder}LowResNewAMR/LowResNewAMR_red.csv', delimiter=',', dtype=float)
    tfbsL_lum, LumsL = dataL[:, 1], dataL[:, 2]
    LumsL, tfbsL_lum = sort_list([LumsL, tfbsL_lum], tfbsL_lum, unique=True)
    tfbL_max = tfbsL_lum[np.argmax(LumsL)]
    _, tfbL, mfallL, mwind_dimCellL, mwind_RL, mwind_R_nonzeroL, _, _, _, _ = \
            np.loadtxt(f'{abspath}/data/{folder}LowResNewAMR/wind/Mdot_LowResNewAMR{which_r_title}{statist}.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True)
    MdotLmax = mwind_dimCellL[np.argmin(np.abs(tfbL - tfbL_max))]

    dataM = np.loadtxt(f'{abspath}/data/{folder}NewAMR/NewAMR_red.csv', delimiter=',', dtype=float)
    tfbsM_lum, LumsM = dataM[:, 1], dataM[:, 2]
    LumsM, tfbsM_lum = sort_list([LumsM, tfbsM_lum], tfbsM_lum, unique=True)
    tfbM_max = tfbsM_lum[np.argmax(LumsM)]
    _, tfbM, mfallM, mwind_dimCellM, mwind_RM, mwind_R_nonzeroM, _, _, _, _ = \
            np.loadtxt(f'{abspath}/data/{folder}NewAMR/wind/Mdot_NewAMR{which_r_title}{statist}.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True)
    MdotMmax = mwind_dimCellM[np.argmin(np.abs(tfbM - tfbM_max))]
    tfb_ratioL, ratioL, rel_errL  = ratio_BigOverSmall(tfbM, mwind_dimCellM, tfbL, mwind_dimCellL)

    dataH = np.loadtxt(f'{abspath}/data/{folder}HiResNewAMR/HiResNewAMR_red.csv', delimiter=',', dtype=float)
    tfbsH_lum, LumsH = dataH[:, 1], dataH[:, 2]
    LumsH, tfbsH_lum = sort_list([LumsH, tfbsH_lum], tfbsH_lum, unique=True)
    tfbH_max = tfbsH_lum[np.argmax(LumsH)]
    _, tfbH, mfallH, mwind_dimCellH, mwind_RH, mwind_R_nonzeroH, _, _, tot_IE_H, tot_Rad_H = \
            np.loadtxt(f'{abspath}/data/{folder}HiResNewAMR/wind/Mdot_HiResNewAMR{which_r_title}{statist}.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True) 
    MdotHmax = mwind_dimCellH[np.argmin(np.abs(tfbH - tfbH_max))]
    tfb_ratioH, ratioH, rel_errH  = ratio_BigOverSmall(tfbM, mwind_RM, tfbH, mwind_RH)
    data_E = np.loadtxt(f'{abspath}/data/{folder}HiResNewAMR/convE_{check}.csv', delimiter=',', dtype=float, skiprows=1)    
    # tfb_E, IE, Rad = data_E[:, 1], data_E[:, 2], data_E[:, 5]
    ratio_RadIE = tot_Rad_H/tot_IE_H #Rad/IE
    # not the best way to do it, but Mdot starts later than energies
    # ratio_RadIE = np.array(ratio_RadIE[len(ratio_RadIE)-len(mwind_dimCellH):])
    print('final ratios Rad/IE:', ratio_RadIE[-10:])

    print('Naive estimate L:', 0.1 * np.max(np.abs(mfallH))* prel.Msol_cgs/prel.tsol_cgs * prel.c_cgs**2)

    # integrate mwind_dimCell in tfb 
    # mwind_dimCell_int = cumulative_trapezoid(np.abs(mwind_dimCell), tfb, initial = 0)
    # mfall_int = cumulative_trapezoid(np.abs(mfall), tfb, initial = 0)
    # print(f'integral of Mw at the last time: {mwind_dimCell_int[-1]/mstar} Mstar')
    # print(f'integral of Mfb at the last time: {mfall_int[-1]/mstar} Mstar')
    # print(f'End of simualation, Mw/Mfb in {check}:', np.abs(mwind_dimCell[-1]/mfall[-1]))
    
    ax1.plot(tfbH, np.abs(mfallH)/Medd_sol, ls = '--', c = 'k', label = r'$\dot{M}_{\rm fb}$')
    img = ax1.scatter(tfbH, np.abs(mwind_dimCellH)/Medd_sol, c = ratio_RadIE, cmap = 'PuOr', edgecolors = 'gray', norm = colors.LogNorm(vmin=3e-2, vmax=5e1) ,label = r'$\dot{M}_{\rm w}$')
    cbar = fig.colorbar(img, ax = ax1)
    cbar.set_label(r'$E_{\rm rad}/E_{\rm th}$')
    cbar.ax.tick_params(which = 'major', length=8, width=0.9)
    cbar.ax.tick_params(which = 'minor', length=5, width=0.7)
    ax1.axvline(tfbH_max, ls = ':', c = 'gray')
    ax1.text(0.011+tfbH_max, 20, r'$t=t_{\rm p}$', rotation = 90, fontsize = 20) 

    axCon.plot(tfbL, np.abs(mwind_dimCellL)/Medd_sol, c = 'C1', label = 'Low') 
    axCon.plot(tfbM, np.abs(mwind_dimCellM)/Medd_sol, c = 'yellowgreen', label = 'Middle') 
    axCon.plot(tfbH, np.abs(mwind_dimCellH)/Medd_sol, c = 'darkviolet', label = 'High') 
    axCon.scatter([tfbL_max, tfbM_max, tfbH_max], np.array([MdotLmax, MdotMmax, MdotHmax])/Medd_sol, c = ['C1', 'yellowgreen', 'darkviolet'], s = 90, marker = 'd')
    axerr.plot(tfb_ratioL, ratioL, c = 'C1')
    axerr.plot(tfb_ratioL, ratioL, c = 'yellowgreen', ls = (0, (5, 10)))
    axerr.plot(tfb_ratioH, ratioH, c = 'darkviolet')
    axerr.plot(tfb_ratioH, ratioH, c = 'yellowgreen', ls = (0, (5, 10)))
    
    original_ticks = ax1.get_xticks()
    for ax in (axCon, ax1, axerr):
        if ax != axerr:
            ax.set_yscale('log')
            ax.set_ylim(10, 9e6)
            ax.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')   
        else:
            ax.set_ylim(0.9, 4)
            ax.set_ylabel(r'$\mathcal{R}$')

        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
        new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
        ax.set_xticks(new_ticks)
        # if ax == axCon:
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
        ax.set_xticklabels(labels)  
        if ax != axCon:  
            ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.tick_params(axis='both', which='major', width=1.2, length=9)
        ax.tick_params(axis='both', which='minor', width=1, length=5)
        ax.legend(fontsize = 20)
        ax.grid()
    axCon.set_xlim(np.min(tfbM), np.max(tfbM))
    ax1.set_xlim(np.min(tfbH), np.max(tfbH))

    fig.tight_layout()
    figCon.tight_layout()
    fig.savefig(f'{abspath}/Figs/paper/Mw.pdf', bbox_inches = 'tight')
    figCon.savefig(f'{abspath}/Figs/paper/Mw_conv.pdf', bbox_inches = 'tight')

    fig, ax = plt.subplots(1,1, figsize = (8,6))
    ax.plot(tfbH, np.abs(mwind_dimCellH/mfallH), c = 'k')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t [t_{\rm fb}]$')
    ax.set_ylabel(r'$|\dot{M}_{\rm w}/\dot{M}_{\rm fb}|$')
    original_ticks = ax.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    ax.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]    
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', width=1.2, length=9)
    ax.tick_params(axis='both', which='minor', width=1, length=5)
    ax.set_ylim(1e-2, 1)
    ax.set_xlim(np.min(tfbH), np.max(tfbH))
    ax.grid()
    fig.tight_layout()


    
# %%
print('naive L from dotM_fb: ', 0.1 * np.max(np.abs(mfallH)) * prel.Msol_cgs/prel.tsol_cgs * prel.c_cgs**2)
# %% compute constant wind
dataDiss = np.loadtxt(f'{abspath}/data/{folder}HiResNewAMR/Rdiss_HiResNewAMR.csv', delimiter=',', dtype=float, skiprows=1)
timeRDiss, RDiss = dataDiss[:,1], dataDiss[:,2] 
print('Rdiss at max lum', RDiss[np.argmax(LumsH)]/Rp, ' Rp')
zeta = np.abs(mwind_dimCellH[np.argmax(LumsH)]/mfallH[np.argmax(LumsH)]) #7e4/2e6 
A_w_cgs = np.sqrt(prel.G_cgs) * (np.sqrt(2) / (3*np.pi))**(1/3) * (prel.Msol_cgs*1e4)**(1/18) * (prel.Msol_cgs)**(7/9) * (prel.Rsol_cgs)**(-5/6)
print(f'constant A wind', 1e-15*A_w_cgs, '1e15') 
print('zeta:', zeta)
print('Ledd^1/3, :', 1e-13*Ledd_cgs**(1/3), '1e13')
print('ratio, :', A_w_cgs/Ledd_cgs**(1/3))

def Ltr_an(Mbh, mstar, Rstar, beta, t_over_tfb, zeta):
    A = A_w_cgs #1e15
    Ledd_sol_T, _ = orb.Edd(Mbh, 0.34/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
    Ledd_cgs_T = Ledd_sol_T * prel.en_converter/prel.tsol_cgs
    Ltr = A * zeta**(1/3) * Ledd_cgs_T**(2/3) * beta**(1/3) * (Mbh/1e4)**(1/18) * (mstar)**(7/9) * (1/Rstar)**(5/6) * (1/t_over_tfb)**(5/9)
    Ltr_over_Edd = Ltr / Ledd_cgs_T
    return Ltr_over_Edd

pred_1e4 = Ltr_an(Mbh, mstar, Rstar, beta, tfbL_max, zeta)
print('predicted Ltr at tfbL_max tfb (in Ledd) from Eq.16: ', pred_1e4)
print('predicted Ltr at tfbL_max tfb (in Ledd) from Eq.15: ', (Rg*mwind_dimCellH[np.argmax(LumsH)]/(Rp*Medd_sol))**(1/3))
# %%
pred_1e6 = Ltr_an(1e6, mstar, Rstar, beta, tfbL_max, zeta)
print('predicted Ltr (in Ledd) from Eq.16 for a WD: ', pred_1e6)
#%%
pred_WD = Ltr_an(Mbh, mstar, 0.01, beta, tfbL_max, zeta)
print('predicted Ltr (in Ledd) from Eq.16 for a 1e6BH: ', pred_WD)
# %%
 