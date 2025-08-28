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
check = 'NewAMR'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
tfallback = things['t_fb_days']
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
amin = things['a_mb'] # semimajor axis of the bound orbit
norm_dMdE = things['E_mb'] 
t_fb_days_cgs = things['t_fb_days'] * 24 * 3600 # in seconds
max_Mdot = mstar*prel.Msol_cgs/(3*t_fb_days_cgs) # in code units

radii = np.array([Rt, 0.5*amin, amin, 50*Rt])
radii_names = [f'Rt', f'0.5 a_mb', f'a_mb', f'50 R_t']
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Medd = Ledd/(0.1*prel.c_cgs**2)
v_esc = np.sqrt(2*prel.G*Mbh/Rp)
convers_kms = prel.Rsol_cgs * 1e-5/prel.tsol_cgs # it's aorund 400
print(f'escape velocity at Rp: {v_esc/prel.csol_cgs} c')
# print(np.log10(max_Mdot/Medd), 'Mdot max in Eddington units')

#
## FUNCTIONS
#
def Mpeak_intfb(params):
    # Mpeak in cgs
    Mbh, Rstar, mstar, beta = params
    things = orb.get_things_about(params)
    tfallback = things['t_fb_days']
    tfballback_cgs = tfallback * 24 * 3600 # in seconds
    Mpeak = mstar*prel.Msol_cgs/(3*tfballback_cgs)
    return Mpeak

def Mdot_theory(params, t_over_tmin):
    Mpeak = Mpeak_intfb(params)
    Mdot = Mpeak * (t_over_tmin)**(-5/3)
    return Mdot

def f_out_LodatoRossi(M_fb, M_edd):
    f = 2/np.pi * np.arctan(1/7.5 * (M_fb/M_edd-1))
    return f

def t_Edd(params):
    Mbh, Rstar, mstar, beta = params
    things = orb.get_things_about(params)
    tfallback = things['t_fb_days']
    t_ed = 0.1 * (Mbh * 1e-6)**(2/5) * (Rp/(3*Rs))**(6/5) * mstar**(3/5) * Rstar**(-3/5) # years
    t_ed_days = t_ed * 365 
    return t_ed_days/tfallback

def t_edge(params, f_out, f_v):
    """ Eq.9 in StrubbeQuataert09. Give the time in t_fb """
    Mbh, Rstar, mstar, beta = params
    things = orb.get_things_about(params)
    tfallback = things['t_fb_days']
    Rs = things['Rs']
    Rp = things['Rp']
    t_fb_days_cgs = things['t_fb_days'] * 24 * 3600 # in seconds
    t_ed = f_out**(3/8) * f_v**(-3/4) * (Mbh * 1e-6)**(5/8) * (Rp/(3*Rs))**(9/8) * mstar**(3/8) * Rstar**(-3/8)
    
    return t_ed/tfallback

# MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    tfb_cgs = tfb * t_fb_days_cgs #converted to seconds
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
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, X, Y, Z)
        # Positive velocity (and unbound)
        cond = np.logical_and(v_rad >= 0, np.logical_and(bern > 0, X > -amin))
        X_pos, Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([X, Den, Rsph, v_rad, dim_cell], cond)
        Mdot_pos_casted = np.zeros(len(radii))
        v_rad_pos_casted = np.zeros(len(radii))
        if Den_pos.size == 0:
            print(f'no positive', flush=True)
            continue
        else:
            Mdot_pos = dim_cell_pos**2 * Den_pos * v_rad_pos # there should be a pi factor here, but you put it later
            # print('Mdot_pos: ')
            for j, r in enumerate(radii):
                selected_pos = np.abs(Rsph_pos - r) < dim_cell_pos
                if Mdot_pos[selected_pos].size == 0:
                    continue
                else:
                    Mdot_pos_casted[j] = np.sum(Mdot_pos[selected_pos]) * np.pi 
                    v_rad_pos_casted[j] = np.mean(v_rad_pos[selected_pos])
                    # print('sum of circles/sphere you want: ', np.pi*np.sum(dim_cell_pos[selected_pos]**2)/(4*np.pi*r**2))
        # Negative velocity (and bound)
        cond = np.logical_and(v_rad < 0, bern <= 0)
        X_neg, Den_neg, Rsph_neg, v_rad_neg, dim_cell_neg = \
            make_slices([X, Den, Rsph, v_rad, dim_cell], cond)
        Mdot_neg_casted = np.zeros(len(radii))
        v_rad_neg_casted = np.zeros(len(radii))
        if Den_neg.size == 0:
            print(f'no bern negative: {bern}', flush=True)
            continue
        else:
            Mdot_neg = dim_cell_neg**2 * Den_neg * v_rad_neg # there should be a pi factor here, but you put it later
            # print('Mdot_neg: ')
            for j, r in enumerate(radii):
                selected_neg = np.abs(Rsph_neg - r) < dim_cell_neg
                if Mdot_neg[selected_neg].size == 0:
                    continue
                else:
                    Mdot_neg_casted[j] = np.sum(Mdot_neg[selected_neg]) * np.pi #4 *  * radii**2
                    v_rad_neg_casted[j] = np.mean(v_rad_neg[selected_neg])
                    # print('sum of circles/sphere you want: ', np.pi*np.sum(dim_cell_neg[selected_neg]**2)/(4*np.pi*r**2))

        csv_path = f'{abspath}/data/{folder}/wind/Mdot_{check}.csv'
        data_row = np.concatenate([[snap, tfb[i], mfall], Mdot_pos_casted, v_rad_pos_casted, Mdot_neg_casted, v_rad_neg_casted])
        
        if alice:
            # os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    header = ['snap', ' tfb', ' Mdot_fb']
                    header += [f' Mdot_wind_pos at {r} ' for r in radii_names]
                    header += [f' v_rad_pos at {r} ' for r in radii_names]
                    header += [f' Mdot_wind_neg at {r} ' for r in radii_names]
                    header += [f' v_rad_neg at {r} ' for r in radii_names]
                    writer.writerow(header)
                writer.writerow(data_row) 
        file.close() 


if plot:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    Medd_code = Medd * prel.tsol_cgs / prel.Msol_cgs  # [g/s]
    snaps, tfb, mfall, \
    mwind_pos_Rt, mwind_pos_half_amb, mwind_pos_amb, mwind_pos_50Rt, \
    Vwind_pos_Rt, Vwind_pos_half_amb, Vwind_pos_amb, Vwind_pos_50Rt, \
    mwind_neg_Rt, mwind_neg_half_amb, mwind_neg_amb, mwind_neg_50Rt, \
    Vwind_neg_Rt, Vwind_neg_half_amb, Vwind_neg_amb, Vwind_neg_50Rt = \
        np.loadtxt(f'{abspath}/data/{folder}{check}/wind/Mdot_{check}.csv', 
                   delimiter = ',', 
                   skiprows=1, 
                   unpack=True)
    tfb_th = np.arange(0, 2, .1)
    Mpeak = Mpeak_intfb(params)
    Mdot_th = Mdot_theory(params, tfb_th)
    f_out_th = f_out_LodatoRossi(mfall, Medd_code)
    t_lag = amin*prel.Rsol_cgs / 1e9 #consider 10^4 km/s = 10^9 cm/s 
    t_lag_fb = t_lag / t_fb_days_cgs

    # compare with other dM/dE
    # snaps02, tfb02, mfall02, \
    # mwind_posRt02, mwind_pos_half_amb02, mwind_pos_amb02, mwind_pos_50Rt02, \
    # Vwind_posRt02, Vwind_pos_half_amb02, Vwind_pos_amb02, Vwind_pos_50Rt02, \
    # mwind_negRt02, mwind_neg_half_amb02, mwind_neg_amb02, mwind_neg_50Rt02, \
    # Vwind_negRt02, Vwind_neg_half_amb02, Vwind_neg_amb02, Vwind_neg_50Rt02 = \
    #     np.loadtxt(f'{abspath}/data/{folder}{check}/wind/Mdot_{check}02.csv', 
    #                delimiter = ',', 
    #                skiprows=1, 
    #                unpack=True)
    # fig, ax1 = plt.subplots(1, 1, figsize = (8,7))
    # ax1.plot(tfb[7:], np.abs(mwind_pos_amb[7:])/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm w}$ at $a_{\rm min}$')
    # ax1.plot(tfb, np.abs(mwind_neg_amb)/Medd_code,  c = 'forestgreen', label = r'$\dot{M}_{\rm in}$ at $a_{\rm min}$')
    # ax1.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm fb}$', c = 'k')
    # ax1.plot(tfb02[7:], np.abs(mwind_pos_amb02[7:])/Medd_code, ls = '--', c = 'b', label = r'$\dot{M}_{\rm w}$ at $a_{\rm min}$')
    # ax1.plot(tfb02, np.abs(mwind_neg_amb02)/Medd_code, ls = '--',  c = 'yellow', label = r'$\dot{M}_{\rm in}$ at $a_{\rm min}$')
    # ax1.plot(tfb02, np.abs(mfall02)/Medd_code, ls = '--', label = r'$\dot{M}_{\rm fb}$', c = 'gray')
    # ax1.set_yscale('log')
    # ax1.set_ylim(1e-1, 8e5)
    # ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')    
    # ax1.legend(fontsize = 18)
    # ax1.set_xlim(0, 1.7)
    # ax1.set_xlabel(r't/t$_{fb}$')
    # ax1.tick_params(axis='both', which='major', width=1, length=7)
    # ax1.tick_params(axis='both', which='minor', width=.8, length=4)
    # ax1.grid()
    # ax1.set_title(r'Dashed lines use dM/dE at t=0.2 t$_{\rm fb}$', fontsize = 20)
    # plt.tight_layout()
    # fig.savefig(f'{abspath}/Figs/Test/wind/Mdot_{check}dMdE02.png', bbox_inches = 'tight')

    fig, ax1 = plt.subplots(1, 1, figsize = (8,7))
    fig2, ax2 = plt.subplots(1, 1, figsize = (8,7))
    ax1.plot(tfb[7:], np.abs(mwind_pos_amb[7:])/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm w}$ at $a_{\rm min}$')
    ax1.plot(tfb, np.abs(mwind_neg_Rt)/Medd_code,  c = 'forestgreen', label = r'$\dot{M}_{\rm in}$ at $R_{\rm t}$')
    # ax1.plot(tfb_th, Mdot_th/Medd, c = 'gray', ls = '--', label = r'$\dot{M}_{\rm fb, theory} = \dot{M}_{\rm peak}(t/t_{\rm fb})^{-5/3}$')
    # ax1.axhline(Mpeak/Medd, c = 'deepskyblue', ls = ':', label = r'$\dot{M}_{\rm peak}=M_\star/3t_{\rm fb}$')
    ax1.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm fb}$ num', c = 'k')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 2e5)
    ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')    
    # ax2.plot(tfb, Vwind_pos_amb/prel.csol_cgs, c = 'dodgerblue', label = r'$v_{\rm w}$')
    # ax2.plot(tfb, Vwind_neg_amb/prel.csol_cgs, c = 'forestgreen', label = r'$v_{\rm in}$')
    ax2.plot(tfb, Vwind_pos_amb/v_esc, c = 'dodgerblue', label = r'$v_{\rm w}$ at $a_{\rm min}$')
    ax2.plot(tfb, Vwind_neg_amb/v_esc, c = 'forestgreen', label = r'$v_{\rm in}$ at $a_{\rm min}$')
    ax2.set_ylabel(r'$<v_{\rm w}> [v_{\rm esc(R_p)}]$')
    original_ticks = ax1.get_xticks()
    midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$t [t_{\rm fb}]$')
        ax.legend(fontsize = 18)
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', which='major', width=1, length=7)
        ax.tick_params(axis='both', which='minor', width=.8, length=4)
        ax.set_xlim(0, 1.7)
        ax.grid()
    ax2.set_title(r'$v_{\rm esc}\approx$'+f'{np.round(v_esc/prel.csol_cgs, 2)}c', fontsize = 20)
    plt.tight_layout()
    fig.savefig(f'{abspath}/Figs/paper/Mdot_{check}.pdf', bbox_inches = 'tight')

    # reproduce LodatoRossi11 Fig.6
    f_out_th = f_out_LodatoRossi(mfall, Medd_code)
    plt.figure(figsize = (8,6))
    plt.plot(np.abs(mfall/Medd_code), np.abs(f_out_th), c = 'k', label = 'LodatoRossi11')
    plt.plot(np.abs(mfall/Medd_code), np.abs(mwind_pos_amb/mfall), c = 'orange', label = r'numerical at $a_{\rm min}$')
    # plt.xlim(0, 100)
    plt.ylim(1e-7, 2)
    plt.legend(fontsize = 14)
    plt.xlabel(r'$\dot{M}_{\rm f} [\dot{M}_{\rm Edd}]$')
    plt.ylabel(r'$f_{\rm out}$')
    plt.loglog()
    plt.grid()
    ## Check convergence
    # dataposLow = np.loadtxt(f'{abspath}/data/{folder}LowResNewAMR/wind/Mdot_LowResNewAMR_Bpos.txt')
    # tfbL,  mwind_pos_ambL = dataposLow[0], dataposLow[4]
    # datanegLow = np.loadtxt(f'{abspath}/data/{folder}LowResNewAMR/wind/Mdot_LowResNewAMR_Bneg.txt')
    # mwind_neg_ambL =  datanegLow[3]
    # dataposHi = np.loadtxt(f'{abspath}/data/{folder}HiResNewAMR/wind/Mdot_HiResNewAMR_Bpos.txt')
    # tfbH,  mwind_pos_ambH = dataposHi[0], dataposHi[4]
    # datanegHi = np.loadtxt(f'{abspath}/data/{folder}HiResNewAMR/wind/Mdot_HiResNewAMR_Bneg.txt')
    # mwind_neg_ambH = datanegHi[3]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))
    # ax1.plot(tfbL, np.abs(mwind_pos_ambL)/Medd_code, c = 'C1', label = r'Low')
    # ax1.plot(tfb, np.abs(mwind_pos_amb)/Medd_code, c = 'yellowgreen', label = r'Fid')
    # ax1.plot(tfbH, np.abs(mwind_pos_ambH)/Medd_code, c = 'darkviolet', label = r'High') 
    # ax1.set_ylabel(r'$|\dot{M}_{\rm out}| [\dot{M}_{\rm Edd}]$')  
    # ax2.plot(tfbL, np.abs(mwind_neg_ambL)/Medd_code, c = 'C1')
    # ax2.plot(tfb, np.abs(mwind_neg_amb)/Medd_code, c = 'yellowgreen')
    # ax2.plot(tfbH, np.abs(mwind_neg_ambH)/Medd_code, c = 'darkviolet')
    # ax2.set_ylabel(r'$|\dot{M}_{\rm in}| [\dot{M}_{\rm Edd}]$')  
    # for ax in (ax1, ax2):
    #     ax.set_xlabel(r'$t [t_{\rm fb}]$')
    #     ax.legend(fontsize = 18)
    #     ax.set_xlim(0, 1.8)
    #     # ax.set_xticks(new_ticks)
    #     # labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]       
    #     # ax.set_xticklabels(labels)
    #     ax.tick_params(axis='both', which='major', width=0.7, length=7)
    #     ax.tick_params(axis='both', which='minor', width=0.5, length=5)
    #     ax.set_yscale('log')
    #     ax.grid()
    # ax1.set_ylim(1e-1, 1e1)
    # ax2.set_ylim(1e-1, 6e6)
    # ax1.legend(fontsize = 18)
    # plt.tight_layout()
    # fig.savefig(f'{abspath}/Figs/outflow/Mdot_convergence.pdf', bbox_inches = 'tight')
    


    

# %%
