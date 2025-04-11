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
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel
import src.orbits as orb
from Utilities.operators import make_tree, multiple_branch, to_spherical_components
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
check = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

tfallback = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
tfallback_cgs = tfallback * 24 * 3600 #converted to seconds
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt/beta
norm_dMdE = Mbh/Rt * (Mbh/Rstar)**(-1/3) # Normalisation (what on the x axis you call \Delta E). It's GM/Rt^2 * Rstar
apo = orb.apocentre(Rstar, mstar, Mbh, beta) 
amin = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
radii = [0.2*amin, 0.5*amin, 0.7 * amin, amin] 
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Medd = Ledd/(0.1*prel.c_cgs**2)
v_esc = np.sqrt(2*prel.G*Mbh/Rt)

#
## FUNCTIONS
#
def f_out_LodatoRossi(M_fb, M_edd):
    f = 2/np.pi * np.arctan(1/7.5 * (M_fb/M_edd-1))
    return f
#%% MAIN
if compute: # compute dM/dt = dM/dE * dE/dt
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    tfb_cgs = tfb * tfallback_cgs #converted to seconds
    bins = np.loadtxt(f'{abspath}/data/{folder}/dMdE_{check}_bins.txt')
    max_bin_negative = np.abs(np.min(bins))
    mid_points = (bins[:-1]+bins[1:])* norm_dMdE/2  # get rid of the normalization
    dMdE_distr = np.loadtxt(f'{abspath}/data/{folder}/dMdE_{check}.txt')[0] # distribution just after the disruption
    bins_tokeep, dMdE_distr_tokeep = mid_points[mid_points<0], dMdE_distr[mid_points<0] # keep only the bound energies
    mfall = np.zeros(len(tfb_cgs))
    mwind = np.zeros(len(tfb_cgs))
    mwind1 = np.zeros(len(tfb_cgs))
    mwind2 = np.zeros(len(tfb_cgs))
    mwind3 = np.zeros(len(tfb_cgs))
    Vwind = np.zeros(len(tfb_cgs))
    Vwind1 = np.zeros(len(tfb_cgs))
    Vwind2 = np.zeros(len(tfb_cgs))
    Vwind3 = np.zeros(len(tfb_cgs))
    unbound_ratio = np.zeros(len(tfb_cgs))
    unbound_ratio1 = np.zeros(len(tfb_cgs))
    unbound_ratio2 = np.zeros(len(tfb_cgs))
    unbound_ratio3 = np.zeros(len(tfb_cgs))
    for i, snap in enumerate(snaps):
        print(snap, flush=True)
        sys.stdout.flush()
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
        if energy/bins_tokeep[i_bin] > max_bin_negative:
            print('You overcome the maximum negative bin')
        dMdE_t = dMdE_distr_tokeep[i_bin]
        mdot = orb.Mdot_fb(Mbh, prel.G, tsol, dMdE_t)
        mfall[i] = mdot # code units

        data = make_tree(path, snap, energy = True)
        X, Y, Z, Mass, Den, Press, IE, VX, VY, VZ = \
            data.X, data.Y, data.Z, data.Mass, data.Den, data.Press, data.IE, data.VX, data.VY, data.VZ
        cut = Den > 1e-19
        X, Y, Z, Mass, Den, Press, IE, VX, VY, VZ = \
            make_slices([X, Y, Z, Mass, Den, Press, IE, VX, VY, VZ], cut)
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        cells_len = np.arange(len(X))
        # xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
        #     np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
        # xph, yph, zph, volph, denph, Tempph, Rad_denph, Vxph, Vyph, Vzph = \
        #     np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.txt')
        # rph = np.sqrt(xph**2 + yph**2 + zph**2)
        # idx_r_wind = np.argmin(np.abs(rph - radii))
        # x_w, y_w, z_w, r_w, vx_w, vy_w, vz_w = \
        #     xph[idx_r_wind], yph[idx_r_wind], zph[idx_r_wind], rph[idx_r_wind], Vxph[idx_r_wind], Vyph[idx_r_wind], Vzph[idx_r_wind]
        # long_w = np.arctan2(y_w, x_w)          # Azimuthal angle in radians
        # lat_w = np.arccos(z_w / r_w)
        # v_rad_w, _, _ = to_spherical_components(vx_w, vy_w, vz_w, lat_w, long_w)
        long = np.arctan2(Y, X)          # Azimuthal angle in radians
        lat = np.arccos(Z / Rsph)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, lat, long)
        Den_casted, v_rad_casted, indices = \
            multiple_branch(radii, Rsph, [Den, v_rad], weights = [Mass, Mass], keep_track = True)
        
        for j, Mw, Vw, UnW in zip(np.arange(4), [mwind, mwind1, mwind2, mwind3], [Vwind, Vwind1, Vwind2, Vwind3], [unbound_ratio, unbound_ratio1, unbound_ratio2, unbound_ratio3]):
            Mw[i] = 4 * np.pi * radii[j]**2 * Den_casted[j] * v_rad_casted[j] # v_wind = 4pi*r^2 * rho(r) * v(r) with r far enough so that velocity ~const, but not too far or it overcome tfb
            Vw[i] = v_rad_casted[j] 
            cell_indices = cells_len[indices[j]]
            Rsph_cell, Mass_cell, V_cell, Press_cell, Den_cell, IE_cell = \
                make_slices([Rsph, Mass, V, Press, Den, IE], cell_indices)
            OE_cell = orb.orbital_energy(Rsph_cell, V_cell, Mass_cell, prel.G, prel.csol_cgs, Mbh)
            B = OE_cell / Mass_cell + IE_cell + Press_cell/Den_cell
            UnW[i] = len(B[B>0])/len(B)

    with open(f'{abspath}/data/{folder}/Mdot_{check}.txt','a') as file:
        file.write(f'# t/tfb \n')
        file.write(f' '.join(map(str, tfb)) + '\n')
        file.write(f'# Mdot_f \n')
        file.write(f' '.join(map(str, mfall)) + '\n')
        file.write(f'# Mdot_wind at 0.2 amin\n')
        file.write(f' '.join(map(str, mwind)) + '\n')
        file.write(f'# Mdot_wind at 0.5 amin\n')
        file.write(f' '.join(map(str, mwind1)) + '\n')
        file.write(f'# Mdot_wind at 0.7 amin\n')
        file.write(f' '.join(map(str, mwind2)) + '\n')
        file.write(f'# Mdot_wind at amin\n')
        file.write(f' '.join(map(str, mwind3)) + '\n')
        file.write(f'# v_wind at 0.2 amin\n')
        file.write(f' '.join(map(str, Vwind)) + '\n')
        file.write(f'# v_wind at 0.5 amin\n')
        file.write(f' '.join(map(str, Vwind1)) + '\n')
        file.write(f'# v_wind at 0.7 amin\n')
        file.write(f' '.join(map(str, Vwind2)) + '\n')
        file.write(f'# v_wind at amin\n')
        file.write(f' '.join(map(str, Vwind3)) + '\n')
        file.write(f'# unbound ratio at 0.2 amin\n')
        file.write(f' '.join(map(str, unbound_ratio)) + '\n')
        file.write(f'# unbound ratio at 0.5 amin\n')
        file.write(f' '.join(map(str, unbound_ratio1)) + '\n')
        file.write(f'# unbound ratio at 0.7 amin\n')
        file.write(f' '.join(map(str, unbound_ratio2)) + '\n')
        file.write(f'# unbound ratio at amin\n')
        file.write(f' '.join(map(str, unbound_ratio3)) + '\n')
        file.close()

if plot:
    tfb, mfall, mwind, mwind1, mwind2, mwind3, Vwind, Vwind1, Vwind2, Vwind3 = np.loadtxt(f'{abspath}/data/{folder}/Mdot_{check}.txt')
    Medd_code = Medd * prel.tsol_cgs / prel.Msol_cgs  # [g/s]
    f_out_th = f_out_LodatoRossi(mfall, Medd_code)

    plt.figure(figsize = (8,6))
    plt.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm f}$', c = 'k')
    plt.plot(tfb, np.abs(mwind)/Medd_code,  label = r'$\dot{M}_{\rm w}$ 0.2$a_{\rm min}$', c = 'dodgerblue')
    plt.plot(tfb, np.abs(mwind1)/Medd_code,  label = r'$\dot{M}_{\rm w}$ 0.5$a_{\rm min}$', c = 'orange')
    plt.axvline(tfb[np.argmax(np.abs(mfall)/Medd_code)], c = 'k', linestyle = 'dotted')
    plt.text(tfb[np.argmax(np.abs(mfall)/Medd_code)]+0.01, 0.1, r'$t_{\dot{M}_{\rm peak}}$', fontsize = 20, rotation = 90)
    plt.yscale('log')
    # plt.ylim(1e-7, 3)
    plt.legend(fontsize = 14)
    plt.xlabel(r'$t/t_{\rm fb}$')
    plt.ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')
    plt.savefig(f'{abspath}/Figs/outflow/Mdot.png')
    
    # reproduce LodatoRossi11 Fig.6
    plt.figure(figsize = (8,6))
    plt.plot(np.abs(mfall/Medd_code), np.abs(f_out_th), c = 'k')
    plt.xlim(0, 100)
    plt.legend(fontsize = 14)
    plt.xlabel(r'$\dot{M}_{\rm f} [\dot{M}_{\rm Edd}]$')
    plt.ylabel(r'$f_{\rm out}$')

    plt.figure(figsize = (8,6))
    plt.plot(tfb, np.abs(mwind/mfall), c = 'dodgerblue', label = r'f$_{\rm out}$ (0.2$a_{\rm min})$') 
    plt.plot(tfb, np.abs(mwind1/mfall), '--', c = 'orange', label = r'f$_{\rm out}$ (0.5$a_{\rm min})$')
    plt.plot(tfb, np.abs(mwind2/mfall), '--', c = 'purple', label = r'f$_{\rm out}$ (0.7$a_{\rm min})$')
    plt.plot(tfb, np.abs(mwind3/mfall), '--', c = 'green', label = r'f$_{\rm out}$ (1$a_{\rm min})$')
    plt.plot(tfb, np.abs(f_out_th), c = 'k', label = 'LodatoRossi11')
    plt.legend(fontsize = 14)
    plt.xlabel(r't $[t_{\rm fb}]$')
    plt.ylabel(r'$f_{\rm out}\equiv \dot{M}_{\rm wind}/\dot{M}_{\rm fb}$')
    plt.yscale('log')
    # plt.savefig(f'{abspath}/Figs/outflow/Mdot.png')

    plt.figure(figsize = (8,6))
    plt.plot(tfb, Vwind/v_esc, c = 'dodgerblue', label = r'$v_{\rm wind}(0.2 a_{\rm min})$')
    # plt.plot(tfb, Vwind1/v_esc), '--', c = 'orange', label = r'$v_{\rm wind}(0.5 a_{\rm min})$')
    # plt.plot(tfb, Vwind2/v_esc), '--', c = 'purple', label = r'$v_{\rm wind}(0.7 a_{\rm min})$')
    plt.plot(tfb, Vwind3/v_esc, '--', c = 'green', label = r'$v_{\rm wind}(a_{\rm min})$')
    plt.xlabel(r't $[t_{\rm fb}]$')
    plt.ylabel(r'$v_{\rm wind}/v_{\rm esc}(R_{\rm t})$')
    plt.yscale('symlog')
    plt.legend(fontsize = 14)
    # plt.savefig(f'{abspath}/Figs/outflow/Mdot.png')

# %%
