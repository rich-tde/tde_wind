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
radii = np.array([0.2*amin, 0.5*amin, 0.7 * amin, amin])
Ledd = 1.26e38 * Mbh # [erg/s] Mbh is in solar masses
Medd = Ledd/(0.1*prel.c_cgs**2)
v_esc = np.sqrt(2*prel.G*Mbh/Rt)
convers_kms = prel.Rsol_cgs * 1e-5/prel.tsol_cgs # it's aorund 400

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
    mwind_pos = []
    Vwind_pos = []
    mwind_neg = []
    Vwind_neg = []
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
        X, Y, Z, Vol, Den, Mass, Press, VX, VY, VZ, IE_den = \
            data.X, data.Y, data.Z, data.Vol, data.Den, data.Mass, data.Vol, data.VX, data.VY, data.VZ, data.IE
        dim_cell = Vol**(1/3)
        cut = Den > 1e-19
        X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den= \
            make_slices([X, Y, Z, dim_cell, Den, Mass, Press, VX, VY, VZ, IE_den], cut)
        IE_spec = IE_den/Den
        Rsph = np.sqrt(X**2 + Y**2 + Z**2)
        V = np.sqrt(VX**2 + VY**2 + VZ**2)
        orb_en = orb.orbital_energy(Rsph, V, Mass, prel.G, prel.csol_cgs, Mbh) 
        bern = orb_en/Mass + IE_spec + Press/Den
        long = np.arctan2(Y, X)          # Azimuthal angle in radians
        lat = np.arccos(Z / Rsph)
        v_rad, _, _ = to_spherical_components(VX, VY, VZ, lat, long)
        # Postive velocity
        v_rad_pos_cond = bern >= 0  
        Den_pos, Rsph_pos, v_rad_pos, dim_cell_pos = \
            make_slices([Den, Rsph, v_rad, dim_cell], v_rad_pos_cond)
        if Den_pos.size == 0:
            print(bern, flush=True)
            sys.stdout.flush()
            mwind_pos.append(0)
            Vwind_pos.append(0)
            continue
        Mdot_pos = dim_cell_pos**2 * Den_pos * v_rad_pos # there should be a pi factor here, but you put it later
        casted = multiple_branch(radii, Rsph_pos, [Mdot_pos, v_rad_pos], weights_matrix = ['sum', 'mean'])
        # Mdot_pos = Den_pos * v_rad_pos # there should be a 4piR^2 factor here, but you put it later
        # casted = multiple_branch(radii, Rsph_pos, [Mdot_pos, v_rad_pos], weights_matrix = [dim_cell_pos, 1])
        Mdot_pos_casted, v_rad_pos_casted = casted[0], casted[1]
        mwind_pos.append(Mdot_pos_casted * np.pi)#) 4 *  * radii**2)
        Vwind_pos.append(v_rad_pos_casted)
        # Negative velocity 
        v_rad_neg_cond = bern < 0
        Den_neg, Rsph_neg, v_rad_neg, dim_cell_neg = \
            make_slices([Den, Rsph, v_rad, dim_cell], v_rad_neg_cond)
        if Den_neg.size == 0:
            print(bern, flush=True)
            sys.stdout.flush()
            mwind_pos.append(0)
            Vwind_pos.append(0)
            continue
        # Mdot_neg = Den_neg * v_rad_neg # there should be a 4piR^2 factor here, but you put it later
        # casted = multiple_branch(radii, Rsph_neg, [Mdot_neg, v_rad_neg], weights_matrix = [dim_cell_neg, 1])
        Mdot_neg = dim_cell_neg**2 * Den_neg * v_rad_neg        
        casted, indices_in = multiple_branch(radii, Rsph_neg, [Mdot_neg, v_rad_neg], weights_matrix = ['sum', 'mean'], keep_track=True)
        # print('in:', np.pi*np.sum((dim_cell_neg[indices_in[0]])**2))
        Mdot_neg_casted, v_rad_neg_casted = casted[0], casted[1]
        mwind_neg.append(Mdot_neg_casted * np.pi) #4 * radii**2
        Vwind_neg.append(v_rad_neg_casted)
    # print('theory:', 4*np.pi*radii**2)

    mwind_pos = np.transpose(np.array(mwind_pos)) # shape pass from len(snap) x len(radii) to len(radii) x len(snap)
    mwind_neg = np.transpose(np.array(mwind_neg))
    Vwind_pos = np.transpose(np.array(Vwind_pos))
    Vwind_neg = np.transpose(np.array(Vwind_neg))

    if alice:
        with open(f'{abspath}/data/{folder}/Mdot_{check}_Bpos.txt','w') as file:
            file.write(f'# Distinguish using Bernouilli criterion \n#t/tfb \n')
            file.write(f' '.join(map(str, tfb)) + '\n')
            file.write(f'# Mdot_f \n')
            file.write(f' '.join(map(str, mfall)) + '\n')
            file.write(f'# Mdot_wind at 0.2 amin\n')
            file.write(f' '.join(map(str, mwind_pos[0])) + '\n')
            file.write(f'# Mdot_wind at 0.5 amin\n')
            file.write(f' '.join(map(str, mwind_pos[1])) + '\n')
            file.write(f'# Mdot_wind at 0.7 amin\n')
            file.write(f' '.join(map(str, mwind_pos[2])) + '\n')
            file.write(f'# Mdot_wind at amin\n')
            file.write(f' '.join(map(str, mwind_pos[3])) + '\n')
            file.write(f'# v_wind at 0.2 amin\n')
            file.write(f' '.join(map(str, Vwind_pos[0])) + '\n')
            file.write(f'# v_wind at 0.5 amin\n')
            file.write(f' '.join(map(str, Vwind_pos[1])) + '\n')
            file.write(f'# v_wind at 0.7 amin\n')
            file.write(f' '.join(map(str, Vwind_pos[2])) + '\n')
            file.write(f'# v_wind at amin\n')
            file.write(f' '.join(map(str, Vwind_pos[3])) + '\n')
            file.close()
        
        with open(f'{abspath}/data/{folder}/Mdot_{check}_Bneg.txt','w') as file:
            file.write(f'# Distinguish using Bernouilli criterion \n#t/tfb \n')
            file.write(f' '.join(map(str, tfb)) + '\n')
            file.write(f'# Mdot_wind at 0.2 amin\n')
            file.write(f' '.join(map(str, mwind_neg[0])) + '\n')
            file.write(f'# Mdot_wind at 0.5 amin\n')
            file.write(f' '.join(map(str, mwind_neg[1])) + '\n')
            file.write(f'# Mdot_wind at 0.7 amin\n')
            file.write(f' '.join(map(str, mwind_neg[2])) + '\n')
            file.write(f'# Mdot_wind at amin\n')
            file.write(f' '.join(map(str, mwind_neg[3])) + '\n')
            file.write(f'# v_wind at 0.2 amin\n')
            file.write(f' '.join(map(str, Vwind_neg[0])) + '\n')
            file.write(f'# v_wind at 0.5 amin\n')
            file.write(f' '.join(map(str, Vwind_neg[1])) + '\n')
            file.write(f'# v_wind at 0.7 amin\n')
            file.write(f' '.join(map(str, Vwind_neg[2])) + '\n')
            file.write(f'# v_wind at amin\n')
            file.write(f' '.join(map(str, Vwind_neg[3])) + '\n')
            file.close()

if plot:
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))
    ax1.plot(tfb, np.abs(mfall)/Medd_code, label = r'$\dot{M}_{\rm f}$', c = 'k')
    ax1.plot(tfb, np.abs(mwind_pos1)/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm out}$ 0.5$a_{\rm min}$')
    ax1.plot(tfb, np.abs(mwind_neg1)/Medd_code, c = 'dodgerblue', label = r'$\dot{M}_{\rm in}$ 0.5$a_{\rm min}$', ls = '--')
    ax1.axvline(tfb[np.argmax(np.abs(mfall)/Medd_code)], c = 'k', linestyle = 'dotted')
    ax1.text(tfb[np.argmax(np.abs(mfall)/Medd_code)]+0.01, 0.1, r'$t_{\dot{M}_{\rm peak}}$', fontsize = 20, rotation = 90)
    ax1.set_yscale('log')
    # ax1.ylim(1e-7, 3)
    ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')    
    ax2.plot(tfb, Vwind_pos1/v_esc, c = 'dodgerblue', label = r'$v_{\rm out}(0.5 a_{\rm min})$')
    ax2.plot(tfb, Vwind_neg1/v_esc, '--', c = 'dodgerblue', label = r'$v_{\rm in}(0.5 a_{\rm min})$')
    ax2.set_ylabel(r'$v_{\rm out}/v_{\rm esc}(R_{\rm t})$')
    for ax in (ax1, ax2):
        ax.legend(fontsize = 14)
        ax.set_xlabel(r'$t/t_{\rm fb}$')
    # plt.suptitle(r'$\dot{M}_{\rm out, in} = \pi\sum_i v_{\rm{rad},i}\rho_i V_i^{2/3}$ distinguishing for $v_{\rm{rad}}><0$', fontsize = 20)
    plt.suptitle(r'$\dot{M}_{\rm out, in} = 4\pi R^2\sum_i (v_{\rm{rad},i}\rho_i V_i^3) /\sum_i V_i^3$ distinguishing for $v_{\rm{rad}}><0$, where $R$ is the distance from the BH', fontsize = 20)
    plt.tight_layout()

    # reproduce LodatoRossi11 Fig.6
    plt.figure(figsize = (8,6))
    plt.plot(np.abs(mfall/Medd_code), np.abs(f_out_th), c = 'k')
    plt.xlim(0, 100)
    plt.legend(fontsize = 14)
    plt.xlabel(r'$\dot{M}_{\rm f} [\dot{M}_{\rm Edd}]$')
    plt.ylabel(r'$f_{\rm out}$')

    plt.figure(figsize = (8,6))
    plt.plot(tfb, np.abs(mwind_pos/mfall), c = 'dodgerblue', label = r'f$_{\rm out}$ (0.2$a_{\rm min})$') 
    plt.plot(tfb, np.abs(mwind_pos1/mfall), c = 'orange', label = r'f$_{\rm out}$ (0.5$a_{\rm min})$')
    plt.plot(tfb, np.abs(mwind_pos2/mfall), c = 'purple', label = r'f$_{\rm out}$ (0.7$a_{\rm min})$')
    plt.plot(tfb, np.abs(mwind_pos3/mfall), c = 'green', label = r'f$_{\rm out}$ (1$a_{\rm min})$')
    plt.plot(tfb, np.abs(f_out_th), c = 'k', label = 'LodatoRossi11')
    plt.legend(fontsize = 14)
    plt.xlabel(r't $[t_{\rm fb}]$')
    plt.ylabel(r'$f_{\rm out}\equiv \dot{M}_{\rm wind}/\dot{M}_{\rm fb}$')
    plt.yscale('log')
    # plt.savefig(f'{abspath}/Figs/outflow/Mdot.png')


    
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))
ax1.plot(tfb, np.abs(mfall)*prel.Msol_cgs/prel.tsol_cgs, label = r'$\dot{M}_{\rm f}$', c = 'k')
ax1.plot(tfb, np.abs(mwind_pos)*prel.Msol_cgs/prel.tsol_cgs, c = 'dodgerblue', label = r'$\dot{M}_{\rm out}$ 0.2$a_{\rm min}$')
ax1.plot(tfb, np.abs(mwind_pos2)*prel.Msol_cgs/prel.tsol_cgs, c = 'purple', label = r'$\dot{M}_{\rm out}$ 0.7$a_{\rm min}$')
ax1.plot(tfb, np.abs(mwind_neg)*prel.Msol_cgs/prel.tsol_cgs, c = 'dodgerblue', label = r'$\dot{M}_{\rm in}$ 0.2$a_{\rm min}$', ls = '--')
ax1.plot(tfb, np.abs(mwind_neg2)*prel.Msol_cgs/prel.tsol_cgs, c = 'purple', label = r'$\dot{M}_{\rm in}$ 0.7$a_{\rm min}$', ls = '--')
ax1.axvline(tfb[np.argmax(np.abs(mfall)*prel.Msol_cgs/prel.tsol_cgs)], c = 'k', linestyle = 'dotted')
ax1.text(tfb[np.argmax(np.abs(mfall)*prel.Msol_cgs/prel.tsol_cgs)]+0.01, 0.1, r'$t_{\dot{M}_{\rm peak}}$', fontsize = 20, rotation = 90)
ax1.set_yscale('log')
# ax1.ylim(1e-7, 3)
ax1.set_ylabel(r'$|\dot{M}| [\dot{M}_{\rm Edd}]$')   
# %%
