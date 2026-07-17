import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities import operators as op
import src.orbits as orb

# Physical constants in cgs
m_e = 9.10938356e-28    # g
conversion_vel = prel.Rsol_cgs/prel.tsol_cgs

def hydrogenic_levels(n, chi_eV):
    chi = chi_eV * prel.ev_to_erg
    g = 2.0 * n**2
    E = chi * (1.0 - 1.0 / n**2)
    return g, E

def hydrogenic_partition_function(T, chi_eV, n_max=30):
    # Z = sum_{n=1}^{n_max} g_n * exp(-E_n / (k_B * T)) 
    # Bradt: https://www.cambridge.org/us/files/5413/6681/8627/7706_Saha_equation.pdf
    chi = chi_eV * prel.ev_to_erg
    Z = 0.0
    for n in range(1, n_max + 1):
        g, E = hydrogenic_levels(n, chi_eV)
        Z += g * np.exp(-E / (prel.Kb_cgs * T)) # should we have some weights? But n_max is already a cutoff, so maybe not
    return Z

def saha_omega(T, n_e, chi_eV, Z_i, Z_ip1):
    # Eq 35 from Bradt
    chi = chi_eV * prel.ev_to_erg
    e_debroglie = prel.h_cgs / np.sqrt(2.0 * np.pi * m_e * prel.Kb_cgs * T)
    return 2.0 * Z_ip1 / Z_i * np.exp(-chi / (prel.Kb_cgs * T)) / (n_e * e_debroglie**3)

def solve_saha_h_he(T, rho, X_H=0.71, X_He=0.28, tol=1e-8, maxiter=200):
    n_H = X_H * rho / prel.m_p_cgs
    n_He = X_He * rho / (4.0 * prel.m_p_cgs)

    Z_HI = hydrogenic_partition_function(T, 13.6, 30) # neutral hydrogen
    Z_HII = 1.0 #  bare proton, so there are no bound electronic levels to sum 
    Z_HeI = 1.0 # crude placeholder. Neutral helium is not hydrogenic, so this is not physically accurate
    Z_HeII = hydrogenic_partition_function(T, 54.418, 30) # He II is hydrogen-like: one electron around a nucleus
    Z_HeIII = 1.0 # bare helium nucleus, so again no bound electronic levels

    # first guess: If all hydrogen is ionized, each H contributes one electron and each He contributes up to two electrons 
    # since you have 1 electron from HI -> HII, and 1 electron from He HeI -> HeII and 1 electron from HeII -> HeIII
    # it's usually stable andnot far from the final solution, so it should converge quickly
    n_e = n_H + 2.0 * n_He 

    for _ in range(maxiter):
        S_H = saha_omega(T, n_e, 13.6, Z_HI, Z_HII)
        n_HII = n_H * S_H / (1.0 + S_H)
        n_HI = n_H - n_HII

        S_He1 = saha_omega(T, n_e, 24.587, Z_HeI, Z_HeII)
        S_He2 = saha_omega(T, n_e, 54.418, Z_HeII, Z_HeIII)

        denom = 1.0 + S_He1 + S_He1 * S_He2
        n_HeI = n_He / denom
        n_HeII = n_HeI * S_He1
        n_HeIII = n_HeII * S_He2

        # Update the electron density will have 1 electron for every HII, 1 electron for every HeII, and 2 electrons for every HeIII
        n_e_new = n_HII + n_HeII + 2.0 * n_HeIII
        if abs(n_e_new - n_e) / max(n_e, 1e-30) < tol:
            n_e = n_e_new
            break
        n_e = 0.5 * (n_e + n_e_new)

    return {
        "n_e": n_e,
        "n_HI": n_HI, "n_HII": n_HII,
        "n_HeI": n_HeI, "n_HeII": n_HeII, "n_HeIII": n_HeIII,
        "Z_HI": Z_HI, "Z_HII": Z_HII,
        "Z_HeI": Z_HeI, "Z_HeII": Z_HeII, "Z_HeIII": Z_HeIII
    }

def line_ratio_paper(T, rho, line="Ha", X_H=0.7, X_He=0.28, delta_r=1e13, v=1e9, n_p=None):
    # for A and lambda0 vslues: https://www.nist.gov/system/files/documents/srd/jpcrd382009565p.pdf
    # for H look at pag 573 (line 40, 41)
    # for HeII look at pag 576 (line 1)
    pops = solve_saha_h_he(T, rho, X_H=X_H, X_He=X_He)
    if n_p is None:
        n_p = pops["n_HII"]

    if line == "Ha": 
        chi_eV = 13.6
        n_starter = 3
        A = 4.4101e7 # transition 3->2
        lam0 = 6.56283e-5
        Z_i = pops["Z_HI"]
        omega = saha_omega(T, pops["n_e"], chi_eV, Z_i, pops["Z_HII"])
    elif line == "Hb":
        chi_eV = 13.6
        n_starter = 4
        A = 8.4193e6 # transition 4->2
        lam0 = 4.86134e-5
        Z_i = pops["Z_HI"]
        omega = saha_omega(T, pops["n_e"], chi_eV, Z_i, pops["Z_HII"])
    elif line == "HeII4686":
        chi_eV = 54.418
        n_starter = 4
        A = 8.215e6
        lam0 = 4.6857e-5
        Z_i = pops["Z_HeII"]
        omega = saha_omega(T, pops["n_e"], chi_eV, Z_i, pops["Z_HeIII"])
    else:
        raise ValueError("line must be 'Ha', 'Hb', or 'HeII4686'")

    g_u, E_u = hydrogenic_levels(n_starter, chi_eV)

    pref = (g_u / Z_i) * np.exp(-E_u / (prel.Kb_cgs * T))
    const = (A * prel.h_cgs * prel.c_cgs / (2.0 * np.pi * prel.Kb_cgs))
    ratio = pref * const / (1.0 + omega)  * (n_p * delta_r * lam0**2 / (T * v))

    return ratio, pops, {"omega": omega, "g_u": g_u, "E_u": E_u, "Z_i": Z_i}

####### MAIN

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
what_to_plot = 'time_evol' # 'single_snap_all_obs', 'single_snap_sec', 'time_evol' 'ratio_el'
choice = 'left_right_z'
delta_r = 1e14

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = op.sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_obs, label_obs, color_obs, _ = op.choose_observers(observers_xyz, choice)
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rt = things['Rt']

if what_to_plot == 'single_snap_all_obs' or what_to_plot == 'single_snap_sec':
    snap = 151
    markers = ['o', 's', 'X']

    time = tfb[snaps == snap][0]
    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    x_ph, y_ph, z_ph, vx_ph, vy_ph, vz_ph, d_ph, temp_ph = \
        photo['x'], photo['y'], photo['z'], photo['vx'], photo['vy'], photo['vz'], photo['den'], photo['temp']
    vr_ph, _, _ = op.to_spherical_components(vx_ph, vy_ph, vz_ph, x_ph, y_ph, z_ph) 
    vr_ph *= conversion_vel
    ratiosHa = np.zeros(len(temp_ph))
    ratiosHb = np.zeros(len(temp_ph))
    ratiosHe = np.zeros(len(temp_ph))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(temp_ph)):
        ratiosHa[i], _, _ = line_ratio_paper(
                                T=temp_ph[i],
                                rho=d_ph[i],
                                line="Ha",
                                delta_r=delta_r,
                                v=vr_ph[i])
        
        ratiosHb[i], _, _ = line_ratio_paper(
                                T=temp_ph[i],
                                rho=d_ph[i],
                                line="Hb",
                                delta_r=delta_r,
                                v=vr_ph[i])
        
        ratiosHe[i], _, _ = line_ratio_paper(
                                T=temp_ph[i],
                                rho=d_ph[i],
                                line="HeII4686",
                                delta_r=delta_r,
                                v=vr_ph[i])

    if what_to_plot == 'single_snap_all_obs':
        fig.suptitle(f't = {time:.1f}' + r'$t_{\rm fb}$, all observers', fontsize=20)
        # this first loop is just for the legend of the markers
        for i, index in enumerate(indices_obs):
            if label_obs[i] == 'South pole':
                continue
            ax.scatter(temp_ph[index], ratiosHa[index], s = 100, color='none', marker = markers[i], edgecolor='k', label = label_obs[i])
            ax.scatter(temp_ph[index], ratiosHb[index], s = 100, color='none', marker = markers[i], edgecolor='k')
            ax.scatter(temp_ph[index], ratiosHe[index], s = 100, color='none', marker = markers[i], edgecolor='k')
        
        for i, index in enumerate(indices_obs):
            if label_obs[i] == 'South pole':
                continue
            ax.scatter(temp_ph[index], ratiosHa[index], label="Ha" if i == 0 else "", s = 100, marker = markers[i], color='C1')
            ax.scatter(temp_ph[index], ratiosHb[index], label="Hb" if i == 0 else "", s = 100, marker = markers[i], color='darkviolet')
            ax.scatter(temp_ph[index], ratiosHe[index], label="HeII4686" if i == 0 else "", s = 100, marker = markers[i], color='dodgerblue')
        
        # ax.scatter(temp_ph[88:103], ratiosHa[88:103], s = 100, color='C1', edgecolor='k')
        # ax.scatter(temp_ph[88:103], ratiosHb[88:103], s = 100, color='darkviolet', edgecolor='k')
        # ax.scatter(temp_ph[88:103], ratiosHe[88:103], s = 100, color='none', edgecolor='k', label = 'orb.pl.')

    if what_to_plot == 'single_snap_sec':
        fig.suptitle(f't = {time:.1f}' + r'$t_{\rm fb}$, sections', fontsize=20)
        ratiosHa_sec = np.zeros(len(indices_obs))
        ratiosHb_sec = np.zeros(len(indices_obs))
        ratiosHe_sec = np.zeros(len(indices_obs))
        temp_ph_sec = np.zeros(len(indices_obs))
        for i, indices in enumerate(indices_obs): # you need that if the splitting is not homogeneous
            ratiosHa_sec[i] = np.median(ratiosHa[indices])
            ratiosHb_sec[i] = np.median(ratiosHb[indices])
            ratiosHe_sec[i] = np.median(ratiosHe[indices])
            temp_ph_sec[i] = np.median(temp_ph[indices])

        for i in range(len(indices_obs)):
            # this first loop is just for the legend of the markers
            if label_obs[i] == 'South pole':
                continue
            ax.scatter(temp_ph_sec[i], ratiosHa_sec[i], label = label_obs[i], marker = markers[i], s = 95, color = 'none', edgecolor = 'k')
        for i in range(len(indices_obs)):
            if label_obs[i] == 'South pole':
                continue
            ax.scatter(temp_ph_sec[i], ratiosHa_sec[i], label="Ha" if i == 0 else "", marker = markers[i], s = 100, color = 'C1')
            ax.scatter(temp_ph_sec[i], ratiosHb_sec[i], label="Hb" if i == 0 else "", marker = markers[i], s = 100, color = 'darkviolet')
            ax.scatter(temp_ph_sec[i], ratiosHe_sec[i], label="HeII4686" if i == 0 else "", marker = markers[i], s = 100, color = 'dodgerblue')

    ax.set_xlabel("Temperature")
    ax.set_ylabel(r"$L_{\lambda, \rm {line}}/L_{\lambda, \rm {cont}}$")
    ax.loglog()
    ax.set_xlim(5e3, 1e5)
    ax.set_ylim(5e-3, 2e3)
    ax.tick_params(axis='both', which='major', length=7, width=1)
    ax.tick_params(axis='both', which='minor', length=4, width=.9)
    ax.legend(fontsize=16)
    ax.grid()
    plt.tight_layout()

if what_to_plot == 'time_evol' or what_to_plot == 'ratio_el':
    dataBB = np.loadtxt(f'{abspath}/data/{folder}/wind/Tfit_intime_{choice}.txt', delimiter=',', skiprows=1, unpack=True)
    radiiBB = dataBB[len(indices_obs)+1:2*len(indices_obs)+1, :] # they're saved in cgs
    radiiBB /= prel.Rsol_cgs 
    ratiosHa_t = []
    ratiosHb_t = []
    ratiosHe_t = []
    temp_t = []
    fig, ax = plt.subplots(1, len(indices_obs), figsize=(8*len(indices_obs), 6))
    figTR, (axT, axR) = plt.subplots(2, len(indices_obs), figsize=(8*len(indices_obs), 12))
    for snap in snaps:
        time = tfb[snaps == snap][0]
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        x_ph, y_ph, z_ph, vx_ph, vy_ph, vz_ph, d_ph, temp_ph = \
            photo['x'], photo['y'], photo['z'], photo['vx'], photo['vy'], photo['vz'], photo['den'], photo['temp']
        vr_ph, _, _ = op.to_spherical_components(vx_ph, vy_ph, vz_ph, x_ph, y_ph, z_ph) 
        vr_ph *= conversion_vel
        ratiosHa = np.zeros(len(temp_ph))
        ratiosHb = np.zeros(len(temp_ph))
        ratiosHe = np.zeros(len(temp_ph))

        for i in range(len(temp_ph)):
            ratiosHa[i], _, _ = line_ratio_paper(
                                    T=temp_ph[i],
                                    rho=d_ph[i],
                                    line="Ha",
                                    delta_r=delta_r,
                                    v=vr_ph[i])
            
            ratiosHb[i], _, _ = line_ratio_paper(
                                    T=temp_ph[i],
                                    rho=d_ph[i],
                                    line="Hb",
                                    delta_r=delta_r,
                                    v=vr_ph[i])
            
            ratiosHe[i], _, _ = line_ratio_paper(
                                    T=temp_ph[i],
                                    rho=d_ph[i],
                                    line="HeII4686",
                                    delta_r=delta_r,
                                    v=vr_ph[i])
        ratiosHa_sec = np.zeros(len(indices_obs))
        ratiosHb_sec = np.zeros(len(indices_obs))
        ratiosHe_sec = np.zeros(len(indices_obs))
        temp_sec = np.zeros(len(indices_obs))
        for i, indices in enumerate(indices_obs): # you need that if the splitting is not homogeneous
            if snap == 151:
                print('max ', np.max(ratiosHa[indices]))
            ratiosHa_sec[i] = np.median(ratiosHa[indices])
            if snap == 151:
                print('mean ', np.mean(ratiosHa[indices]))
            ratiosHb_sec[i] = np.median(ratiosHb[indices])
            ratiosHe_sec[i] = np.median(ratiosHe[indices])
            temp_sec[i] = np.median(temp_ph[indices])
        ratiosHa_t.append(ratiosHa_sec)
        ratiosHb_t.append(ratiosHb_sec)
        ratiosHe_t.append(ratiosHe_sec)
        temp_t.append(temp_sec)
        
    ratiosHa_t = np.transpose(ratiosHa_t)
    ratiosHb_t = np.transpose(ratiosHb_t)
    ratiosHe_t = np.transpose(ratiosHe_t) 
    temp_t = np.transpose(temp_t)
    for obs in range(len(indices_obs)):
        # if label_obs[obs] == 'South pole':
        #     continue
        ax[obs].plot(tfb, ratiosHa_t[obs], label="Ha" if obs == 0 else "", color='C1')
        ax[obs].plot(tfb, ratiosHb_t[obs], label="Hb" if obs == 0 else "", color='darkviolet')
        ax[obs].plot(tfb, ratiosHe_t[obs], label="HeII4686" if obs == 0 else "", color='dodgerblue')

        axT[obs].plot(temp_t[obs], ratiosHa_t[obs], label="Ha" if obs == 0 else "", color='C1')
        axT[obs].plot(temp_t[obs], ratiosHb_t[obs], label="Hb" if obs == 0 else "", color='darkviolet')
        axT[obs].plot(temp_t[obs], ratiosHe_t[obs], label="HeII4686" if obs == 0 else "", color='dodgerblue')

        rBB = radiiBB[obs]
        axR[obs].plot(rBB/Rt, ratiosHa_t[obs], label="Ha" if obs == 0 else "", color='C1')
        axR[obs].plot(rBB/Rt, ratiosHb_t[obs], label="Hb" if obs == 0 else "", color='darkviolet')
        axR[obs].plot(rBB/Rt, ratiosHe_t[obs], label="HeII4686" if obs == 0 else "", color='dodgerblue')

        ax[obs].set_xlim(1, 2.25)
        ax[obs].set_xlabel(r"$t/t_{\rm fb}$")
        ax[obs].set_yscale('log')

        axT[obs].set_xlim(5e3, 1e5)
        axT[obs].set_xlabel("Temperature (K)")
        axT[obs].loglog()

        axR[obs].plot([1e-3, 1e3], [1e-3, 1e3], color='k', linestyle='--', linewidth=1)
        axR[obs].set_xlabel(r"$r/r_{\rm t}$")
        axR[obs].loglog()

        for a in [ax[obs], axT[obs], axR[obs]]:
            a.tick_params(axis='both', which='major', length=7, width=1)
            a.tick_params(axis='both', which='minor', length=4, width=.9)
            a.grid()
            if a != axR[obs]:
                a.set_title(label_obs[obs], fontsize=20)
            a.set_ylim(5e-3, 5e2)
            if obs == 0:
                a.set_ylabel(r"$L_{\lambda, \rm {line}}/L_{\lambda, \rm {cont}}$")
                a.legend(fontsize=16)
    fig.tight_layout()
    figTR.tight_layout()
# %%
