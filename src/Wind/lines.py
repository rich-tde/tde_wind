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

def continuum_luminosity(line, T, r):
    '''
    Calculate the continuum luminosity at a given wavelength according to Planck's law.
    Parameters:
    T : float
        Temperature in Kelvin.
    r : float
        Radius of the emitting region in cm.
    lam : float
        Wavelength in cm.
    Returns:
    L_lambda : float
        Continuum luminosity at the specified wavelength in erg/s/cm.
    '''
    if line == "Ha":
        lam = 6.5628e-5
    elif line == "Hb":
        lam = 4.8613e-5
    elif line == "HeII4686":
        lam = 4.6858e-5
    else:
        raise ValueError("line must be 'Ha', 'Hb', or 'HeII4686'")
    B_lambda = 2.0 * prel.h_cgs * prel.c_cgs**2 / lam**5 * (1.0 / (np.exp(prel.h_cgs * prel.c_cgs / (lam * prel.Kb_cgs * T)) - 1.0))
    L_lambda = 4 * np.pi**2 * r**2 * B_lambda 
    return L_lambda

def hydrogenic_levels_above_ground(n, chi_eV):
    '''
    Calculate the statistical weight and energy of a hydrogenic level above the ground state.
    Parameters:
    n : int
        Principal quantum number of the level.
    chi_eV : float
        Ionization energy in electron volts (eV).
    Returns:
    g : float
        Statistical weight of the level.
    E : float
        Energy of the level in erg.
    '''
    chi = chi_eV * prel.ev_to_erg
    g = 2.0 * n**2
    E = chi * (1.0 - 1.0 / n**2)
    return g, E

# for n in [1, 2, 3]:
#     g, E = hydrogenic_levels_above_ground(n, 13.6)
#     print(n, g, E / prel.ev_to_erg)

def hydrogenic_partition_function(T, chi_eV, n_max=30):
    ''' Compute Z = sum_{n=1}^{n_max} g_n * exp(-E_n / (k_B * T)) from Bradt: https://www.cambridge.org/us/files/5413/6681/8627/7706_Saha_equation.pdf
    Parameters:
    T : float
        Temperature in Kelvin.
    chi_eV : float
        Ionization energy in electron volts (eV).
    n_max : int, optional
        Maximum principal quantum number to include in the sum (default is 30).
    Returns:
    Z : float
        Partition function value.
    '''
    Z = 0.0
    for n in range(1, n_max + 1):
        g, E = hydrogenic_levels_above_ground(n, chi_eV)
        Z += g * np.exp(-E / (prel.Kb_cgs * T)) # should we have some weights? But n_max is already a cutoff, so maybe not
    return Z

# Test: At sufficiently low T, this should be close to 2 for H. 
# print(hydrogenic_partition_function(3000.0, 13.6, n_max=30))

def helium_i_partition_function(T, levels):
    '''
    Compute the internal partition function of neutral He I.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    levels : iterable of tuples
        Each element must be (g, E_eV), where

            g    = statistical weight of the level
            E_eV = excitation energy above the He I ground state [eV]

        The ground state should be included explicitly.

    Returns
    -------
    Z : float
        Partition function.
    '''
    Z = 0.0

    for g, E_eV in levels:
        E = E_eV * prel.ev_to_erg
        Z += g * np.exp(-E / (prel.Kb_cgs * T))

    return Z

# Z_HeI = helium_i_partition_function(1e4, helium_i_levels)

def saha_omega(T, n_e, chi_eV, Z_i, Z_ip1):
    ''' Eq 35 from Bradt
    '''
    chi = chi_eV * prel.ev_to_erg
    e_debroglie = prel.h_cgs / np.sqrt(2.0 * np.pi * m_e * prel.Kb_cgs * T)
    return 2.0 * Z_ip1 / Z_i * np.exp(-chi / (prel.Kb_cgs * T)) / (n_e * e_debroglie**3)

def solve_saha_h_he(T, rho, X_H=0.71, X_He=0.28, tol=1e-8, maxiter=200):
    n_H = X_H * rho / prel.m_p_cgs
    n_He = X_He * rho / (4.0 * prel.m_p_cgs)

    Z_HI = hydrogenic_partition_function(T, 13.6, 30) # neutral hydrogen
    Z_HII = 1.0 #  bare proton, so there are no bound electronic levels to sum 
    Z_HeI = helium_i_partition_function(T, helium_i_levels)
    Z_HeII = hydrogenic_partition_function(T, 54.418, 30) # He II is hydrogen-like: one electron around a nucleus
    Z_HeIII = 1.0 # bare helium nucleus, so again no bound electronic levels

    # first guess: If all hydrogen is ionized, each H contributes one electron and each He contributes up to two electrons 
    # since you have 1 electron from HI -> HII, and 1 electron from He HeI -> HeII and 1 electron from HeII -> HeIII
    # it's usually stable and not far from the final solution, so it should converge quickly
    n_e = n_H + 2.0 * n_He 

    for _ in range(maxiter):
        Omega_H = saha_omega(T, n_e, 13.6, Z_HI, Z_HII)
        n_HI = n_H / (1.0 + Omega_H)
        n_HII = n_H - n_HI

        Omega_He1 = saha_omega(T, n_e, 24.587, Z_HeI, Z_HeII)
        Omega_He2 = saha_omega(T, n_e, 54.418, Z_HeII, Z_HeIII)

        denom = 1.0 + Omega_He1 + Omega_He1 * Omega_He2
        n_HeI = n_He / denom
        n_HeII = n_HeI * Omega_He1
        n_HeIII = n_HeII * Omega_He2

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

# Test for n_i+1/n_i=Omega
# T_test = 1e5
# rho_test = 1e-9
# out = solve_saha_h_he(T_test, rho_test, X_H=1.0, X_He=0.0)
# omega = saha_omega(T_test, out["n_e"], 13.6, out["Z_HI"], out["Z_HII"])
# ratio = out["n_HII"] / out["n_HI"]
# print("Direct consistency check")
# print("relative diff =", abs(ratio - omega) / max(abs(omega), 1e-300))

# Test for Saha solver
# rho_test = 1e-6
# out = solve_saha_h_he(T = 3000.0, rho = rho_test, X_H=1.0, X_He=0.0)
# n_H = rho_test / prel.m_p_cgs
# frac_HII = out["n_HII"] / n_H
# frac_HI = out["n_HI"] / n_H
# print("Low-T, high-rho (expected: mostly neutral):")
# print("frac_HII =", frac_HII)
# print("frac_HI  =", frac_HI)
# print("n_e / n_H =", out["n_e"] / n_H)

# rho_test = 1e-10
# out = solve_saha_h_he(T = 1e6, rho = rho_test, X_H=1.0, X_He=0.0)
# n_H = rho_test / prel.m_p_cgs
# frac_HII = out["n_HII"] / n_H
# frac_HI = out["n_HI"] / n_H
# print("High-T, low-rho (expected: mostly ionized):")
# print("frac_HII =", frac_HII)
# print("frac_HI  =", frac_HI)
# print("n_e / n_H =", out["n_e"] / n_H)

def line_ratio_paper(line, T, rho, delta_r, r, v, X_H=0.7, X_He=0.28, n_p=None):
    # for A and lambda0 vslues: https://www.nist.gov/system/files/documents/srd/jpcrd382009565p.pdf
    # for H look at pag 573 (line 40, 41)
    # for HeII look at pag 576 (line 1)
    pops = solve_saha_h_he(T, rho, X_H=X_H, X_He=X_He)
    if n_p is None:
        n_p = pops["n_HII"]

    if line == "Ha": 
        chi_eV = 13.6
        n_starter = 3
        A = 6.4651e7 # transition 3->2 (retrive from NIST: H I, range 6560–6565 Å, 3d (j=5/2)->2p(j=3/2))
        lam0 = 6.5628e-5
        n_i = pops["n_HI"]
        Z_i = pops["Z_HI"]
        omega = saha_omega(T, pops["n_e"], chi_eV, Z_i, pops["Z_HII"])
    elif line == "Hb":
        chi_eV = 13.6
        n_starter = 4
        A = 8.4193e6 # transition 4->2
        lam0 = 4.8613e-5
        n_i = pops["n_HI"]
        Z_i = pops["Z_HI"]
        omega = saha_omega(T, pops["n_e"], chi_eV, Z_i, pops["Z_HII"])
    elif line == "HeII4686":
        chi_eV = 54.418
        n_starter = 4
        A = 2.2076e8 #  (retrive from NIST: He II, range 4680-4690 Å, 4f (j=7/2)-> 3d (j=5/2))
        lam0 = 4.6858e-5
        n_i = pops["n_HeII"]
        Z_i = pops["Z_HeII"]
        omega = saha_omega(T, pops["n_e"], chi_eV, Z_i, pops["Z_HeIII"])
    else:
        raise ValueError("line must be 'Ha', 'Hb', or 'HeII4686'") 

    g_u, E_u = hydrogenic_levels_above_ground(n_starter, chi_eV)

    # pref = (g_u / Z_i) * np.exp(-E_u / (prel.Kb_cgs * T))
    # const = (A * prel.h_cgs * prel.c_cgs / (2.0 * np.pi * prel.Kb_cgs))
    # ratio = pref * const / (1.0 + omega)  * (n_p * delta_r * lam0**2 / (T * v))

    n_u = n_i * g_u / Z_i * np.exp(-E_u / (prel.Kb_cgs * T))
    eta = n_u * A * prel.h_cgs * prel.c_cgs / lam0
    vol = 4 * np.pi * r**2 * delta_r
    Lline = eta * vol * prel.c_cgs / (lam0 * v)

    return Lline, pops, {"omega": omega, "g_u": g_u, "E_u": E_u, "Z_i": Z_i}

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
choice = 'split_stream'
# delta_r = 1e14

helium_i_levels = [
    # Configuration       Term       J    g = 2J + 1     E_exc [eV]
    (1.0,   0.000000000),  # 1s^2      1S       0
    (3.0,  19.819614525),  # 1s 2s     3S       1
    (1.0,  20.615774823),  # 1s 2s     1S       0
    (5.0,  20.964086889),  # 1s 2p     3P^o      2
    (3.0,  20.964096365),  # 1s 2p     3P^o      1
    (1.0,  20.964218851),  # 1s 2p     3P^o      0
    (3.0,  21.218022711),  # 1s 2p     1P^o      1
    # (3.0,  22.718466419),  # 1s 3s     3I S       0
]

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
snaps, tfb, Lum = data[:, 0], data[:, 1], data[:, 2]
snaps, Lum, tfb = op.sort_list([snaps, Lum, tfb], tfb, unique=True) 
snaps = snaps.astype(int)
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_obs, label_obs, color_obs, _, _= op.choose_observers(observers_xyz, choice)
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
    r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
    vr_ph, _, _ = op.to_spherical_components(vx_ph, vy_ph, vz_ph, x_ph, y_ph, z_ph) 
    vr_ph *= conversion_vel
    ratiosHa = np.zeros(len(temp_ph))
    ratiosHb = np.zeros(len(temp_ph))
    ratiosHe = np.zeros(len(temp_ph))
    fig, ax = plt.subplots(figsize = prel.set_size(columns=1))

    for i in range(len(temp_ph)):
        delta_r = r_ph[i] * prel.Rsol_cgs
        ratiosHa[i], _, _ = line_ratio_paper(
                                line="Ha",
                                T=temp_ph[i],
                                rho=d_ph[i],
                                delta_r=delta_r,
                                r=r_ph[i]*prel.Rsol_cgs,
                                v=vr_ph[i])
        
        ratiosHb[i], _, _ = line_ratio_paper(
                                line="Hb",
                                T=temp_ph[i],
                                rho=d_ph[i],
                                delta_r=delta_r,
                                r=r_ph[i]*prel.Rsol_cgs,
                                v=vr_ph[i])
        
        ratiosHe[i], _, _ = line_ratio_paper(
                                line="HeII4686",
                                T=temp_ph[i],
                                rho=d_ph[i],
                                delta_r=delta_r,
                                r=r_ph[i]*prel.Rsol_cgs,
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
            ax.scatter(temp_ph[index], ratiosHa[index], label="Ha" if i == 0 else "", s = 100, marker = markers[i], color='#d00000')
            ax.scatter(temp_ph[index], ratiosHb[index], label="Hb" if i == 0 else "", s = 100, marker = markers[i], color='#fb8500')
            ax.scatter(temp_ph[index], ratiosHe[index], label="HeII4686" if i == 0 else "", s = 100, marker = markers[i], color='#ffd166')
        
        # ax.scatter(temp_ph[88:103], ratiosHa[88:103], s = 100, color='#d00000', edgecolor='k')
        # ax.scatter(temp_ph[88:103], ratiosHb[88:103], s = 100, color='#fb8500', edgecolor='k')
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
            ax.scatter(temp_ph_sec[i], ratiosHa_sec[i], label="Ha" if i == 0 else "", marker = markers[i], s = 100, color = '#d00000')
            ax.scatter(temp_ph_sec[i], ratiosHb_sec[i], label="Hb" if i == 0 else "", marker = markers[i], s = 100, color = '#fb8500')
            ax.scatter(temp_ph_sec[i], ratiosHe_sec[i], label="HeII4686" if i == 0 else "", marker = markers[i], s = 100, color = '#ffd166')

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
    # print(Rt*prel.Rsol_cgs/1e14)
    x_testT = [1e3, 1e6]
    y_testT = op.draw_line(x_testT, [1e50, -1.3], 'powerlaw')
    x_testR = [10, 1e3]
    y_testR = op.draw_line(x_testR, [1e43, 1], 'powerlaw')
    dataBB = np.loadtxt(f'{abspath}/data/{folder}/wind/Tfit_intime_{choice}.txt', delimiter=',', skiprows=1, unpack=True)
    TempBB = dataBB[1:1+len(indices_obs)+1, :] # they're saved in cgs
    radiiBB = dataBB[len(indices_obs)+1:2*len(indices_obs)+1, :] # they're saved in cgs
    radiiBB /= prel.Rsol_cgs 
    LHa_t = []
    LHb_t = []
    LHe_t = []
    ratiosHa_t = []
    ratiosHb_t = []
    ratiosHe_t = []
    temp_t = []
    delta_r_t = []
    fig, ax = plt.subplots(1, 3, figsize= (24, 7))
    figTR, (axT, axR) = plt.subplots(2, len(indices_obs)-1, figsize=(8*len(indices_obs), 12))
    axes = np.concatenate([[ax[i] for i in range(3)] + [axT[i] for i in range(len(indices_obs)-1)] + [axR[i] for i in range(len(indices_obs)-1)]])
    for snap in snaps:
        time = tfb[snaps == snap][0]
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        x_ph, y_ph, z_ph, vx_ph, vy_ph, vz_ph, d_ph, temp_ph = \
            photo['x'], photo['y'], photo['z'], photo['vx'], photo['vy'], photo['vz'], photo['den'], photo['temp']
        r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
        vr_ph, _, _ = op.to_spherical_components(vx_ph, vy_ph, vz_ph, x_ph, y_ph, z_ph) 
        vr_ph *= conversion_vel
        LHa = np.zeros(len(temp_ph))
        LHb = np.zeros(len(temp_ph))
        LHe = np.zeros(len(temp_ph))
        ratiosHa = np.zeros(len(temp_ph))
        ratiosHb = np.zeros(len(temp_ph))
        ratiosHe = np.zeros(len(temp_ph))
        deltas_r = np.zeros(len(temp_ph))
        for i in range(len(temp_ph)):
            if vr_ph[i] < 0:
                continue
            delta_r = r_ph[i] * prel.Rsol_cgs
            deltas_r[i] = delta_r

            cont = continuum_luminosity("Ha", temp_ph[i], r_ph[i]*prel.Rsol_cgs)
            LHa[i], _, _ = line_ratio_paper(
                                    line="Ha",
                                    T=temp_ph[i],
                                    rho=d_ph[i],
                                    delta_r=delta_r,
                                    r=r_ph[i]*prel.Rsol_cgs,
                                    v=vr_ph[i])
            ratiosHa[i] = LHa[i] / cont

            cont = continuum_luminosity("Hb", temp_ph[i], r_ph[i]*prel.Rsol_cgs)
            LHb[i], _, _ = line_ratio_paper(
                                    line="Hb",
                                    T=temp_ph[i],
                                    rho=d_ph[i],
                                    delta_r=delta_r,
                                    r=r_ph[i]*prel.Rsol_cgs,
                                    v=vr_ph[i])
            ratiosHb[i] = LHb[i] / cont

            cont = continuum_luminosity("HeII4686", temp_ph[i], r_ph[i]*prel.Rsol_cgs)
            LHe[i], _, _ = line_ratio_paper(
                                    line="HeII4686",
                                    T=temp_ph[i],
                                    rho=d_ph[i],
                                    delta_r=delta_r,
                                    r=r_ph[i]*prel.Rsol_cgs,
                                    v=vr_ph[i])
            ratiosHe[i] = LHe[i] / cont
        LHa_sec = np.zeros(len(indices_obs))
        LHb_sec = np.zeros(len(indices_obs))
        LHe_sec = np.zeros(len(indices_obs))
        ratiosHa_sec = np.zeros(len(indices_obs))
        ratiosHb_sec = np.zeros(len(indices_obs))
        ratiosHe_sec = np.zeros(len(indices_obs))
        temp_sec = np.zeros(len(indices_obs))
        delta_r_sec = np.zeros(len(indices_obs))
        for i, indices in enumerate(indices_obs): # you need that if the splitting is not homogeneous
            LHa_sec[i] = np.median(LHa[indices])
            LHb_sec[i] = np.median(LHb[indices])
            LHe_sec[i] = np.median(LHe[indices]) 
            ratiosHa_sec[i] = np.median(ratiosHa[indices])
            ratiosHb_sec[i] = np.median(ratiosHb[indices])
            ratiosHe_sec[i] = np.median(ratiosHe[indices])
            temp_sec[i] = np.median(temp_ph[indices])
            delta_r_sec[i] = np.median(deltas_r[indices])
            if snap == 151:
                print(f'delta_R/1e15cm {label_obs[i]}', delta_r_sec[i]/1e15)
        LHa_t.append(LHa_sec)
        LHb_t.append(LHb_sec)
        LHe_t.append(LHe_sec)
        ratiosHa_t.append(ratiosHa_sec)
        ratiosHb_t.append(ratiosHb_sec)
        ratiosHe_t.append(ratiosHe_sec)
        temp_t.append(temp_sec)
        delta_r_t.append(delta_r_sec)

    LHa_t = np.transpose(LHa_t)
    LHb_t = np.transpose(LHb_t)
    LHe_t = np.transpose(LHe_t)
    ratiosHa_t = np.transpose(ratiosHa_t)
    ratiosHb_t = np.transpose(ratiosHb_t)
    ratiosHe_t = np.transpose(ratiosHe_t) 
    temp_t = np.transpose(temp_t)
    delta_r_t = np.transpose(delta_r_t)

    handles_color = []
    labels_color = []
    for obs in range(len(indices_obs)-1):
        if label_obs[obs] == 'South pole':
            continue
        line = ax[0].plot(tfb, ratiosHa_t[obs], label = label_obs[obs], color = color_obs[obs])[0]
        handles_color.append(line)
        labels_color.append(label_obs[obs])
        ax[1].plot(tfb, ratiosHb_t[obs], label = label_obs[obs], color = color_obs[obs])
        ax[2].plot(tfb, ratiosHe_t[obs], label = label_obs[obs], color = color_obs[obs])

        TBB = TempBB[obs] # temp_t[obs]
        axT[obs].plot(TBB, LHa_t[obs], label=r"H$\alpha$" if obs == 0 else "", color='#d00000')
        axT[obs].plot(TBB, LHb_t[obs], label=r"H$\beta$" if obs == 0 else "", color='#fb8500')
        axT[obs].plot(TBB, LHe_t[obs], label=r"HeII4686" if obs == 0 else "", color='#ffd166')

        rBB = radiiBB[obs]
        axR[obs].plot(rBB/Rt, LHa_t[obs], label=r"H$\alpha$" if obs == 0 else "", color='#d00000')
        axR[obs].plot(rBB/Rt, LHb_t[obs], label=r"H$\beta$" if obs == 0 else "", color='#fb8500')
        axR[obs].plot(rBB/Rt, LHe_t[obs], label=r"HeII4686" if obs == 0 else "", color='#ffd166')

        axT[obs].set_xlim(5e3, 1e5)
        axT[obs].plot(x_testT, y_testT, color='k', linestyle='--', linewidth=1)
        axT[obs].set_title(label_obs[obs], fontsize=20)
        axT[obs].loglog()
        axT[obs].set_xlabel("Temperature (K)")

        axR[obs].plot(x_testR, y_testR, color='k', linestyle='--', linewidth=1)
        axR[obs].loglog()
        axR[obs].set_xlabel(r"$r/r_{\rm t}$", fontsize= 30)
        axR[obs].set_xlim(20, 1e3)

    for a in axes:
        a.tick_params(axis='both', which='major', length=10, width=1)
        a.tick_params(axis='both', which='minor', length=6, width=.9)
        a.grid()
        a.set_ylabel(r"$L_{\lambda, \rm {line}}/L_{\lambda, \rm {cont}}$", fontsize= 30)
        if a in ax[0:3]:
            if a != ax[2]:
                a.set_ylim(5e-2, 5e2)
            a.set_xlim(1, 2.25)
            a.set_yscale('log')
            a.set_xlabel(r"$t/t_{\rm fb}$", fontsize= 30)
        else:
            a.set_ylim(1e43, 1e47)

    ax[2].set_ylim(5e-3, 5)
    # ax[2].legend(fontsize=18)

    legend_colors = fig.legend(
                handles=handles_color,
                labels=labels_color,
                loc='upper center',
                bbox_to_anchor=(0.525, 0.022),  # centered, near bottom of figure
                ncol=len(labels_color),
                fontsize=22) 

    for i in range(3): # you need the new ylim
        ax[i].set_title(r'H$\alpha$' if i == 0 else r'H$\beta$' if i == 1 else r'HeII $\lambda$4686', fontsize= 30)
        # ax[i].text(1.08, 0.4*ax[i].get_ylim()[1], r'H$\alpha$' if i == 0 else r'H$\beta$' if i == 1 else r'HeII4686', fontsize=25)

    fig.tight_layout()
    figTR.tight_layout()
    fig.savefig(f'{abspath}/Figs/2.paperWind/line_ratio_time_evol_{choice}.pdf', bbox_inches = 'tight')
# %%
