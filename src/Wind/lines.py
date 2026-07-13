import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'

import numpy as np
import matplotlib.pyplot as plt
import Utilities.prelude as prel
from Utilities import operators as op

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
    # (https://www.cambridge.org/us/files/5413/6681/8627/7706_Saha_equation.pdf)
    chi = chi_eV * prel.ev_to_erg
    Z = 0.0
    for n in range(1, n_max + 1):
        g, E = hydrogenic_levels(n, chi_eV)
        Z += g * np.exp(-E / (prel.Kb_cgs * T))
    return Z

def saha_omega(T, n_e, chi_eV, Z_i, Z_ip1):
    chi = chi_eV * prel.ev_to_erg
    e_debroglie = prel.h_cgs / np.sqrt(2.0 * np.pi * m_e * prel.Kb_cgs * T)
    return 2.0 * Z_ip1 / Z_i * np.exp(-chi / (prel.Kb_cgs * T)) / (n_e * e_debroglie**3)

def solve_saha_h_he(T, rho, X_H=0.71, X_He=0.28, tol=1e-8, maxiter=200):
    n_H = X_H * rho / prel.m_p_cgs
    n_He = X_He * rho / (4.0 * prel.m_p_cgs)

    Z_HI = hydrogenic_partition_function(T, 13.6, 30)
    Z_HII = 1.0
    Z_HeI = 1.0
    Z_HeII = hydrogenic_partition_function(T, 54.418, 30)
    Z_HeIII = 1.0

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
    pops = solve_saha_h_he(T, rho, X_H=X_H, X_He=X_He)
    if n_p is None:
        n_p = pops["n_HII"]

    if line == "Ha": 
        chi_eV = 13.6
        n_upper = 3
        A = 4.410e7 # transition 3->2
        lam0 = 6.5628e-5
        Z_i = pops["Z_HI"]
        omega = saha_omega(T, pops["n_e"], 13.6, pops["Z_HI"], pops["Z_HII"])
    elif line == "Hb":
        chi_eV = 13.6
        n_upper = 4
        A = 8.419e6 # transition 4->2
        lam0 = 4.8613e-5
        Z_i = pops["Z_HI"]
        omega = saha_omega(T, pops["n_e"], 13.6, pops["Z_HI"], pops["Z_HII"])
    elif line == "HeII4686":
        chi_eV = 54.418
        n_upper = 4
        A = 8.215e6
        lam0 = 4.6857e-5
        Z_i = pops["Z_HeII"]
        omega = saha_omega(T, pops["n_e"], 54.418, pops["Z_HeII"], pops["Z_HeIII"])
    else:
        raise ValueError("line must be 'Ha', 'Hb', or 'HeII4686'")

    g_u, E_u = hydrogenic_levels(n_upper, chi_eV)

    pref = (g_u / Z_i) * np.exp(-E_u / (prel.Kb_cgs * T))
    const = (A * prel.h_cgs * prel.c_cgs / (2.0 * np.pi * prel.Kb_cgs))
    ratio = pref * const / (1.0 + omega)  * (n_p * delta_r * lam0**2 / (T * v))

    return ratio, pops, {"omega": omega, "g_u": g_u, "E_u": E_u, "Z_i": Z_i}



m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 109
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
x_ph, y_ph, z_ph, vx_ph, vy_ph, vz_ph, d_ph, temp_ph = \
    photo['x'], photo['y'], photo['z'], photo['vx'], photo['vy'], photo['vz'], photo['den'], photo['temp']
vr_ph, _, _ = op.to_spherical_components(vx_ph, vy_ph, vz_ph, x_ph, y_ph, z_ph) 
vr_ph *= conversion_vel
delta_r = 1e14

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

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(temp_ph, ratiosHa, label="Ha", color='C1')
ax.scatter(temp_ph, ratiosHb, label="Hb", color='forestgreen')
ax.scatter(temp_ph, ratiosHe, label="HeII4686", color='dodgerblue')
ax.scatter(temp_ph[88:103], ratiosHa[88:103], color='C1', edgecolor='k')
ax.scatter(temp_ph[88:103], ratiosHb[88:103], color='forestgreen', edgecolor='k')
ax.scatter(temp_ph[88:103], ratiosHe[88:103], color='dodgerblue', edgecolor='k')

ax.set_xlabel("Temperature")
ax.set_ylabel("Line Ratio")
ax.loglog()
ax.set_xlim(5e3, 1e5)
ax.set_ylim(5e-3, 1e3)
ax.tick_params(axis='both', which='major', length=7, width=1)
ax.tick_params(axis='both', which='minor', length=4, width=.9)
ax.legend(fontsize=16)
fig.suptitle(f"Snap {snap}", fontsize=20)
plt.tight_layout()

