
abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import csv
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
import Utilities.prelude as prel

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

lam_g_min, lam_g_max = 4000*u.AA, 5500*u.AA
lam_r_min, lam_r_max = 5500*u.AA, 7000*u.AA
lam_uv_min, lam_uv_max = 2300*u.AA, 2900*u.AA  # ULTRASAT-like

def fluxBol_at_redshifts(Lbol_cgs, DL_cgs):
    F = Lbol_cgs / (4 * np.pi * DL_cgs**2)   
    return F

def mBol_from_flux(F_cgs, F0 = 2.5e-5):
    m_bol = -2.5 * np.log10(F_cgs/F0) 
    return m_bol

def mBol_from_lum(L_cgs, DL_pc, K = 0):
    Mabs = 4.74 - 2.5 * np.log10(L_cgs / 3.828e33)
    m_bol = Mabs + 5 * np.log10(DL_pc/10) + K
    return m_bol

def B_lambda(lam, T_K):
    """Return Planck function in erg s^-1 cm^-2 sr^-1 cm^-1."""
    h = const.h.cgs.value
    c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    lam_cm = lam.to(u.cm).value
    x = h * c / (lam_cm * k_B * T_K)
    return (2 * h * c**2 / lam_cm**5) / (np.expm1(x))

def band_fraction(lam_min, lam_max, T, z):
    """Fraction of bolometric flux in observed band (top-hat)."""
    # Convert to rest-frame wavelengths
    lam_rest_min = lam_min / (1 + z)
    lam_rest_max = lam_max / (1 + z)
    
    lam_vals = np.linspace(lam_rest_min.value, lam_rest_max.value, 3000) * lam_rest_min.unit
    B_vals = B_lambda(lam_vals, T)
    
    # Integrate over wavelength in cm
    lam_cm = lam_vals.to(u.cm).value
    integral_band = np.trapz(B_vals, lam_cm) 
    
    sigma_sb = const.sigma_sb.cgs.value
    flux_total = sigma_sb * T**4 / np.pi  # B integrated over all λ per steradian
    
    return integral_band / flux_total

def compute_m_ab(L_bol, T, z, lam_min, lam_max, band_name="band"):
    """Compute AB magnitude for a top-hat band."""
    # Fractional flux in band
    frac = band_fraction(lam_min, lam_max, T, z)
    
    # Flux integrated over band
    F_bol = (L_bol / (4 * np.pi * cosmo.luminosity_distance(z).to(u.cm)**2)).cgs.value
    F_band = F_bol * frac  # erg s^-1 cm^-2
    
    # Convert to average F_nu
    c = const.c.cgs.value
    lam_center = 0.5 * (lam_min + lam_max)
    delta_lam = (lam_max - lam_min)
    F_nu = F_band * lam_center.to(u.cm).value**2 / (c * delta_lam.to(u.cm).value)
    
    # AB magnitude
    m_ab = -2.5 * np.log10(F_nu) - 48.6
    
    # print(f"\n--- {band_name} band ---")
    # print(f"Fraction of bolometric flux: {frac:.4f}")
    # print(f"F_band = {F_band:.3e} erg/s/cm^2")
    # print(f"m_AB = {m_ab:.2f}")
    return m_ab

data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
Lum = data[:, 2]
Lum_max = np.max(Lum)
Temp_max = 2.6e4
z = [0.05, 0.1, 0.4]

cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # implies Omega_Lambda = 0.7
# Compute luminosity distance in Mpc and convert to cm
DL = cosmo.luminosity_distance(z)

m = np.zeros(len(z))
for i, dist in enumerate(DL):
    DL_pc = dist.to(u.pc).value
    DL_cm = dist.to(u.cm).value
    # m[i] = m_from_lum(Lum_max, DL_pc)  # C is the bolometric correction
    F_cgs = fluxBol_at_redshifts(Lum_max, DL_cm)
    print(F_cgs)
    m[i] = mBol_from_flux(F_cgs)
print(z)
print(m)

z_chosen = 0.05
m_g = compute_m_ab(Lum_max, Temp_max, z_chosen, lam_g_min, lam_g_max, "g")
m_r = compute_m_ab(Lum_max, Temp_max, z_chosen, lam_r_min, lam_r_max, "r")
m_uv = compute_m_ab(Lum_max, Temp_max, z_chosen, lam_uv_min, lam_uv_max, "UV")

print("\nAB magnitudes at z = ", z_chosen)
print(f"UV-band: {m_uv:.2f}")
print(f"g-band: {m_g:.2f}")
print(f"r-band: {m_r:.2f}")

