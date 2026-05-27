
abspath = '/Users/paolamartire/shocks/'
import sys
sys.path.append(abspath)

import csv
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
import Utilities.prelude as prel
from scipy.optimize import brentq

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
Lum = data[:, 2]
Lum_max = np.max(Lum)
Temp_max = 2.5e4
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # implies Omega_Lambda = 0.7

def ev_to_nu(E_ev):
    """Convert photon energy in eV to frequency in Hz."""
    E_erg = E_ev * 1.60218e-12  # erg
    h = const.h.cgs.value       # erg*s
    nu = E_erg / h
    return nu

def mBol_from_flux(Lbol_cgs, z, F0 = 2.5e-5):
    DL_cgs = cosmo.luminosity_distance(z).to(u.cm).value
    Fobs_cgs = Lbol_cgs / (4 * np.pi * DL_cgs**2)   
    m_bol = -2.5 * np.log10(Fobs_cgs/F0) 
    return m_bol

def mBol_from_lum(L_cgs, DL_pc, K = 0):
    Mabs = 4.74 - 2.5 * np.log10(L_cgs / 3.828e33)
    m_bol = Mabs + 5 * np.log10(DL_pc/10) + K
    return m_bol
    
def B_nu(nu, T):
    h = const.h.cgs.value
    k_B = const.k_B.cgs.value
    c = const.c.cgs.value
    return (2 * h * nu**3 / c**2) / (np.expm1(h * nu / (k_B * T)))

def m_ab_band(L_bol, T, z, lamnu_center, central_value = 'lambda'):
    '''Find AB magnitude in a band centered at lamnu_center (can be wavelength or frequency)
    Parameters:
    -----------
    L_bol: float
        Bolometric luminosity in erg/s
    T: float        
        Blackbody temperature in K
    z: float
        Redshift
    lamnu_center: astropy Quantity
        Central wavelength (if central_value='lambda') or frequency (if central_value='frequency') of the band
    central_value: str
        'lambda' or 'frequency' to specify the type of lamnu_center
    Returns:
    --------
    m_AB: float
        AB magnitude in the specified band
    '''
    # Luminosity distance
    DL = cosmo.luminosity_distance(z).to(u.cm).value
    # Observed bolometric flux
    F_obs = L_bol / (4 * np.pi * DL**2)
     
    # Central frequency (in case you have to correct lambda for redshift, as lam_center = lam_center/(1+z))
    if central_value == 'lambda':
        nu_0 = (const.c / lamnu_center).cgs.value
    elif central_value == 'frequency':
        nu_0 = lamnu_center.cgs.value
    
    # Fraction of flux in band: F_band = F_obs * (pi * B_nu / sigma T^4) * delta_nu
    sigma_sb = const.sigma_sb.cgs.value
    # lam_min_cm = lam_min.to(u.cm).value
    # lam_max_cm = lam_max.to(u.cm).value
    # delta_nu = const.c.cgs.value / lam_min_cm - const.c.cgs.value / lam_max_cm
    
    F_band = F_obs * (np.pi * B_nu(nu_0, T) / (sigma_sb * T**4)) 
    
    # AB magnitude
    F_band_Jy = F_band / 1e-23  # erg/s/cm^2/Hz -> Jy
    m_AB = -2.5 * np.log10(F_band_Jy / 3631)
    return m_AB

def find_horizon(L_bol, T, lam_center, m_lim, z_max=1.0):
    """
    Compute maximum redshift where the object is brighter than m_lim.
    """
    # Function whose root gives m_AB(z) - m_lim = 0
    f = lambda z: m_ab_band(L_bol, T, z, lam_center) - m_lim
    
    # Solve numerically for z
    try:
        z_horizon = brentq(f, 1e-5, z_max)
    except ValueError:
        z_horizon = np.nan  # no solution within [0, z_max]
    return z_horizon

# def F_nu_band_ev(L_bol, T, z, E_center_ev):
#     """Compute observed flux density in erg/s/cm²/Hz for a band centered at energy E (eV)."""
#     nu_0 = ev_to_nu(E_center_ev)
#     DL = cosmo.luminosity_distance(z).to(u.cm).value
#     F_obs = L_bol / (4 * np.pi * DL**2)
#     sigma_sb = const.sigma_sb.cgs.value
#     F_band = F_obs * (np.pi * B_nu(nu_0, T) / sigma_sb / T**4)
#     return F_band

# def find_horizon_flux_ev(L_bol, T, E_center_ev, F_nu_lim, z_max= 1.0):
#     """Redshift horizon for given flux limit and band energy in eV."""
#     f = lambda z: F_nu_band_ev(L_bol, T, z, E_center_ev) - F_nu_lim
#     try:
#         z_horizon = brentq(f, 1e-3, z_max)
#     except ValueError:
#         z_horizon = np.nan  # source never below flux limit
#     return z_horizon

#
## MAIN
# 
# Bands 
# lam_g_min, lam_g_max = 3676*u.AA, 5613.82*u.AA
lam_g_mean = 4829.50*u.AA
# lam_r_min, lam_r_max = 5497.60*u.AA, 7394.40*u.AA 
lam_r_mean = 6463.75*u.AA

lamLSST_g_mean = 4746.4*u.AA
lamLSST_r_mean = 6201.5*u.AA

lam_uv_min, lam_uv_max = 2300*u.AA, 2900*u.AA # ULTRASAT-like
lam_uv_mean = (lam_uv_min + lam_uv_max)/2

evROS_min, evROS_max = 0.3e3, 2.3e3# eROSITA band in eV
evbdaeROS_mean = (evROS_min + evROS_max)/2
nueROS_mean = ev_to_nu(evbdaeROS_mean)

#%% Compute luminosity distance in Mpc and convert to cm
z_arr = [0.05, 0.1, 0.4]
m = np.zeros(len(z_arr))
print('Magnitude')
for i, z in enumerate(z_arr):
    # print(F_cgs)
    m[i] = mBol_from_flux(Lum_max, z)
print(z_arr)
print(m)

#%%
z_chosen = 0.05
# m_g = compute_m_ab(Lum_max, Temp_max, z_chosen, lam_g_min, lam_g_max, "g")
m_g = m_ab_band(Lum_max, Temp_max, z_chosen, lam_g_mean)
m_r = m_ab_band(Lum_max, Temp_max, z_chosen, lam_r_mean)
print("\nZTF AB magnitudes at z = ", z_chosen)
print(f"g-band: {m_g:.2f}")
print(f"r-band: {m_r:.2f}")

m_gLSST = m_ab_band(Lum_max, Temp_max, z_chosen, lamLSST_g_mean)
m_rLSST = m_ab_band(Lum_max, Temp_max, z_chosen, lamLSST_r_mean)
print("\nLSST AB magnitudes at z = ", z_chosen)
print(f"g-band: {m_gLSST:.2f}")
print(f"r-band: {m_rLSST:.2f}")

m_uv_ULTRASAT = m_ab_band(Lum_max, Temp_max, z_chosen, lam_uv_mean)
print("\nULTRASAT-like UV-band AB magnitude at z = ", z_chosen)
print(f"UV-band: {m_uv_ULTRASAT:.2f}")

## Compute horizon
m_lim_ZTF = 20.5
m_lim_LSST = 24.7
m_lim_ULTRASAT = 22.4
flux_eROS = 3e-13 # erg/s/cm^2
# fluz_eROS_Hz = flux_eROS / nueROS_mean
# F_eROS_Jy = fluz_eROS_Hz / 1e-23  # erg/s/cm^2/Hz -> Jy
# m_lim_eROS = -2.5 * np.log10(F_eROS_Jy / 3631)
distance_eROS_Mpc = np.sqrt(0.1 * Lum_max / (4 * np.pi * flux_eROS)) / 3.086e24  # in Mpc (https://en.wikipedia.org/wiki/Parsec 1pc = 3.086e16 m)
print("\neROSITA limiting distance (Mpc):", distance_eROS_Mpc)

z_horizon_r_ZTF = find_horizon(Lum_max, Temp_max, lam_r_mean, m_lim_ZTF)
z_horizon_r_LSST = find_horizon(Lum_max, Temp_max, lamLSST_r_mean, m_lim_LSST)
z_horizon_uv_ULTRASAT = find_horizon(Lum_max, Temp_max, lam_uv_mean, m_lim_ULTRASAT)
# z_horizon_eROSITA = find_horizon(0.1*Lum_max, Temp_max, lambdaeROS_mean, m_lim_eROS)
print("\nHorizon redshift:")
print(f"ZTF r-band (m_lim = {m_lim_ZTF}): z_horizon = {z_horizon_r_ZTF:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_r_ZTF).to(u.Mpc).value:.1f}")
print(f"LSST r-band (m_lim = {m_lim_LSST}): z_horizon = {z_horizon_r_LSST:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_r_LSST).to(u.Mpc).value:.1f}")
print(f"ULTRASAT UV-band (m_lim = {m_lim_ULTRASAT}): z_horizon = {z_horizon_uv_ULTRASAT:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_uv_ULTRASAT).to(u.Mpc).value:.1f}")
# print(f"eROSITA (flux_lim = {flux_eROS} erg/s/cm^2): z_horizon = {z_horizon_eROSITA:.3f}")


