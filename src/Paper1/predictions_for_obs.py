
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

def m_ab_band_fromLbol(L_bol, T, z, lamnu_center, central_value = 'lambda'):
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

def m_ab_band_fromLband(L_band, z, nu_min, nu_max):
    """L_band: luminosity integrated over the band [erg/s]
    nu_min, nu_max: frequency limits of the band [Hz]
    """
    # Luminosity distance
    DL = cosmo.luminosity_distance(z).to(u.cm).value

    # Observed  flux
    F_band = L_band / (4 * np.pi * DL**2)
    
    # Bandwidth [Hz]
    delta_nu = np.abs(nu_max - nu_min)

    # Mean flux density [erg/s/cm^2/Hz]
    F_nu = F_band / delta_nu

    # Jy
    F_nu_Jy = F_nu / 1e-23

    m_AB = -2.5 * np.log10(F_nu_Jy / 3631)
    return m_AB

def find_horizon(L, T, lam_center, m_lim, z_max=1.0, which_L = 'bol', nu_min=None, nu_max=None):
    """
    Compute maximum redshift where the object is brighter than m_lim.
    """
    # Function whose root gives m_AB(z) - m_lim = 0
    if which_L == 'bol':
        f = lambda z: m_ab_band_fromLbol(L, T, z, lam_center) - m_lim
    elif which_L == 'band':
        f = lambda z: m_ab_band_fromLband(L, z, nu_min, nu_max) - m_lim
    else:
        raise ValueError("which_L must be 'bol' or 'band'")

    # Solve numerically for z
    try:
        z_horizon = brentq(f, 1e-5, z_max)
    except ValueError:
        print(f"No solution for z_horizon within [0, {z_max}] for L={L}")
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


if __name__ == "__main__":
    # Bands 
    evROS_min, evROS_max = 0.2e3, 2.3e3# eROSITA band in eV
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
    m_g = m_ab_band_fromLbol(Lum_max, Temp_max, z_chosen, prel.lamZTF_g_mean)
    m_r = m_ab_band_fromLbol(Lum_max, Temp_max, z_chosen, prel.lamZTF_r_mean)
    print("\nZTF AB magnitudes at z = ", z_chosen)
    print(f"g-band: {m_g:.2f}")
    print(f"r-band: {m_r:.2f}")

    m_gLSST = m_ab_band_fromLbol(Lum_max, Temp_max, z_chosen, prel.lamLSST_g_mean*u.AA)
    m_rLSST = m_ab_band_fromLbol(Lum_max, Temp_max, z_chosen, prel.lamLSST_r_mean*u.AA)
    print("\nLSST AB magnitudes at z = ", z_chosen)
    print(f"g-band: {m_gLSST:.2f}")
    print(f"r-band: {m_rLSST:.2f}")

    m_uv_ULTRASAT = m_ab_band_fromLbol(Lum_max, Temp_max, z_chosen, prel.lam_ULTR_mean)
    print("\nULTRASAT-like UV-band AB magnitude at z = ", z_chosen)
    print(f"UV-band: {m_uv_ULTRASAT:.2f}")

    ## Compute horizon
    flux_eROS = 3e-13 # erg/s/cm^2
    # fluz_eROS_Hz = flux_eROS / nueROS_mean
    # F_eROS_Jy = fluz_eROS_Hz / 1e-23  # erg/s/cm^2/Hz -> Jy
    # m_lim_eROS = -2.5 * np.log10(F_eROS_Jy / 3631)
    distance_eROS_Mpc = np.sqrt(0.1 * Lum_max / (4 * np.pi * flux_eROS)) / 3.086e24  # in Mpc (https://en.wikipedia.org/wiki/Parsec 1pc = 3.086e16 m)
    print("\neROSITA limiting distance (Mpc):", distance_eROS_Mpc)

    z_horizon_r_ZTF = find_horizon(Lum_max, Temp_max, prel.lamZTF_r_mean * u.AA, prel.mr_lim_ZTF)
    z_horizon_r_LSST = find_horizon(Lum_max, Temp_max, prel.lamLSST_r_mean * u.AA, prel.mr_lim_Rubin)
    z_horizon_uv_ULTRASAT = find_horizon(Lum_max, Temp_max, prel.lam_ULTR_mean * u.AA, prel.m_lim_ULTRASAT)
    # z_horizon_eROSITA = find_horizon(0.1*Lum_max, Temp_max, lambdaeROS_mean, m_lim_eROS)
    print(f"\nHorizon redshift with bolometric L = {Lum_max:.2e} erg/s:")
    print(f"ZTF r-band (m_lim = {prel.mr_lim_ZTF}): z_horizon = {z_horizon_r_ZTF:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_r_ZTF).to(u.Mpc).value:.1f}")
    print(f"LSST r-band (m_lim = {prel.mr_lim_Rubin}): z_horizon = {z_horizon_r_LSST:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_r_LSST).to(u.Mpc).value:.1f}")
    print(f"ULTRASAT UV-band (m_lim = {prel.m_lim_ULTRASAT}): z_horizon = {z_horizon_uv_ULTRASAT:.3f}, in Mpc = {cosmo.luminosity_distance(z_horizon_uv_ULTRASAT).to(u.Mpc).value:.1f}")
    # print(f"eROSITA (flux_lim = {flux_eROS} erg/s/cm^2): z_horizon = {z_horizon_eROSITA:.3f}")

    # %% PAPER 2
    def L_from100(flux):
        """Convert flux in erg/s/cm^2 to luminosity in erg/s at 100 Mpc."""
        D = 100 * 3.086e24  # 100 Mpc in cm
        L = flux * 4 * np.pi * D**2
        return L

    # F_g = 3600 Jy * nu * 10^(-2*m/5) with Jy = 1e-23 erg/s/cm^2/Hz
    # limF_ZTF = 3600 * 10**(-2*prel.mg_lim_ZTF/5)
    limF_LSST = 3600*1e-23 * prel.c_cgs/4.8e-5 *10**(-2*prel.mg_lim_Rubin/5) # (=3.6e-15 erg/cm^2/s) at lambda = 480 nm and m = 24.8 in the g-band)
    limF_ULTRASAT = 3600*1e-23 * prel.c_cgs/2.6e-5 * 10**(-2*prel.m_lim_ULTRASAT/5) # at lambda = 260 nm and m = 22.5
    # limF_eROSITA = 5e-14 # erg/s/cm^2 (https://erosita.mpe.mpg.de/dr1/ for 100s exposure time, Fig. 6 https://arxiv.org/pdf/2401.17305)
    limF_eROSITA = 1e-13 # erg/s/cm^2 (Table 1 https://link.springer.com/article/10.1007/s11433-024-2600-3)
    limF_Einstein = 3e-11 # erg/s/cm^2 (Table 1 https://link.springer.com/article/10.1007/s11433-024-2600-3)
    print(f"\nLimiting luminosities in erg/s at 100 Mpc:")
    print(f"LSST g-band (m_lim = {prel.mg_lim_Rubin}): L_lim = {L_from100(limF_LSST):.2e}")
    print(f"ULTRASAT UV-band (m_lim = {prel.m_lim_ULTRASAT}): L_lim = {L_from100(limF_ULTRASAT):.2e}")
    print(f"eROSITA (flux_lim = {limF_eROSITA} erg/s/cm^2): L_lim = {L_from100(limF_eROSITA):.2e}")
    print(f"Einstein (flux_lim = {limF_Einstein} erg/s/cm^2): L_lim = {L_from100(limF_Einstein):.2e}")

# %%
