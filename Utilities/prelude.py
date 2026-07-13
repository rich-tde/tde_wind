import numpy as np
import matplotlib.colors as colors
# import colorcet 
# import cmocean #to call the color cmocean.cm.[color]
# colorblind friendly colors: cividis, viridis, plasma, magma, inferno, Gray

# User lines
import sys
sys.path.append('/Users/paolamartire/shocks')
import astropy.units as u

# Constants
m_p_cgs = 1.6726219e-24 # [g] proton mass
c_cgs = 2.99792458e10 #[cm/s]
h_cgs = 6.62607015e-27 #[gcm^2/s]
Kb_cgs = 1.380649e-16 #[gcm^2/s^2K]
alpha_cgs = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
sigmaB_cgs = 5.670374419e-5 # [erg/cm^2/s/K^4] stefan-boltzmann constant
sigma_T_cgs = 6.6524e-25 #[cm^2] thomson cross section
G_cgs = 6.6743e-8 # cgs
Rsol_cgs = 7e10 #6.957e10 [cm]
Rsol_AU = 0.00465 # [AU]
Msol_cgs = 2e33 #1.98847e33 [g] 
ev_toK = 11604.505 # [K/eV] conversion factor from eV to K
ev_to_erg = 1.602176634e-12 # conversion factor from eV to erg
Hz_toK = 4.799e-11 # Hz to K
ev_toHz = 2.417989e14 # eV to Hz

# Solar and SI units
c_SI = 2.99e8 #m
G_SI = 6.6743e-11 # SI 
Msol_SI = 2e30 #1.98847e30 # kg
Rsol_SI = 7e8 #6.957e8 # m

# Sim units 
G = 1
solarR_to_au = 215

# Converters
tsol_cgs = 1603 #np.sqrt(Rsol_cgs**3 / (Msol_cgs*G_cgs )) # from RICH. Follows from G = 1. To pass from code units to cgs, you multiply by tsol_cgs
csol_cgs = c_cgs / (Rsol_cgs/tsol_cgs) # c in code units
den_converter = Msol_cgs / Rsol_cgs**3
en_den_converter = Msol_cgs / (Rsol_cgs  * tsol_cgs**2 ) # Energy Density converter
en_converter = Msol_cgs * Rsol_cgs**2 / tsol_cgs**2 # Energy converter
m_p_sol = m_p_cgs / Msol_cgs # [g] proton mass in code units

# Healpy
import healpy as hp
NSIDE = 4 # observers = 12 * NSIDE **2
NPIX = hp.nside2npix(NSIDE)#  int(NSIDE * 96)

# TOPS abundancies in the order you have to put them to have opacity (https://aphysics2.lanl.gov/apps/)
# NB: these are number fractions, not mass fractions
X_nf = 0.9082339738214822 #1 
He_nf = 0.09082339738214791 #2 
C_nf = 0.0002229450033069998 #6 
N_nf = 7.730324827099994e-05 #7 
O_nf = 0.00044483376340899965 #8 
Ne_nf = 9.082339738199991e-05 #10 
Mg_nf = 3.149181875599997e-05 #12 
Si_nf = 3.149181875599997e-05 #14 
S_nf = 1.6527141419999988e-05 #16 
Fe_nf = 2.5597511293999976e-05 #26 
Ni_nf = 1.6150937749999985e-06 #28

# Frequency array for BB in src.fld_curve
f_min = Kb_cgs * 1e3 / h_cgs
f_max = Kb_cgs * 3e13 / h_cgs
f_num = 1_000
freqs = np.logspace(np.log10(f_min), np.log10(f_max), f_num)
# bands
## ZTF 
ztf_g_band = (4100, 5500)
ztf_r_band = (5600, 7100)
ztf_i_band = (7000, 8400)
## Swift
swift_uvw2_center, swift_uvw2_fwhm = 1928, 657 
swift_uvm2_center, swift_uvm2_fwhm = 2246, 498
swift_uvw1_center, swift_uvw1_fwhm = 2600, 693
swift_u_center, swift_u_fwhm = 3465, 785 
swift_b_center, swift_b_fwhm = 4392, 975 
swift_v_center, swift_v_fwhm = 5468, 769
swift_v_band = (swift_v_center - swift_v_fwhm/2, swift_v_center + swift_v_fwhm/2)
swift_b_band = (swift_b_center - swift_b_fwhm/2, swift_b_center + swift_b_fwhm/2)
swift_u_band = (swift_u_center - swift_u_fwhm/2, swift_u_center + swift_u_fwhm/2)
swift_uvw1_band = (swift_uvw1_center - swift_uvw1_fwhm/2, swift_uvw1_center + swift_uvw1_fwhm/2)
swift_uvm2_band = (swift_uvm2_center - swift_uvm2_fwhm/2, swift_uvm2_center + swift_uvm2_fwhm/2)
swift_uvw2_band = (swift_uvw2_center - swift_uvw2_fwhm/2, swift_uvw2_center + swift_uvw2_fwhm/2)

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet
mpl.rcParams['figure.dpi'] = 300
# plt.rcParams['text.usetex'] = True
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['figure.figsize'] = [8 , 6]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
mpl.rcParams['savefig.transparent'] = True  # Only figure patch
mpl.rcParams['figure.facecolor'] = 'none'    # Figure when displayed
mpl.rcParams['axes.facecolor'] = 'none'      # Axes patch
cmap = mpl.colormaps['tab20']  # nice, but only 8:Dark2
colors_cy = cmap.colors  # type: list
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors_cy)
# colors_cy = plt.get_cmap('tab20').colors[:10] + plt.get_cmap('tab20').colors[:10]
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors_cy)
wanted_colors = ['firebrick', 'coral', 'sandybrown', 'darkviolet', 'orchid', 'yellowgreen', 'forestgreen', 'dodgerblue', 'navy']
reverse_colors = wanted_colors[::-1]
wanted_colors = np.concatenate([wanted_colors, reverse_colors])
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color = wanted_colors)

if __name__ == '__main__':
    # it's the same converting from cgs ans SI ... of course lol
    tsol_SI = np.sqrt(Rsol_SI**3 / (Msol_SI*G_SI )) # Follows from G = 1
    csol_SI = c_SI / (Rsol_SI/tsol_SI)
    # print(csol_cgs/csol_SI)
    # print(tsol_cgs/tsol_SI)
    print(np.sqrt(8*Kb_cgs/(np.pi*m_p_cgs)))
