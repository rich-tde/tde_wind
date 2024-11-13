import numpy as np

# User lines
import sys
sys.path.append('/Users/paolamartire/shocks')

# Constants
c_cgs = 2.99792458e10 #[cm/s]
h_cgs = 6.62607015e-27 #[gcm^2/s]
Kb_cgs = 1.380649e-16 #[gcm^2/s^2K]
alpha_cgs = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
sigma_T_cgs = 6.6524e-25 #[cm^2] thomson cross section
G_cgs = 6.6743e-8 # cgs
Rsol_cgs = 6.957e10 # [cm]
Msol_cgs = 1.989e33 # [g]

# Solar and SI units
c_SI = 2.99e8 #m
G_SI = 6.6743e-11 # SI 
Msol_SI = 2e30 #1.98847e30 # kg
Rsol_SI = 7e8 #6.957e8 # m

# Sim units
G = 1

# Converters
tsol_cgs = np.sqrt(Rsol_cgs**3 / (Msol_cgs*G_cgs )) # Follows from G = 1
csol_cgs = c_cgs / (Rsol_cgs/tsol_cgs)
den_converter = Msol_cgs / Rsol_cgs**3
en_den_converter = Msol_cgs / (Rsol_cgs  * tsol_cgs**2 ) # Energy Density converter
en_converter = Msol_cgs * Rsol_cgs**2 / tsol_cgs**2 # Energy converter

# Healpy
import healpy as hp
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)#  int(NSIDE * 96)

    
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = [8 , 6]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.tight_layout()
AEK = '#F1C410'

if __name__ == '__main__':
    # it's the same converting from cgs ans SI ... of course lol
    tsol_SI = np.sqrt(Rsol_SI**3 / (Msol_SI*G_SI )) # Follows from G = 1
    csol_SI = c_SI / (Rsol_SI/tsol_SI)
    print(csol_cgs/csol_SI)
    print(tsol_cgs/tsol_SI)