import numpy as np

# User lines
import sys
sys.path.append('/Users/paolamartire/shocks')


# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
G = 6.6743e-11 # SI
sigma_T = 6.6524e-25 #[cm^2] thomson cross section

# Solar to SI units
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1

# Converters
Rsol_to_cm = 6.957e10 # [cm]
Msol_to_g = 2e33 # 1.989e33 # [g]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter
en_converter = Msol_to_g * Rsol_to_cm**2 / t**2 # Energy converter

# Healpy
NSIDE = 4
    
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
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.top'] = True
plt.tight_layout()
AEK = '#F1C410'