#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:52:21 2024

@author: konstantinos
"""
import numpy as np
# User lines
import sys
sys.path.append('/Users/paolamartire/tde_comparison')


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

# Healpy
NSIDE = 4
    
# Plotting
import matplotlib.pyplot as plt
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [8 , 6]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
AEK = '#F1C410'