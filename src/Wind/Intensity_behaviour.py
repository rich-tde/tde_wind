""" Check how intesity behaves. """
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel

#%%
def temp_of_z(z, z0, T0 = 2e3, deltaT = 8e3):
    T = -z0/10 + z0 * np.exp(-z/z0)
    return T

def der_T_z(z, z0, T0 = 2e3, deltaT = 8e3):
    dT_dz = - np.exp(-z/z0)
    return dT_dz
 
def planck(nu, z, z0):
    """ Planck function. """
    T = temp_of_z(z, z0)
    B_nu = (2*prel.h_cgs*nu**3/prel.c_cgs**2) / (np.exp(prel.h_cgs*nu/(prel.Kb_cgs*T)) - 1)
    return B_nu

def der_planck_z(nu, z, z0):
    T = temp_of_z(z, z0)
    dB_dT = planck(nu, z, z0) * prel.h_cgs*nu/(prel.Kb_cgs*T**2) / (np.exp(prel.h_cgs*nu/(prel.Kb_cgs*T)) - 1)
    dB_dz = dB_dT * der_T_z(z, z0)
    print(dB_dz, planck(nu, z, z0))
    return dB_dz

def I_from_B(nu, z, theta, z0):
    dB_dz = der_planck_z(nu, z, z0)
    I = planck(nu, z, z0) - np.cos(theta)/0.34 * dB_dz
    return I

freq_arr = np.logspace(4, 20, 1000) # in Hz
z_arr = np.logspace(10, 14, 1000) # in cm
theta = np.pi/8
z_chosen = 2e6
z0 = z_chosen

I_planck_arr = np.zeros_like(freq_arr)
I_arr = np.zeros_like(freq_arr) 
for i, nu in enumerate(freq_arr):
    I_planck_arr[i] = planck(nu, z_chosen, z0)
    I_arr[i] = I_from_B(nu, z_chosen, theta, z0) 

plt.figure(figsize=(12,10))
plt.loglog(freq_arr, I_arr, label='Intensity')
plt.loglog(freq_arr, I_planck_arr, label='Planck function', linestyle='dashed', c = 'k')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Intensity (erg/s/cm$^2$/Hz/sr)')
plt.ylim(1e-15, 1e9)




