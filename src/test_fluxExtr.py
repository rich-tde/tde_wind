"""Test extraction of flux components"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Utilities.prelude as prel

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
snap = 151

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
photo = np.loadtxt(f'{abspath}/data/{folder}/photo/{check}_photo{snap}POL.txt')
x, y, z, Lum, Fx, Fy, Fz = photo[0], photo[1], photo[2], photo[14], photo[16], photo[17], photo[18]

r_vec = np.vstack([x,y,z]).T
r_mag = np.linalg.norm(r_vec, axis = 1)
r_hat = r_vec/ r_mag[:, None]
F_vec = np.vstack([Fx,Fy,Fz]).T
F_mag = np.linalg.norm(F_vec, axis = 1)
F_hat = F_vec/F_mag[:, None]
F_r_from_L = Lum / (4*np.pi*(r_mag*prel.Rsol_cgs)**2)

cos_theta = np.zeros_like(F_mag)
for i in range(len(F_mag)):
    cos_theta[i] = np.dot(F_hat[i], r_hat[i])
F_r= F_mag * cos_theta
cut = F_r_from_L!=0

plt.plot(F_r_from_L/F_r, 'o')
