""" Expand tables from RICH"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from src.Opacity.linextrapolator import opacity_extrap, opacity_linear

opac_path = f'{abspath}/src/Opacity/MG'
T_cool = np.loadtxt(f'{opac_path}/fromRICH/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/fromRICH/rho.txt')
for e_idx in np.arange(1, 11):
    rossland = np.loadtxt(f'{opac_path}/fromRICH/sigma_rossland_{e_idx}.txt') # this is the interpolation for the rossland opacity, which is the one we need for the photosphere. We save it as a txt to avoid doing the interpolation every time, which is very expensive.
    T_cool2, Rho_cool2, rossland2 = opacity_extrap(T_cool, Rho_cool, rossland, which_opacity = 'rossland', scatter = None)
    np.savetxt(f'{opac_path}/sigma_rossland_{e_idx}.txt', rossland2)
np.savetxt(f'{opac_path}/T.txt', T_cool2)
np.savetxt(f'{opac_path}/rho.txt', Rho_cool2)