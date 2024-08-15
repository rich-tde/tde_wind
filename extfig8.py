#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:19:32 2023

@author: konstantinos
"""
# Vanilla
import numba
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [8.0, 4.0]

# Chocolate
from src.Calculators.ONE_TREE_CASTER import BONSAI
from src.Calculators.casters import THE_SMALL_CASTER
from src.Extractors.time_extractor import days_since_distruption
from src.Eccentricity.eccentricity import e_calc
from src.Utilities.isalice import isalice
from src.Utilities.parser import parse
alice, plot = isalice()

# Choose Simulation
if alice:
    args = parse()
    sim = args.name
    m = args.mass
    r = args.radius
    Mbh = args.blackhole
    fixes = np.arange(args.first, args.last + 1)
else:
    m = 10
    # do it yourself

save = True
method = 'caster'

# Constants
G = 6.6743e-11  # SI
Msol = 1.98847e30  # kg
Rsol = 6.957e8  # m

Rt = Mbh**(1/3)
t = np.sqrt(Rsol**3 / (Msol*G))  # Follows from G=1
# Need these for the PW potential
c = 3e8 * t/Rsol  # c in simulator units.
rg = 2*Mbh/c**2
t_fall = 40 * (Mbh/1e6)**(0.5)  # days EMR+20 p13
apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee

@numba.njit
def masker(arr, mask):
    len_bound = np.sum(mask)
    new_arr = np.zeros(len_bound)
    k = 0
    for i in range(len(arr)):
        if mask[i]:
            new_arr[k] = arr[i]
            k += 1
    return new_arr

# %%
# MAIN
colarr = []
fixdays = []

for fix in fixes:
    if alice:
        fix = str(fix)
        pre = '/home/s3745597/data1/TDE/'
        # Import
        X = np.load(pre + sim + '/snap_'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load(pre + sim + '/snap_'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load(pre + sim + '/snap_'  + fix + '/CMz_' + fix + '.npy')
        Vx = np.load(pre + sim + '/snap_'  + fix + '/Vx_' + fix + '.npy')
        Vy = np.load(pre + sim + '/snap_'  +fix + '/Vy_' + fix + '.npy')
        Vz = np.load(pre + sim + '/snap_'  +fix + '/Vz_' + fix + '.npy')
        Den = np.load(pre + sim + '/snap_'  + fix + '/Den_' + fix + '.npy')
        Vol = np.load(pre + sim + '/snap_'  + fix + '/Vol_' + fix + '.npy')
        M = np.multiply(Den, Vol)
        del Den, Vol
    else:
        X = np.load(str(m) + '/' + fix + '/CMx_' + fix + '.npy')
        Y = np.load(str(m) + '/' + fix + '/CMy_' + fix + '.npy')
        Z = np.load(str(m) + '/' + fix + '/CMz_' + fix + '.npy')
        Vx = np.load(str(m) + '/' + fix + '/Vx_' + fix + '.npy')
        Vy = np.load(str(m) + '/' + fix + '/Vy_' + fix + '.npy')
        Vz = np.load(str(m) + '/' + fix + '/Vz_' + fix + '.npy')
        M = np.load(str(m) + '/' + fix + '/Mass_' + fix + '.npy')

    # Make Bound Mask
    R = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
    V = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2) + np.power(Vz, 2))
    Orbital = (0.5 * V**2) - Mbh / (R-rg)
    bound_mask = np.where(Orbital < 0, 1, 0)

    # Apply Mask
    X = masker(X, bound_mask)
    Y = masker(Y, bound_mask)
    Z = masker(Z, bound_mask)
    # Redefine only for bound
    R_bound = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
    Vx = masker(Vx, bound_mask)
    Vy = masker(Vy, bound_mask)
    Vz = masker(Vz, bound_mask)
    M = masker(M, bound_mask)

    position = np.array((X, Y, Z)).T  # Transpose for col. vectors
    velocity = np.array((Vx, Vy, Vz)).T
    del X, Y, Z, Vx, Vy, Vz

    # EVOKE eccentricity
    _, ecc = e_calc(position, velocity, Mbh)

    # Cast down to 100 values
    radii = np.logspace(np.log10(0.4*Rt), np.log10(apocenter),
                        num=100)  # simulator units

    if method == 'caster':
        ecc_cast = THE_SMALL_CASTER(radii, R_bound, ecc, weights=M)
    if method == 'tree':
        ecc_cast = BONSAI(radii, R_bound, ecc)

    # mw_ecc_casted = np.nan_to_num(mw_ecc_casted)
    colarr.append(ecc_cast)

    if alice:
        t_by_tfb = np.loadtxt(f'{pre}{sim}/snap_{fix}/tbytfb_{fix}.txt')
        fixdays.append(t_by_tfb)
    else:
        day = np.round(days_since_distruption(
            sim + fix + '/snap_' + fix + '.h5'), 1)
        t_by_tfb = day  # /t_fall
        fixdays.append(t_by_tfb)

    if save:
        if alice:
            np.savetxt(f'{pre}tde_comparison/data/ecc{sim}.txt', colarr)
            np.savetxt(f'{pre}tde_comparison/data/eccdays{sim}.txt', fixdays)
        else:
             with open('data/ecc'+ str(m) + '.txt', 'a') as file:
                for i in range(len(colarr)):
                    file.write('# snap' + ' '.join(map(str, fixes[i])) + '\n')
                    file.write('# Eccentricity \n') 
                    file.write(' '.join(map(str, colarr[i])) + '\n')
                file.close() 
# %% Plotting
if plot:
    img = plt.pcolormesh(radii, fixdays, colarr,
                        cmap='jet', vmin=0, vmax=1)
    # plt.xlim(radii[0]-0.8 , radii[-12])
    cb = plt.colorbar(img)
    cb.set_label('Eccentricity')
    plt.xscale('log')
    plt.ylabel('Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Eccentricity as a function of time and radius', fontsize=17)
    plt.show()
