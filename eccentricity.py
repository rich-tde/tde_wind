#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:53:06 2023

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [8.0, 4.0]
import numba

@numba.njit 
def e_calc(position, velocity, mu):
    
    # To be returned
    ecc = np.zeros((len(position),len(position[0])))
    ecc_mag = np.zeros((len(position)))
    # Grav. parameter μ = GM_bh but G=1

    for i in range(len(position)):
        # Calc. the magnitude of the vectors
        r_mag = np.linalg.norm(position[i])
        v_mag = np.linalg.norm(velocity[i])
        
        # Calc. the multipliers
        r_multi = v_mag**2/mu - 1/r_mag
        v_multi = - np.dot(position[i], velocity[i]) / mu
        
        # Actually multiply
        r_for_ecc = np.multiply(r_multi, position[i])
        v_for_ecc = np.multiply(v_multi, velocity[i])
        
        # Add'em!
        ecc[i] = np.add(r_for_ecc, v_for_ecc)
        
        # Grab the mag. as well
        ecc_mag[i] = np.linalg.norm(ecc[i])
        
    return ecc, ecc_mag

@numba.njit 
def ta_calc(ecc, position, velocity):
    ta = np.zeros(len(ecc))
    for i in range(len(ta)):
        # dot it
        dot = np.dot(ecc[i], position[i])
        
        # mag it
        e_mag = np.linalg.norm(ecc[i])
        r_mag = np.linalg.norm(position[i])
        
        # calc it
        temp = dot / (e_mag * r_mag)
        ta[i] = np.arccos(temp)
        
        # fix it
        dot2 = np.dot(position[i], velocity[i])
        if dot2 < 0:
            ta[i] = 2*np.pi - ta[i]
    return ta

if __name__ == '__main__':
    plot = False
    # Load data
    fix = '820'
    X = np.load(fix + '/CMx_' + fix + '.npy')
    Y = np.load(fix + '/CMy_' + fix + '.npy')
    Z = np.load(fix + '/CMz_' + fix + '.npy')
    Vx = np.load(fix + '/Vx_' + fix + '.npy')
    Vy = np.load(fix + '/Vy_' + fix + '.npy')
    Vz = np.load(fix + '/Vz_' + fix + '.npy')
    R = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y,X)
    
    # Make into Vectors
    position = np.array((X,Y,Z)).T # Transpose for col. vectors
    velocity = np.array((Vx,Vy,Vz)).T 
    
    # EVOKE
    e_vec, e = e_calc(position, velocity)
    true_anomaly = ta_calc(e_vec, position, velocity)
    
    #%%
    if plot:
        THETA = np.arctan(Y/X)
        plt.plot(THETA[::1_000],
                's', markersize=1, color = 'teal', label = 'polar')
        plt.plot(true_anomaly[::1_000],
                'o' ,markersize=1, color='orchid', label = 'from ecc')
        plt.legend()
        plt.grid()