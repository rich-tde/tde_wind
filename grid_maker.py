"""
Created in February 2024.
Author: Paola Martire 

Make a 3d grid
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import h5py

import Utilities.prelude as prel

def make_grid(num):
    """ Build a tree from simulation cells. """
    X = np.load('data/CMx.npy')
    Y = np.load('data/CMy.npy')
    Z = np.load('data/CMz.npy')
    VX = np.load('data/Vx.npy')
    VY = np.load('data/Vy.npy')
    VZ = np.load('data/Vz.npy')
    T = np.load('data/T.npy')
    Den = np.load('data/Den.npy')
    P = np.load('data/P.npy')
    if all(T)==0:
        print('all T=0, bro. CHANGE!')
        T = P/Den
    Vol = np.load('data/Vol.npy')

    # Convert
    # Den *= prel.en_den_converter

    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 

    # query points
    x_radii = np.linspace(-1, 1, num)
    y_radii = np.linspace(-1, 1, num)
    z_radii = np.linspace(-1, 1, num)
    if len(T)< num**3:
        print('ALERT: Too many cells!')
        # return 1

    # store values
    gridded_indexes =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_den =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_T =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_P =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_Vx =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_Vy =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_Vz =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_V =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_Rcell =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))

    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            for k in range(len(z_radii)):
                queried_value = [x_radii[i], y_radii[j], z_radii[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                gridded_indexes[i, j, k] = idx
                gridded_den[i, j, k] = Den[idx]
                gridded_T[i, j, k] = T[idx]
                gridded_P[i, j, k] = P[idx]
                gridded_Vx[i, j, k] = VX[idx]
                gridded_Vy[i, j, k] = VY[idx]
                gridded_Vz[i, j, k] = VZ[idx]
                gridded_V[i, j, k] = np.sqrt(VX[idx]**2 + VY[idx]** 2 +VZ[idx]**2)
                gridded_Rcell[i, j, k] = (3*Vol[idx]/(np.pi*4))**(1/3)

                # if np.abs(Z[idx])<0.02:
                #     Xslice.append(X[idx])
                #     Yslice.append(Y[idx])


        # Progress Check
        # progress = 100 * i/len(x_radii)
        # if (progress%10 == 0):
        #     print('Progress: {:1.0%}'.format(i/len(x_radii)))
    print('Tree built!')

    return gridded_indexes, gridded_den, gridded_T, gridded_P, gridded_Vx, gridded_Vy, gridded_Vz, gridded_V, gridded_Rcell, x_radii, y_radii, z_radii

