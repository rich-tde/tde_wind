import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from scipy.spatial import KDTree
import h5py

# import Utilities.prelude as prel

def mask(X, Y, Z, xlim, ylim, zlim):
    """ Mask the data di avoid the cells near the boarders of the simulation box."""
    Xmask = X[np.logical_and(np.logical_and(Y > - ylim, Y < ylim), (X > - xlim, X < xlim))]
    Ymask = Y[np.logical_and(Y > - ylim, Y < ylim)]
    Zmask = Z[np.logical_and(Z > - zlim, Z > zlim)]
    return Xmask, Ymask, Zmask

def loader():
    """ Load data from simulation. """
    X = np.load('data_sim/CMx.npy')
    Y = np.load('data_sim/CMy.npy')
    Z = np.load('data_sim/CMz.npy')
    VX = np.load('data_sim/Vx.npy')
    VY = np.load('data_sim/Vy.npy')
    VZ = np.load('data_sim/Vz.npy')
    T = np.load('data_sim/T.npy')
    Den = np.load('data_sim/Den.npy')
    P = np.load('data_sim/P.npy')
    if all(T)==0:
        print('all T=0, bro. CHANGE!')
        T = P/Den
    
    # Xmask, Ymask, Zmask = mask(X, Y, Z, xlim, ylim, zlim)
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 

    return sim_tree, X, Y, Z, VX, VY, VZ, Den, P, T

