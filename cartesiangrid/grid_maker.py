"""
Created in February 2024.
Author: Paola Martire 

Make a 3d grid.
In the main, you have the code for to plot the cross sections.
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
    Vol = np.load('data_sim/Vol.npy')

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

if __name__ == '__main__':
    num = 45
    idx_slice = int(num/2)

    _, gridded_den, gridded_T, gridded_P, gridded_Vx, gridded_Vy, gridded_Vz, gridded_V, _, x_radii, y_radii, z_radii = make_grid(num)

    flat_den = gridded_den[:,:,idx_slice]
    flat_P = gridded_P[:,:,idx_slice]
    flat_T = gridded_T[:,:,idx_slice]
    flat_Vx = gridded_Vx[:,:,idx_slice]
    flat_Vy = gridded_Vy[:,:,idx_slice]
    flat_V = gridded_V[:,:,idx_slice]

    # pass to log
    P_plot = np.log10(flat_P)
    P_plot = np.nan_to_num(P_plot, neginf = 0)
    den_plot = np.log10(flat_den)
    den_plot = np.nan_to_num(den_plot, neginf = 0)
    T_plot = np.log10(flat_T)
    T_plot = np.nan_to_num(T_plot, neginf = 0)
    V_plot = np.log10(flat_V)
    V_plot = np.nan_to_num(flat_V, neginf = 0)

    # import coordinates shock zone
    # shockzone = np.loadtxt(f'shockzone_num{num}.txt')

    # Plot density
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('X', fontsize = 14)
    ax.set_ylabel('Y', fontsize = 14)
    img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'jet', vmin = 0, vmax = 0.4)
    # plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
    cb = plt.colorbar(img)
    cb.set_label(r'$log_{10}$ Density', fontsize = 14)
    plt.title(f'Density slice, z={np.round(z_radii[idx_slice],3)} ' + r' $N_{cell}$'+ f'= {num}')
    plt.savefig(f'denslice_num{num}.png')

    # Plot pressure
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_xlabel('X', fontsize = 14)
    ax1.set_ylabel('Y', fontsize = 14)
    img1 = ax1.pcolormesh(x_radii, y_radii, P_plot.T, cmap = 'jet')#, vmin = 0, vmax = 0.4)
    # plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
    cb1 = plt.colorbar(img1)
    cb1.set_label(r'$log_{10}$ P', fontsize = 14)
    plt.title(f'Pressure slice, z={np.round(z_radii[idx_slice],3)} ' + r' $N_{cell}$'+ f'= {num}')
    plt.savefig(f'Pslice_num{num}.png')

    # Plot temperature
    fig2, ax2 = plt.subplots(1,1)
    ax2.set_xlabel('X', fontsize = 14)
    ax2.set_ylabel('Y', fontsize = 14)
    img2 = ax2.pcolormesh(x_radii, y_radii, T_plot.T, cmap = 'jet')
    # plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
    cb2 = plt.colorbar(img2)
    cb2.set_label(r'$log_{10}$ T', fontsize = 14)
    plt.title(f'Temperature slice, z={np.round(z_radii[idx_slice],3)} ' + r' $N_{cell}$'+ f'= {num}')
    plt.savefig(f'Tslice_num{num}.png')

    fig, ax4 = plt.subplots(1,1)
    ax4.set_xlabel('X', fontsize = 14)
    ax4.set_ylabel('Y', fontsize = 14)
    img = ax4.pcolormesh(x_radii, y_radii, V_plot.T, cmap = 'jet')
    # plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
    cb = plt.colorbar(img)
    cb.set_label(r'$log_{10} V$', fontsize = 14)
    plt.savefig(f'Figs/Vslice_num{num}_nok.png')

plt.show()
