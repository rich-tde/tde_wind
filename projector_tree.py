"""
Project quantities.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')


import numpy as np
import matplotlib.pyplot as plt
import numba

import Utilities.prelude as prel
from grid_maker import make_grid


@numba.njit
def projector(gridded_den, x_radii, y_radii, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) 
                # mass_zsum += gridded_mass[i,j,k]
                flat_den[i,j] += gridded_den[i,j,k] * dz #* gridded_mass[i,j,k]
            #flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
    return flat_den
 
if __name__ == '__main__':
    plot = True
    kind = 'slice'
    num = 45

    _, gridded_den, gridded_T, gridded_P, gridded_Vx, gridded_Vy, gridded_Vz, gridded_V, _, x_radii, y_radii, z_radii = make_grid(num)

    if kind == 'projection':
        flat_den = projector(gridded_den, x_radii, y_radii, z_radii)
    
    if kind =='slice':
        idx_slice = int(len(z_radii)/2)

        flat_den = gridded_den[:,:,idx_slice]
        flat_P = gridded_P[:,:,idx_slice]
        flat_T = gridded_T[:,:,idx_slice]
        flat_Vx = gridded_Vx[:,:,idx_slice]
        flat_Vy = gridded_Vy[:,:,idx_slice]
        flat_V = gridded_V[:,:,idx_slice]

        # to plot
        P_plot = np.log10(flat_P)
        P_plot = np.nan_to_num(P_plot, neginf = 0)
        T_plot = np.log10(flat_T)
        T_plot = np.nan_to_num(T_plot, neginf = 0)

        # import coordinates shock zone
        shockzone = np.loadtxt(f'shockzone_num{num}.txt')

    if plot:
        fig, ax = plt.subplots(1,1)

        den_plot = np.log10(flat_den)
        den_plot = np.nan_to_num(den_plot, neginf = 0)

        # Plot density
        ax.set_xlabel('X', fontsize = 14)
        ax.set_ylabel('Y', fontsize = 14)
        img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'jet', vmin = 0, vmax = 0.4)
        cb = plt.colorbar(img)
        cb.set_label(r'$log_{10}$ Density', fontsize = 14)

        if kind == 'projection':
            plt.title('Density projection')
            plt.savefig('Figs/denproj.png')
        if kind == 'slice':
            plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
            plt.title(f'Density slice, z={np.round(z_radii[idx_slice],3)} ' + r' $N_{cell}$'+ f'= {num}')
            plt.savefig(f'Figs/denslice_num{num}.png')

            # Plot pressure
            fig1, ax1 = plt.subplots(1,1)
            ax1.set_xlabel('X', fontsize = 14)
            ax1.set_ylabel('Y', fontsize = 14)
            img1 = ax1.pcolormesh(x_radii, y_radii, P_plot.T, cmap = 'jet')#, vmin = 0, vmax = 0.4)
            plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
            cb1 = plt.colorbar(img1)
            cb1.set_label(r'$log_{10}$ P', fontsize = 14)
            plt.title(f'Pressure slice, z={np.round(z_radii[idx_slice],3)} ' + r' $N_{cell}$'+ f'= {num}')
            plt.savefig(f'Figs/Pslice_num{num}.png')

            # Plot temperature
            fig2, ax2 = plt.subplots(1,1)
            ax2.set_xlabel('X', fontsize = 14)
            ax2.set_ylabel('Y', fontsize = 14)
            img2 = ax2.pcolormesh(x_radii, y_radii, T_plot.T, cmap = 'jet')
            plt.plot(shockzone[0], shockzone[1], 'ks', c = 'k', markerfacecolor='none', ms=6, markeredgecolor='k')
            cb2 = plt.colorbar(img2)
            cb2.set_label(r'$log_{10}$ T', fontsize = 14)
            plt.title(f'Temperature slice, z={np.round(z_radii[idx_slice],3)} ' + r' $N_{cell}$'+ f'= {num}')
            plt.savefig(f'Figs/Tslice_num{num}.png')


        # if kind == 'slice':
        #     fig, ax2 = plt.subplots(1,1)
        #     Vx_plot = np.log10(flat_Vx)
        #     Vx_plot = np.nan_to_num(flat_Vx, neginf = 0)
        #     ax2.set_xlabel('X', fontsize = 14)
        #     ax2.set_ylabel('Y', fontsize = 14)
        #     img = ax2.pcolormesh(x_radii, y_radii, Vx_plot.T, cmap = 'jet')
        #     cb = plt.colorbar(img)
        #     cb.set_label(r'$log_{10} V_x$', fontsize = 14)
        #     plt.savefig(f'Figs/Vzslice_z{z_slice}.png')

        #     fig, ax3 = plt.subplots(1,1)
        #     Vy_plot = np.log10(flat_Vy)
        #     Vy_plot = np.nan_to_num(flat_Vy, neginf = 0)
        #     ax3.set_xlabel('X', fontsize = 14)
        #     ax3.set_ylabel('Y', fontsize = 14)
        #     img = ax3.pcolormesh(x_radii, y_radii, Vy_plot.T, cmap = 'jet')
        #     cb = plt.colorbar(img)
        #     cb.set_label(r'$log_{10} V_y$', fontsize = 14)
        #     plt.savefig(f'Figs/Vyslice_z{z_slice}.png')

        #     fig, ax4 = plt.subplots(1,1)
        #     V_plot = np.log10(flat_V)
        #     V_plot = np.nan_to_num(flat_V, neginf = 0)
        #     ax4.set_xlabel('X', fontsize = 14)
        #     ax4.set_ylabel('Y', fontsize = 14)
        #     img = ax4.pcolormesh(x_radii, y_radii, V_plot.T, cmap = 'jet')
        #     cb = plt.colorbar(img)
        #     cb.set_label(r'$log_{10} V$', fontsize = 14)
        #     plt.savefig(f'Figs/Vslice_z{z_slice}.png')

        plt.show()
