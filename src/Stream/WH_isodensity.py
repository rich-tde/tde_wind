""" Measures the stream width and heigh for all considered snaps. """
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    from Utilities.selectors_for_snap import select_snap
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    from Utilities.basic_units import radians
    import matplotlib.colors as colors
    compute = False

import numpy as np
from scipy.optimize import brentq
import gc
import os
import csv
import Utilities.prelude as prel
import Utilities.sections as sec
import src.orbits as orb
from Utilities.operators import make_tree, Ryan_sampler
from Utilities.selectors_for_snap import select_prefix, select_snap

#%%
## Parameters
#
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = 'HiResNewAMR'
compton = 'Compton'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']

##
# FUNCTIONS
##

def bound_mass_density(density_threshold, density_data, mass_data, m_thres):
    """
    Function to use with root finding to find the density threshold to respect the wanted mass enclosed.
    
    Parameters
    ----------
    density_threshold : float
        The density threshold to find.
    density_data : array
        The density data to check the threshold against.
    mass_data : array
        The corresponding mass data.
    m_thres : float
        The threshold mass to respect.
        
    Returns
    -------
    float
        The difference between the total mass enclosed in the density threshold and the wanted threshold mass.
    """
    condition = density_data >= density_threshold  
    mass = mass_data[condition]
    total_mass = np.sum(mass)
    return total_mass - m_thres

def find_single_boundaries_isodensity(x_data, y_data, z_data, dim_data, density_data, mass_data, stream, idx, mass_percentage):
    """
    Find the boundaries using isodensity contours for a single theta.
    
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, density_data, mass_data : array
        X, Y, Z coordinates, dimension, density, and mass of simulation points.
    stream : array
        Stream data containing: theta, x, y, z coordinates, threshold for cutting the plane.
    idx : int
        Index of the theta in the stream to consider.
    mass_percentage : float
        Target mass percentage to enclose.
        
    Returns
    -------
    density_threshold : float
        The density threshold that encloses the target mass percentage.
    indeces_enclosed : array
        Indices of points inside the isodensity contour.
    contour_stats : dict
        Dictionary containing statistics about the contour. 
    """
    global Rt, R0, Rstar
    indeces = np.arange(len(x_data))
    theta_arr, x_stream, y_stream, z_stream, thresh_stream = stream[0], stream[1], stream[2], stream[3], stream[4]
    thresh = thresh_stream[idx]
    # Slice the transverse plane and get the quantities of its points
    condition_T, x_Tplane, _ = sec.transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, Rstar, just_plane=True)
    x_plane, y_plane, z_plane, dim_plane, density_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_data, y_data, z_data, dim_data, density_data, mass_data, indeces], condition_T)
    # Apply the same threshold used to find the stream (even if the plane is slightly different)
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, dim_plane, density_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_plane, x_Tplane, y_plane, z_plane, dim_plane, density_plane, mass_plane, indeces_plane], condition)
    
    r_spherical_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
    if np.min(r_spherical_plane) < R0:
        print(f'The threshold to cut the TZ plane in width is too broad: you overcome R0 at angle {theta_arr[idx]} of the stream')
    
    # Calculate target mass
    total_mass = np.sum(mass_plane)
    mass_to_reach = mass_percentage * total_mass
    
    # Find density range for root finding
    min_density = np.min(density_plane)
    max_density = np.max(density_plane)
    # Use a slightly higher minimum to ensure we don't include all points
    # density_range_min = min_density + 0.001 * (max_density - min_density)
    
    try:
        # Find the density threshold that encloses the target mass
        density_threshold = brentq(bound_mass_density, 
                                 min_density, max_density,
                                 args=(density_plane, mass_plane, mass_to_reach))
        
        # Get condition for points inside the contour
        condition_enclosed = density_plane >= density_threshold
        indeces_enclosed, x_T_enclosed, z_enclosed, dim_enclosed, mass_enclosed = \
            sec.make_slices([indeces_plane, x_Tplane, z_plane, dim_plane, mass_plane], condition_enclosed)
        dim_cell_mean = np.mean(dim_enclosed)

        # Calculate bounding box of the enclosed region
        idx_before = np.argmin(x_T_enclosed)
        idx_after = np.argmax(x_T_enclosed)
        xT_min, xT_max = x_T_enclosed[idx_before], x_T_enclosed[idx_after]
        idx_low, idx_up = indeces_enclosed[idx_before], indeces_enclosed[idx_after]
        width = xT_max - xT_min
        # same for height
        idx_before = np.argmin(z_enclosed)
        idx_after = np.argmax(z_enclosed)
        z_min, z_max = z_enclosed[idx_before], z_enclosed[idx_after]
        idx_low_h, idx_up_h = indeces_enclosed[idx_before], indeces_enclosed[idx_after]
        height = z_max - z_min
        
        indeces_boundary = np.array([idx_low, idx_up, idx_low_h, idx_up_h]).astype(int)
        # Calculate effective dimensions using mean cell size in enclosed region
        ncells_w = np.round(width / dim_cell_mean, 0)
        ncells_h = np.round(height / dim_cell_mean, 0)
        
        # Create contour statistics
        contour_stats = {
            'density_threshold': density_threshold,
            'xT_bounds': [xT_min, xT_max],
            'width': width, # you can delete it
            'ncells_w': ncells_w,
            'z_bounds': [z_min, z_max],
            'height': height,
            'ncells_h': ncells_h,
            'indeces_boundary': indeces_boundary, 
            'mass_fraction': np.sum(mass_enclosed) / total_mass if not alice else None
        } 
        
        # Optional: Create visualization if this is the theta for plotting
        if np.logical_and(alice == False, idx == 4 ):
            plot_isodensity_results(x_plane, x_Tplane, y_plane, z_plane, mass_plane, dim_plane,
                                    condition_enclosed, contour_stats, 
                                    theta_arr[idx], x_data, y_data, indeces_enclosed, 
                                    indeces_plane)
        
        return density_threshold, indeces_enclosed, contour_stats
        
    except ValueError as e:
        print(f"Error finding isodensity contour at theta {np.round(theta_arr[idx], 2)} rad: {e}")
        return None, None, None

def plot_isodensity_results(x_plane, x_Tplane, y_plane, z_plane, mass_plane, dim_plane,
                           condition_enclosed, contour_stats, 
                           theta_val, x_data, y_data, indeces_enclosed, 
                           indeces_plane):
    """
    Plot the isodensity contour results similar to the original plotting style.
    """
    # Get global variables (assuming they exist in your original code)
    # apo = globals().get('apo', 1.0)  # Default to 1 if not found
    x_T_low, x_T_up = contour_stats['xT_bounds'][0], contour_stats['xT_bounds'][1]
    z_low, z_up = contour_stats['z_bounds'][0], contour_stats['z_bounds'][1]
    idx_x_low, idx_x_up = contour_stats['indeces_boundary'][0], contour_stats['indeces_boundary'][1]
    x_low, x_up  = x_data[idx_x_low], x_data[idx_x_up]
    y_low, y_up  = y_data[idx_x_low], y_data[idx_x_up]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mass histogram in T direction
    counts_T, bin_edges_T = np.histogram(x_Tplane, bins=50, weights=mass_plane)
    bin_centers_T = 0.5 * (bin_edges_T[:-1] + bin_edges_T[1:])
    ax1.plot(bin_centers_T, counts_T, c='k', label='Mass distribution')
    ax1.axvline(x=x_T_low, color='k', linestyle='--', label='Isodensity boundary')
    ax1.axvline(x=x_T_up, color='k', linestyle='--')
    ax1.legend()
    ax1.set_ylabel(r'Mass [$M_\odot$]')
    ax1.set_xlabel(r'N [$R_\odot$]')
    
    # Plot 2: Scatter plot with isodensity contour
    # Color points by whether they're inside or outside the contour
    ax2.scatter(x_Tplane, z_plane, s=10, edgecolor='k', facecolor='none') 
    img = ax2.scatter(x_Tplane[condition_enclosed], z_plane[condition_enclosed], c=mass_plane[condition_enclosed], cmap = 'rainbow',s = 12, norm =colors.LogNorm(vmin=np.percentile(mass_plane, 80), vmax=np.max(mass_plane)))
    plt.colorbar(img, ax=ax2, label=r'Mass [$M_\odot$]')
    ax2.axvline(x=x_T_low, color='k', linestyle='--', linewidth=2)
    ax2.axvline(x=x_T_up, color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=z_low, color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=z_up, color='k', linestyle='--', linewidth=2)
    ax2.set_xlabel(r'N [$R_\odot$]')
    ax2.set_ylabel(r'Z [$R_\odot$]')
    ax2.set_ylim(z_low-1, z_up+1)
    
    original_ticks = ax1.get_xticks()
    mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
    new_ticks = np.concatenate((original_ticks, mid_ticks))
    for ax in [ax1, ax2]:
        ax.set_xticks(new_ticks)
        labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
        ax.set_xticklabels(labels)
        ax.set_xlim(x_T_low-1, x_T_up+1)

    # Plot 3: Mass histogram in Z direction
    counts_Z, bin_edges_Z = np.histogram(z_plane, bins=50, weights=mass_plane)
    bin_centers_Z = 0.5 * (bin_edges_Z[:-1] + bin_edges_Z[1:])
    ax3.plot(bin_centers_Z, counts_Z, c='k')
    ax3.axvline(x=z_low, color='k', linestyle='--')
    ax3.axvline(x=z_up, color='k', linestyle='--')
    ax3.set_ylabel(r'Mass [$M_\odot$]')
    ax3.set_xlabel(r'Z [$R_\odot$]')
    original_ticks = ax3.get_xticks()
    mid_ticks = (original_ticks[1:] + original_ticks[:-1]) / 2
    new_ticks = np.concatenate((original_ticks, mid_ticks))
    ax3.set_xticks(new_ticks)
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]
    ax3.set_xticklabels(labels)
    ax3.set_xlim(z_low-1, z_up+1)
    
    # Plot 4: Orbital plane view
    mid = np.abs(z_plane) < dim_plane
    if len(indeces_enclosed) > 0:
        # Map back to original coordinates
        x_enclosed , y_enclosed, z_enclosed, dim_enclosed, mass_enclosed = \
            sec.make_slices([x_plane, y_plane, z_plane, dim_plane, mass_plane], condition_enclosed)
        mid_enclosed = np.abs(z_enclosed) < dim_enclosed
         
        vmin = np.percentile(mass_plane, 80)
        vmax = np.max(mass_plane)
        
        # Plot all points in the plane
        img = ax4.scatter(x_data[indeces_plane][mid], y_data[indeces_plane][mid], 
                         s=10, c=mass_plane[mid], cmap='rainbow', 
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        # Highlight enclosed points
        ax4.scatter(x_enclosed[mid_enclosed], y_enclosed[mid_enclosed], 
                   s=20, c=mass_enclosed[mid_enclosed], edgecolor = 'k', cmap='rainbow', norm=colors.LogNorm(vmin=vmin, vmax=vmax), label='Enclosed points')
        
        cbar = plt.colorbar(img, ax=ax4, label=r'Mass [M$_\odot$]')
        ax4.scatter(0, 0, c='k', s=10)
        ax4.legend()
        ax4.set_xlabel(r'X [$R_\odot$]')
        ax4.set_ylabel(r'Y [$R_\odot$]')
        ax4.set_xlim(x_low-1, x_up+1)
        ax4.set_ylim(y_low-1, y_up+1)
        ax4.set_title('Orbital plane')
    
    # Add text box with statistics
    stats_text = f"Width: {contour_stats['width']:.2f}\nHeight: {contour_stats['height']:.2f}\n"
    stats_text += f"Cells W: {contour_stats['ncells_w']:.0f}\nCells H: {contour_stats['ncells_h']:.0f}\n"
    stats_text += f"Mass enclosed : {100*contour_stats['mass_fraction']:.0f}\% of trans. plane"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(rf'$\theta$ = {np.round(theta_val, 2)} rad', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/{folder}/WHCont{massperc}_{check}{snap}.png')
    plt.show()

def follow_the_stream_isodensity(x_data, y_data, z_data, dim_data, density_data, mass_data, stream, mass_percentage = 0.8):
    """
    Find isodensity contour boundaries all along the stream.
    
    Parameters
    ----------
    x_data, y_data, z_data, dim_data, density_data, mass_data : array
        Simulation data arrays.
    stream : array
        Stream data containing: theta, x, y, z coordinates, threshold.
    mass_percentage : float
        Target mass percentage to enclose (default 0.8 for 80%).
        
    Returns
    -------
    theta_wh : array
        Theta values where boundaries were successfully found.
    density_thresholds : array
        Density thresholds for each theta.
    indeces_enclosed : list
        List of indices arrays for enclosed points at each theta.
    contour_stats : list
        List of contour statistics dictionaries for each theta.
    """
    theta_arr = stream[0]
    
    # Initialize output arrays
    theta_wh = []
    density_thresholds = []
    indeces_enclosed = []
    contour_stats = []
    
    for i in range(len(theta_arr)):
        print(i)
        density_threshold, indeces_enclosed_i, contour_stats_i = \
            find_single_boundaries_isodensity(x_data, y_data, z_data, dim_data, density_data, mass_data, stream, i, mass_percentage)
        
        if density_threshold is None:
            print(f'Skipping theta {np.round(theta_arr[i],2)} due to ValueError in boundary finding.', flush=True)
            continue
        else:
            theta_wh.append(theta_arr[i])
            density_thresholds.append(density_threshold)
            indeces_enclosed.append(indeces_enclosed_i)
            contour_stats.append(contour_stats_i)
    
    return theta_wh, np.array(density_thresholds), indeces_enclosed, contour_stats

if __name__ == '__main__':
    theta_lim =  np.pi
    step = np.round((2*theta_lim)/200, 3)
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    idx_forplot = np.argmin(np.abs(theta_arr)) # indices for theta = -pi/2, 0, pi/2: 54 101 149
    massperc = 0.8
    print(f'We are in folder {folder}', flush=True)
    
    path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    
    if compute:
        if alice:
            snaps = select_snap(m, check, mstar, Rstar, beta, n, compton, time = False) 
        else: 
            snaps = [76]

        for i, snap in enumerate(snaps):
            print(f'Snap {snap}', flush = True)

            path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
            if alice:
                path = f'{path}/snap_{snap}'
            else:
                path = f'{path}/{snap}'

            data = make_tree(path, snap, energy = False)
            X, Y, Z,  VX, VY, VZ, Den, Mass, Vol = \
                data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Den, data.Mass, data.Vol
            cutden = Den >1e-19
            X, Y, Z, VX, VY, VZ, Den, Mass, Vol = \
                sec.make_slices([X, Y, Z, VX, VY, VZ, Den, Mass, Vol], cutden)
            dim_cell = Vol**(1/3) 
            del Vol

            try:
                com = np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npz')
                print('Load stream from file', flush=True)
            except FileNotFoundError:
                from src.Stream.com_stream import find_transverse_com
                print('Stream not found, computing it', flush=True)
                thresh_cm, indices_cm = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr)
                x_cm, y_cm, z_cm, vx_cm, vy_cm, vz_cm, den_cm, mass_cm = \
                    X[indices_cm], Y[indices_cm], Z[indices_cm], \
                        VX[indices_cm], VY[indices_cm], VZ[indices_cm], \
                            Den[indices_cm], Mass[indices_cm]
                com = {
                    'theta_arr': theta_arr,
                    'thresh_cm': thresh_cm,
                    'x_cm': x_cm,
                    'y_cm': y_cm,
                    'z_cm': z_cm,
                    'vx_cm': vx_cm,
                    'vy_cm': vy_cm,
                    'vz_cm': vz_cm,
                    'den_cm': den_cm,
                    'mass_cm': mass_cm,
                    'indices_cm': indices_cm
                } 
                if alice:
                    np.savez(f'{abspath}/data/{folder}/WH/stream/stream_{check}_{snap}.npz', **com)

            stream = [com['theta_arr'], com['x_cm'], com['y_cm'], com['z_cm'], com['thresh_cm']]
            if not alice: # just some computation
                stream = stream[:, idx_forplot-4:idx_forplot+5]

            theta_wh, density_thresholds, indeces_enclosed, contour_stats = follow_the_stream_isodensity(X, Y, Z, dim_cell, Den, Mass, stream, mass_percentage = massperc)
            w_params = np.array([[stats['width'] for stats in contour_stats],
                                [stats['ncells_w'] for stats in contour_stats]])
            h_params = np.array([[stats['height'] for stats in contour_stats],
                                [stats['ncells_h'] for stats in contour_stats]])
            indeces_boundary = np.array([stats['indeces_boundary'] for stats in contour_stats])
            indeces_enclosed = np.array(indeces_enclosed, dtype=object)

            del X, Y, Z, Den, Mass, dim_cell
            gc.collect()

            if alice:
                X_bound = X[indeces_boundary] # nx4: Xlow_w, Xup_w, Xlow_h, Xup_h are the rows
                Y_bound = Y[indeces_boundary]
                Z_bound = Z[indeces_boundary]

                width_data = {
                    'theta_wh': theta_wh,
                    'X_low_w': X_bound[:, 0],
                    'X_up_w': X_bound[:, 1],
                    'X_low_h': X_bound[:, 2],
                    'X_up_h': X_bound[:, 3],
                    'Y_low_w': Y_bound[:, 0],
                    'Y_up_w': Y_bound[:, 1],
                    'Y_low_h': Y_bound[:, 2],
                    'Y_up_h': Y_bound[:, 3],
                    'Z_low_w': Z_bound[:, 0],
                    'Z_up_w': Z_bound[:, 1],
                    'Z_low_h': Z_bound[:, 2],
                    'Z_up_h': Z_bound[:, 3],
                    'width': w_params[0],
                    'N_width': w_params[1],
                    'height': h_params[0],
                    'N_height': h_params[1]
                }

                np.savez(f'{abspath}/data/{folder}/WH/wh_{massperc}{check}_{snap}.npz', **width_data)
                np.save(f'{abspath}/data/{folder}/WH/indeces_boundary_{massperc}{check}_{snap}.npy', indeces_boundary)
                np.save(f'{abspath}/data/{folder}/WH/enclosed/indeces_enclosed_{massperc}{check}_{snap}.npy', indeces_enclosed, allow_pickle=True)

    if plot: 
        x_axis = ''
        data = np.loadtxt(f'{abspath}/data/{folder}/projection/Dentime_proj.txt', ndmin=2)
        snaps = data[0].astype(int)
        tfb = data[1] 

        # Plotting results
        for i, snap in enumerate(snaps):
            # if snap == 81:
            #     continue
            data_width = np.load(f'{abspath}/data/{folder}/WH/wh_{massperc}{check}{snap}.npz', allow_pickle=True)
            theta_wh = data_width['theta_wh']
            width = data_width['width']
            N_width = data_width['N_width']
            height = data_width['height']
            N_height = data_width['N_height']
            if x_axis == 'radius':
                theta_stream, x_stream, y_stream, z_stream, _ = \
                    np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npy', allow_pickle=True)
                r_stream = np.sqrt(x_stream**2 + y_stream**2 + z_stream**2)
                x_ax = r_stream/Rt
            else:
                x_ax = theta_wh * radians
            # print('indices for theta = -pi/2, 0, pi/2:', np.argmin(np.abs(theta_wh + np.pi/2)), np.argmin(np.abs(theta_wh)), np.argmin(np.abs(theta_wh - np.pi/2)))
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
            if x_axis == 'radius':
                ax1.scatter(x_ax, width, c = theta_wh*radians, cmap = 'jet')
                img = ax3.scatter(x_ax, N_width, c = theta_wh*radians, cmap = 'jet')
                cbar = plt.colorbar(img, orientation = 'horizontal')
                cbar.set_label(r'$\theta$ [rad]')
                ax2.scatter(x_ax, height, c = theta_wh*radians, cmap = 'jet')
                ax2.plot(x_ax, 8*np.sqrt(x_ax), c = 'k', ls = '--')
                img = ax4.scatter(x_ax, N_height, c = theta_wh*radians, cmap = 'jet')
                cbar = plt.colorbar(img, orientation = 'horizontal')
                cbar.set_label(r'$\theta$ [rad]')
            else:
                ax1.plot(x_ax, width, c = 'darkviolet')
                ax3.plot(x_ax, N_width, c = 'darkviolet')
                ax2.plot(x_ax, height, c = 'darkviolet')
                ax4.plot(x_ax, N_height, c = 'darkviolet')
            ax1.set_ylabel(r'Width [$R_\odot$]')
            ax3.set_ylabel(r'N cells')
            ax2.set_ylabel(r'Height [$R_\odot$]')
            ax1.set_ylim(1, 15)
            ax2.set_ylim(.2, 10)
            ax3.set_ylim(10, 40)    
            ax4.set_ylim(0.9, 35)
            for ax in [ax1, ax2, ax3, ax4]:
                if x_axis == 'radius':
                    ax.set_xlabel(r'$r [r_{\rm t}]$')
                    ax.set_xlim(.5, 1e2)
                    ax.loglog()
                else:
                    ax.set_xlabel(r'$\theta$')
                    ax.set_xlim(-2.5, 2.5)
                    ax.set_yscale('log')
                ax.tick_params(axis='both', which='major', length = 10, width = 1.2)
                ax.tick_params(axis='both', which='minor', length = 8, width = 1)
                ax.grid()
            plt.suptitle(f't = {np.round(tfbs[i],2)} ' + r't$_{\rm fb}$', fontsize= 25)
            plt.tight_layout()
            plt.savefig(f'{abspath}/Figs/{folder}/stream/WH_theta{snap}{x_axis}.png')
            plt.close()
    




# %%
