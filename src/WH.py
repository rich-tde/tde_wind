import sys
sys.path.append('/Users/paolamartire/shocks')
abspath = '/Users/paolamartire/shocks'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import numba
from Utilities.basic_units import radians
from datetime import datetime

from Utilities.operators import make_tree, to_cylindric, Ryan_sampler
import Utilities.sections as sec
import src.orbits as orb
from Utilities.time_extractor import days_since_distruption
import Utilities.prelude as prel

#
## Parameters
#

#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
params = [Mbh, Rstar, mstar, beta]
check = '' # 'Low' or 'HiRes' or 'Res20'
compton = 'Compton'
snap = '164'

#
## Constants
#


Mbh = 10**m
Rs = 2*prel.G*Mbh / prel.csol_cgs**2
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rp
apo = orb.apocentre(Rstar, mstar, Mbh, beta)

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'{abspath}TDE/{folder}{check}/{snap}'
saving_path = f'{abspath}Figs/{folder}/{check}'
print(f'We are in: {path}, \nWe save in: {saving_path}')

##
# FUNCTIONS
##

@numba.njit
def get_threshold(t_plane, z_plane, r_plane, mass_plane, dim_plane, R0):
    """ Find the T-Z threshold to cut the transverse plane (as a square) in width and height.
    Parameters
    ----------
    t_plane, z_planr, r_plane, mass_plane, dim_plane : array
        T, Z, radial (spherical) coordinates, mass and dimension of points in the TZ plane.
    R0 : float
        Smoothing radius.
    Returns
    -------
    C : float
        The (upper) threshold for t and z.
    """
    # First guess of C and find the mass enclosed in the initial boundaries
    C = 2 #np.min([not_toovercome, 2])
    condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
    mass = mass_plane[condition]
    total_mass = np.sum(mass)
    while True:
        # update 
        step = 2*np.mean(dim_plane[condition])#2*np.max(dim_plane[condition])
        C += step
        condition = np.logical_and(np.abs(t_plane) <= C, np.abs(z_plane) <= C)
        # Check that you add new points
        if len(mass_plane[condition]) == len(mass):
            C += 2
            # print(C)
        else:
            tocheck = r_plane[condition]-R0
            if tocheck.any()<0:
                C -= step
                print('overcome R0', C)
                break
            mass = mass_plane[condition]
            new_mass = np.sum(mass) 
            # old mass is > 95% of the new mass
            if np.logical_and(total_mass > 0.95 * new_mass, total_mass != new_mass): # to be sure that you've done a step
                break
            total_mass = new_mass
   
    return C

def find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, R0):
    """ Find the maxima density points in the radial plane for all the thetas in theta_arr"""
    x_max = np.zeros(len(theta_arr))
    y_max = np.zeros(len(theta_arr))
    z_max = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        # Exclude points inside the smoothing lenght and find radial plane
        condition_distance = np.sqrt(x_data**2 + y_data**2 + z_data**2) > R0 
        condition_Rplane = sec.radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, den_plane = sec.make_slices([x_data, y_data, z_data, den_data], condition_Rplane)
        # Find and save the maximum density point
        idx_max = np.argmax(den_plane) 
        x_max[i] = x_plane[idx_max]
        y_max[i] = y_plane[idx_max]
        z_max[i] = z_plane[idx_max]
        
    return x_max, y_max, z_max    

def find_transverse_com(x_data, y_data, z_data, dim_data, den_data, mass_data, theta_arr, params, test = False):
    """ Find the centres of mass in the transverse plane"""
    Mbh, Rstar, mstar, beta = params[0], params[1], params[2], params[3]
    Rt = Rstar * (Mbh/mstar)**(1/3)
    R0 = 0.6 * Rt
    apo = orb.apocentre(Rstar, mstar, Mbh, beta)
    # indeces = np.arange(len(x_data))
    # Cut a bit the data for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 100, np.abs(y_data) < 10*apo)
    x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut = \
        sec.make_slices([x_data, y_data, z_data, dim_data, den_data, mass_data], cutting)
    # Find the radial maximum points to have a first guess of the stream (necessary for the threshold and the tg)
    x_stream_rad, y_stream_rad, z_stream_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, den_cut, theta_arr, R0)
    print('radial done')

    # First iteration: find the center of mass of each transverse plane of the maxima-density stream
    # indeces_cmTR = np.zeros(len(theta_arr))
    x_cmTR = np.zeros(len(theta_arr))
    y_cmTR = np.zeros(len(theta_arr))
    z_cmTR = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        print(idx)
        # Find the transverse plane
        condition_T, x_T, _ = sec.transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_stream_rad, y_stream_rad, z_stream_rad, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, den_plane, dim_plane = \
            sec.make_slices([x_cut, y_cut, z_cut, mass_cut, den_cut, dim_cut], condition_T)
        if idx == np.argmin(np.abs(theta_arr)): # plot section at pericenter
            print(theta_arr[idx])
            from matplotlib import colors
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,8))
            img = ax1.scatter(x_T, z_plane, c = den_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-6))
            cbar = plt.colorbar(img)
            cbar.set_label(r'Density $[M_\odot/R_\odot^3]$')
            ax1.set_ylabel(r'Z [$R_\odot$]')
            img = ax2.scatter(x_T, z_plane, c = mass_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-8))
            cbar = plt.colorbar(img)
            cbar.set_label(r'Cell mass $[R_\odot]$')
            for ax in [ax1, ax2]:
                ax.scatter(0,0, edgecolor= 'k', marker = 'o', facecolors='none', s=80)
                ax.set_xlim(-50, 30)
                ax.set_ylim(-10, 10)
                ax.set_xlabel(r'T [$R_\odot$]')
            plt.suptitle('(0,0) is the maximum density point of the radial plane', fontsize = 14)
            plt.tight_layout()
        # Cut the TZ plane to not keep points too far away.
        r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_stream_rad[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, y_plane, z_plane, mass_plane = \
            sec.make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
        # Find the center of mass
        x_cmTR[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cmTR[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cmTR[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)

    print('Iteration radial-transverse done')
    # Second iteration: find the center of mass of each transverse plane corresponding to COM stream
    # indeces_cm = np.zeros(len(theta_arr))
    x_cm = np.zeros(len(theta_arr))
    y_cm = np.zeros(len(theta_arr))
    z_cm = np.zeros(len(theta_arr))
    thresh_cm = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        print(idx)
        # Find the transverse plane
        condition_T, x_T, _ = sec.transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_cmTR, y_cmTR, z_cmTR, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, den_plane, dim_plane = \
            sec.make_slices([x_cut, y_cut, z_cut, mass_cut, den_cut, dim_cut], condition_T)
        if idx == np.argmin(np.abs(theta_arr)): # plot section at pericenter
            from matplotlib import colors
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,8))
            img = ax1.scatter(x_T, z_plane, c = den_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-13, vmax = 1e-6))
            cbar = plt.colorbar(img)
            cbar.set_label(r'Density $[M_\odot/R_\odot^3]$')
            ax1.set_ylabel(r'Z [$R_\odot$]')
            img = ax2.scatter(x_T, z_plane, c = mass_plane, s = 10, cmap = 'rainbow', norm = colors.LogNorm(vmin = 1e-12, vmax = 1e-8))
            cbar = plt.colorbar(img)
            cbar.set_label(r'Cell mass $[R_\odot]$')
            for ax in [ax1, ax2]:
                ax.scatter(0,0, edgecolor= 'k', marker = 'o', facecolors='none', s=80)
                ax.set_xlim(-50, 30)
                ax.set_ylim(-10, 10)
                ax.set_xlabel(r'T [$R_\odot$]')
            plt.suptitle('(0,0) is the center of mass of the TZ plane of the max density point', fontsize = 14)
            plt.tight_layout()
        # Restrict the points to not keep points too far away.
        r_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
        thresh = get_threshold(x_T, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_cmTR[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, y_plane, z_plane, mass_plane = \
            sec.make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
        # Find and save the center of mass
        x_cm[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cm[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cm[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
        thresh_cm[idx] = thresh
    #     x_com = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
    #     y_com = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
    #     z_com = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
    #     # Search in the tree the closest point to the center of mass
    #     points = np.array([x_plane, y_plane, z_plane]).T
    #     tree = KDTree(points)
    #     _, idx_cm = tree.query([x_com, y_com, z_com])
    #     indeces_cm[idx]= indeces_plane[idx_cm]
    # indeces_cm = indeces_cm.astype(int)
    if test == True:
        return x_cm, y_cm, z_cm, thresh_cm, x_cmTR, y_cmTR, z_cmTR, x_stream_rad, y_stream_rad, z_stream_rad
    else:
        return x_cm, y_cm, z_cm, thresh_cm

def bound_mass(x, check_data, mass_data, m_thres):
    """ Function to use with root finding to find the coordinate threshold to respect the wanted mass enclosed in"""
    condition = np.abs(check_data) < x # it's either x_T or Z
    mass = mass_data[condition]
    total_mass = np.sum(mass)
    return total_mass - m_thres

def find_single_boundaries(x_data, y_data, z_data, dim_data, mass_data, stream, idx, params):
    """ Find the width and the height of the stream for a single theta """
    Mbh, Rstar, mstar, beta = params[0], params[1], params[2], params[3]
    Rt = Rstar * (Mbh/mstar)**(1/3)
    R0 = 0.6 * (Rt / beta) 
    indeces = np.arange(len(x_data))
    theta_arr, x_stream, y_stream, z_stream, thresh_stream = stream[0], stream[1], stream[2], stream[3], stream[4]
    # Find the transverse plane 
    condition_T, x_Tplane, _ = sec.transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_data, y_data, z_data, dim_data, mass_data, indeces], condition_T)
    # Restrict to not keep points too far away.
    r_spherical_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
    # thresh = get_threshold(x_Tplane, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_cm/Rp)**(1/2)
    thresh = thresh_stream[idx]
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, mass_plane, indeces_plane = \
        sec.make_slices([x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, mass_plane, indeces_plane], condition)
    if np.min(r_spherical_plane)< R0:
        print(f'The threshold to cut the TZ plane in width is too broad: you overcome R0 at angle #{theta_arr[idx]} of the stream')
    mass_to_reach = 0.25 * np.sum(mass_plane) #25% for each quarter
    # Find the threshold for x
    x_Tplanepositive, mass_planepositive, indices_planepositive, dim_cell_xpositive = \
        x_Tplane[x_Tplane>=0], mass_plane[x_Tplane>=0], indeces_plane[x_Tplane>=0], dim_plane[x_Tplane>=0]
    x_Tplanenegative, mass_planenegative, indices_planenegative, dim_cell_xnegative = \
        x_Tplane[x_Tplane<0], mass_plane[x_Tplane<0], indeces_plane[x_Tplane<0], dim_plane[x_Tplane<0]

    contourTpositive = brentq(bound_mass, 0, thresh, args=(x_Tplanepositive, mass_planepositive, mass_to_reach))
    condition_contourTpositive = np.abs(x_Tplanepositive) < contourTpositive
    x_T_contourTpositive, indeces_contourTpositive = sec.make_slices([x_Tplanepositive, indices_planepositive], condition_contourTpositive)

    contourTnegative = brentq(bound_mass, 0, thresh, args=(x_Tplanenegative, mass_planenegative, mass_to_reach))
    condition_contourTnegative = np.abs(x_Tplanenegative) < contourTnegative
    x_T_contourTnegative, indeces_contourTnegative = sec.make_slices([x_Tplanenegative, indices_planenegative], condition_contourTnegative)

    # contourT = brentq(bound_mass, 0, thresh, args=(x_Tplane, mass_plane, mass_to_reach))
    # condition_contourT = np.abs(x_Tplane) < contourT
    # x_T_contourT, indeces_contourT = make_slices([x_Tplane, indeces_plane], condition_contourT)

    idx_before = np.argmin(x_T_contourTnegative)
    idx_after = np.argmax(x_T_contourTpositive)
    x_T_low, idx_low = x_T_contourTnegative[idx_before], indeces_contourTnegative[idx_before]
    x_T_up, idx_up = x_T_contourTpositive[idx_after], indeces_contourTpositive[idx_after]
    width = x_T_up - x_T_low
    # width = np.max([width, dim_stream[idx]]) # to avoid 0 width

    # Find the threshold for z 
    z_planepositive, mass_planepositive, indices_planepositive, dim_cell_zpositive = \
        z_plane[z_plane>=0], mass_plane[z_plane>=0], indeces_plane[z_plane>=0], dim_plane[z_plane>=0]
    z_planenegative, mass_planenegative, indices_planenegative, dim_cell_znegative = \
        z_plane[z_plane<0], mass_plane[z_plane<0], indeces_plane[z_plane<0], dim_plane[z_plane<0]

    contourZpositive = brentq(bound_mass, 0, thresh, args=(z_planepositive, mass_planepositive, mass_to_reach))
    condition_contourZpositive = np.abs(z_planepositive) < contourZpositive
    z_contourZpositive, indeces_contourZpositive = sec.make_slices([z_planepositive, indices_planepositive], condition_contourZpositive)

    contourZnegative = brentq(bound_mass, 0, thresh, args=(z_planenegative, mass_planenegative, mass_to_reach))
    condition_contourZnegative = np.abs(z_planenegative) < contourZnegative
    z_contourZnegative, indeces_contourZnegative = sec.make_slices([z_planenegative, indices_planenegative], condition_contourZnegative)

    # contourZ = brentq(bound_mass, 0, thresh, args=(z_plane, mass_plane, mass_to_reach))
    # condition_contourZ = np.abs(z_plane) < contourZ
    # z_contourZ, indeces_contourZ = sec.make_slices([z_plane, indeces_plane], condition_contourZ)
    
    idx_before = np.argmin(z_contourZnegative)
    idx_after = np.argmax(z_contourZpositive)
    z_low, idx_low_h = z_contourZnegative[idx_before], indeces_contourZnegative[idx_before]
    z_up, idx_up_h = z_contourZpositive[idx_after], indeces_contourZpositive[idx_after]
    height = z_up - z_low
    # height = np.max([height, dim_stream[idx]]) # to avoid 0 height

    # Compute the number of cells in width and height using the cells in the rectangle
    # rectangle = condition_contourT & condition_contourZ
    dim_cell_Wpositive = np.mean(dim_cell_xpositive[condition_contourTpositive])
    dim_cell_Wnegative = np.mean(dim_cell_xnegative[condition_contourTnegative])
    dim_cell_Hpositive = np.mean(dim_cell_zpositive[condition_contourZpositive])
    dim_cell_Hnegative = np.mean(dim_cell_znegative[condition_contourZnegative])
    dim_cell_mean = np.mean([dim_cell_Wpositive, dim_cell_Wnegative, dim_cell_Hpositive, dim_cell_Hnegative])
    # dim_cell_mean = np.mean(dim_plane[rectangle])
    ncells_w = np.round(width/dim_cell_mean, 0) # round to the nearest integer
    ncells_h = np.round(height/dim_cell_mean, 0) # round to the nearest integer

    indeces_boundary = np.array([idx_low, idx_up, idx_low_h, idx_up_h]).astype(int)
    x_T_width = np.array([x_T_low, x_T_up])
    w_params = np.array([width, ncells_w])
    h_params = np.array([height, ncells_h])

    return indeces_boundary, x_T_width, w_params, h_params, thresh

def follow_the_stream(x_data, y_data, z_data, dim_data, mass_data, path, params):
    """ Find width and height all along the stream """
    # Find the stream (load it)
    stream = np.load(path)
    theta_arr = stream[0]
    # Find the boundaries for each theta
    indeces_boundary = []
    x_T_width = []
    w_params = []
    h_params = []
    for i in range(len(theta_arr)):
        indeces_boundary_i, x_T_width_i, w_params_i, h_params_i, _ = \
            find_single_boundaries(x_data, y_data, z_data, dim_data, mass_data, stream, i, params)
        indeces_boundary.append(indeces_boundary_i)
        x_T_width.append(x_T_width_i)
        w_params.append(w_params_i)
        h_params.append(h_params_i)
    indeces_boundary = np.array(indeces_boundary).astype(int)
    x_T_width = np.array(x_T_width)
    w_params = np.transpose(np.array(w_params)) # line 1: width, line 2: ncells
    h_params = np.transpose(np.array(h_params)) # line 1: height, line 2: ncells
    return stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr

#
## MAIN
#

save = False
theta_lim =  np.pi
step = 0.02
theta_init = np.arange(-theta_lim, theta_lim, step)
theta_arr = Ryan_sampler(theta_init)
tfb = days_since_distruption(f'{abspath}TDE/{folder}{check}/{snap}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')

#%% Load data
data = make_tree(path, snap, energy = False)
X, Y, Z, Den, Mass = \
    data.X, data.Y, data.Z, data.Den, data.Mass
R = np.sqrt(X**2 + Y**2 + Z**2)
THETA, RADIUS_cyl = to_cylindric(X, Y)
dim_cell = data.Vol**(1/3) # according to Elad

# Cross section at midplane
midplane = np.abs(data.Z) < dim_cell
X_midplane, Y_midplane, Z_midplane, dim_midplane, Den_midplane, Mass_midplane = \
    sec.make_slices([X, Y, Z, dim_cell, Den, Mass], midplane)

#%%
try:
    theta_arr, x_stream, y_stream, z_stream, thresh_cm = \
        np.load(f'{abspath}/src/stream/stream_{check}{snap}.npy')
except:
    x_stream, y_stream, z_stream, thresh_cm = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr, params)
    np.save(f'{abspath}/src/stream/stream_{check}{snap}.npy', [theta_arr, x_stream, y_stream, z_stream, thresh_cm])

file = f'{abspath}data/{folder}/stream_{check}{snap}.npy'
stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = orb.follow_the_stream(X, Y, Z, dim_cell, Mass, path = file, params = params)

if save:
    try:
        file = open(f'{abspath}data/{folder}/WH/width_time{np.round(tfb,2)}.txt', 'r')
        # Perform operations on the file
        file.close()
    except FileNotFoundError:
        with open(f'{abspath}data/{folder}/WH/width_time{np.round(tfb,2)}.txt','a') as fstart:
            # if file exist, save theta and date of execution
            fstart.write(f'# theta, done on {datetime.now()} \n')
            fstart.write((' '.join(map(str, theta_arr)) + '\n'))

    with open(f'{abspath}data/{folder}/WH/width_time{np.round(tfb,2)}.txt','a') as file:
        file.write(f'# {check}, done on {datetime.now()}, snap {snap} \n# Width \n')
        file.write((' '.join(map(str, w_params[0])) + '\n'))
        file.write(f'# Ncells \n')
        file.write((' '.join(map(str, w_params[1])) + '\n'))
        file.write(f'################################ \n')

    # same for height
    try:
        file = open(f'{abspath}data/{folder}/WH/height_time{np.round(tfb,2)}.txt', 'r')
        # Perform operations on the file
        file.close()
    except FileNotFoundError:
        with open(f'{abspath}data/{folder}/WH/height_time{np.round(tfb,2)}.txt','a') as fstart:
            # if file exist
            fstart.write(f'# theta, done on {datetime.now()} \n')
            fstart.write((' '.join(map(str, theta_arr)) + '\n'))

    with open(f'{abspath}data/{folder}/WH/height_time{np.round(tfb,2)}.txt','a') as file:
        file.write(f'# {check}, done on {datetime.now()}, snap {snap} \n# Height \n')
        file.write((' '.join(map(str, h_params[0])) + '\n'))
        file.write(f'# Ncells \n')
        file.write((' '.join(map(str, h_params[1])) + '\n'))
        file.write(f'################################ \n')


#%% 
# Plot width over r
plt.plot(theta_arr * radians, w_params[0], c = 'k')
img = plt.scatter(theta_arr * radians, w_params[0], c = w_params[1], cmap = 'viridis')
cbar = plt.colorbar(img)
cbar.set_label(r'Ncells', fontsize = 16)
plt.xlabel(r'$\theta$', fontsize = 14)
plt.ylabel(r'Width [$R_\odot$]', fontsize = 14)
plt.xlim(-3/4*np.pi, 3/4*np.pi)
plt.ylim(-5,20)
plt.grid()
plt.suptitle(r't/t$_{fb}$ = ' + str(np.round(tfb,3)) + f', check: {check}', fontsize = 16)
plt.tight_layout()
plt.show()

    