import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from Utilities.sections import transverse_plane, make_slices

def bound_mass(x, den_data, mass_data, m_thres):
    # given a density x, compute the mass enclosed in the region where the density is greater than x to find the threshold m_thres
    condition = den_data > x
    mass = mass_data[condition]
    total_mass = np.sum(mass)
    return total_mass - m_thres

def find_single_boundaries(x_data, y_data, z_data, dim_data, den_data, mass_data, indeces_orbit, idx):
    indeces = np.arange(len(x_data))
    x_orbit, y_orbit, z_orbit, dim_orbit, den_orbit = \
        x_data[indeces_orbit], y_data[indeces_orbit], z_data[indeces_orbit], dim_data[indeces_orbit], den_data[indeces_orbit]
    # Find the transverse plane 
    condition_T, x_Tplane, x0 = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, den_plane, mass_plane, indeces_plane = \
        make_slices([x_data, y_data, z_data, dim_data, den_data, mass_data, indeces], condition_T)
    # Restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
    condition_x = np.abs(x_Tplane) < 5
    condition_z = np.abs(z_plane) < 5
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, dim_plane, den_plane, mass_plane, indeces_plane = \
        make_slices([x_plane, x_Tplane, y_plane, z_plane, dim_plane, den_plane, mass_plane, indeces_plane], condition)
    mass_to_reach = 0.5 * np.sum(mass_plane)

    # Find the density threshold
    den_contour = brentq(bound_mass, 0, den_orbit[idx], args=(den_plane, mass_plane, mass_to_reach))
    condition_den = den_plane > den_contour
    x_T_denth, z_denth, dim_denth, indeces_denth = \
        make_slices([x_Tplane, z_plane, dim_plane, indeces_plane], condition_den)

    idx_before = np.argmin(x_T_denth)
    idx_after = np.argmax(x_T_denth)
    x_T_low, idx_low = x_T_denth[idx_before], indeces_denth[idx_before]
    x_T_up, idx_up = x_T_denth[idx_after], indeces_denth[idx_after]
    width = x_T_up - x_T_low
    width = np.max([width, dim_orbit[idx]]) # to avoid 0 width
    dim_cell_mean = np.mean(dim_denth)
    ncells_w = np.round(width/dim_cell_mean, 0) # round to the nearest integer

    idx_before = np.argmin(z_denth)
    idx_after = np.argmax(z_denth)
    z_low, idx_low_h = z_denth[idx_before], indeces_denth[idx_before]
    z_up, idx_up_h = z_denth[idx_after], indeces_denth[idx_after]
    height = z_up - z_low
    height = np.max([height, dim_orbit[idx]]) # to avoid 0 height
    ncells_h = np.round(height/dim_cell_mean, 0) # round to the nearest integer

    indeces_boundary = np.array([idx_low, idx_up, idx_low_h, idx_up_h]).astype(int)
    x_T_width = np.array([x_T_low, x_T_up])
    w_params = np.array([width, ncells_w])
    h_params = np.array([height, ncells_h])

    return indeces_boundary, x_T_width, w_params, h_params


def follow_the_stream(x_data, y_data, z_data, dim_data, den_data, mass_data, theta_arr, path):
    """ Find width and height all along the stream """
    # Find the center of mass of the stream for each theta
    streamLow = np.load(path)
    theta_arr, indeces_stream = streamLow[0], streamLow[1].astype(int)
    theta_arr = theta_arr[:210]
    indeces_boundary = []
    x_T_width = []
    w_params = []
    h_params = []
    for i in range(len(theta_arr)):
        print(i)
        try:
            indeces_boundary_i, x_T_width_i, w_params_i, h_params_i = \
                find_single_boundaries(x_data, y_data, z_data, dim_data, den_data, mass_data, indeces_stream, i)
            indeces_boundary.append(indeces_boundary_i)
            x_T_width.append(x_T_width_i)
            w_params.append(w_params_i)
            h_params.append(h_params_i)
        except:
            i_final = i
            theta_arr = theta_arr[:i_final]
            break
    indeces_boundary = np.array(indeces_boundary).astype(int)
    x_T_width = np.array(x_T_width)
    w_params = np.transpose(np.array(w_params)) # line 1: width, line 2: ncells
    h_params = np.transpose(np.array(h_params)) # line 1: height, line 2: ncells
    return indeces_stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr


if __name__ == '__main__':
    from Utilities.operators import make_tree, Ryan_sampler
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('/Users/paolamartire/shocks')
    G = 1
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'Low'
    check1 = 'HiRes'
    check2 = 'Res20'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
    snap = '164'
    snap2 = '169'
    path = f'TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    theta_lim = np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    data = make_tree(path, snap, is_tde = True, energy = False)
    data1 = make_tree(f'TDE/{folder}{check1}/{snap}', snap, is_tde = True, energy = False)
    # data2 = make_tree(f'TDE/{folder}{check2}/{snap2}', snap2, is_tde = True, energy = False)
    dim_cell = data.Vol**(1/3)
    dim_cell1 = data1.Vol**(1/3)
    # dim_cell2 = data2.Vol**(1/3)

    midplane = np.abs(data.Z) < dim_cell
    X_midplane, Y_midplane, Z_midplane, dim_midplane, Mass_midplane = \
        make_slices([data.X, data.Y, data.Z, dim_cell, data.Mass], midplane)

    file = f'data/{folder}/stream_{check}{snap}_{step}.npy' 
    file1 = f'data/{folder}/stream_{check1}{snap}_{step}.npy'
    # file2 = f'data/{folder}/stream_{check2}{snap2}_{step}.npy'
    # density = np.load(f'TDE/{folder}{check}/{snap}/smoothed_Den_{snap}.npy')s
    # density1 = np.load(f'TDE/{folder}{check1}/{snap}/smoothed_Den_{snap}.npy')

    #%%
    indeces_stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass, theta_arr, path = file)
    #%%
    indeces_stream1, indeces_boundary1, x_T_width1, w_params1, h_params1, theta_arr1  = follow_the_stream(data1.X, data1.Y, data1.Z, dim_cell1, data1.Den, data1.Mass, theta_arr, path = file1)
    # indeces_stream2, indeces_boundary2, x_T_width2, w_params2, h_params2, theta_arr2  = follow_the_stream(data2.X, data2.Y, data2.Z, dim_cell2, data2.Den, data2.Mass, theta_arr, path = file2)
    #%%
    cm_x, cm_y = data.X[indeces_stream], data.Y[indeces_stream]
    low_x, low_y = data.X[indeces_boundary[:,0]] , data.Y[indeces_boundary[:,0]]
    up_x, up_y = data.X[indeces_boundary[:,1]] , data.Y[indeces_boundary[:,1]]
    cm_x1, cm_y1 = data1.X[indeces_stream1], data1.Y[indeces_stream1]
    low_x1, low_y1 = data1.X[indeces_boundary1[:,0]] , data1.Y[indeces_boundary1[:,0]]
    up_x1, up_y1 = data1.X[indeces_boundary1[:,1]] , data1.Y[indeces_boundary1[:,1]]
    plt.plot(cm_x, cm_y,  c = 'k', label = 'Orbit')
    plt.plot(low_x, low_y, '--', c = 'k',label = 'Lower tube')
    plt.plot(up_x, up_y, '-.', c = 'k', label = 'Upper tube')
    plt.plot(cm_x1, cm_y1,  c = 'r', label = 'Orbit1')
    plt.plot(low_x1, low_y1, '--', c = 'r',label = 'Lower tube1')
    plt.plot(up_x1, up_y1, '-.', c = 'r', label = 'Upper tube1')
    plt.xlim(-100,30)
    plt.ylim(-40,40)
    plt.legend()
    plt.show()

    plt.plot(theta_arr, w_params[0], c= 'k', label = 'Low')
    plt.plot(theta_arr1, w_params1[0], c='r', label = 'Middle')
    # plt.plot(theta_arr2, w_params2[0], c='b', label = 'High')
    # plt.plot(h_params[0], label = 'Height')
    plt.legend()
    plt.ylim(0,20)
    plt.show()

    
# %%
