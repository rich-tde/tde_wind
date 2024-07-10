""" Test the window size for the density smoothing in the orbit calculation.
Test the difference between radial and transverse maximum"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
from Utilities.sections import make_slices, radial_plane, transverse_plane
from Utilities.operators import make_tree, sort_list

## PARAM
G = 1
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'Low' # '' or 'HiRes' or 'Res20'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
snap = '199'
time = 0.7
is_tde = True
path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = Rt/beta

density_max = False
com = True

if density_max:
    def find_radial_maximum(x_data, y_data, z_data, dim_mid, den_mid, theta_arr, Rt, window_size=3, med=True):
        """ Find the maxima density points in a plane (midplane)"""
        x_cm = np.zeros(len(theta_arr))
        y_cm = np.zeros(len(theta_arr))
        z_cm = np.zeros(len(theta_arr))
        for i in range(len(theta_arr)):
            condition_Rplane = radial_plane(x_data, y_data, dim_mid, theta_arr[i])
            condition_distance = np.sqrt(x_data**2 + y_data**2) > Rt # to avoid the center
            condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
            x_plane = x_data[condition_Rplane]
            y_plane = y_data[condition_Rplane]
            z_plane = z_data[condition_Rplane]
            # Order for radial distance to smooth the density
            r_plane = list(np.sqrt(x_plane**2 + y_plane**2))
            den_plane_sorted = sort_list(den_mid[condition_Rplane], r_plane)
            x_plane_sorted = sort_list(x_plane, r_plane)
            y_plane_sorted = sort_list(y_plane, r_plane)
            z_plane_sorted = sort_list(z_plane, r_plane)
            den_median = median_array(den_plane_sorted, window_size)
            if med:
                idx_cm = np.argmax(den_median) 
            else:
                idx_cm = np.argmax(den_plane_sorted) 
            x_cm[i] = x_plane_sorted[idx_cm]
            y_cm[i] = y_plane_sorted[idx_cm]
            z_cm[i] = z_plane_sorted[idx_cm]
            
        return x_cm, y_cm, z_cm

    def find_transverse_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt, window_size = 0):
        """ Find the maxima density points in a plane (transverse).
        The only difference from the code in src is (the name and) that here you can choose the window size for the density smoothing."""
        x_orbit_rad, y_orbit_rad, _ = find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt, window_size)
        x_cm = np.zeros(len(theta_arr))
        y_cm = np.zeros(len(theta_arr))
        z_cm = np.zeros(len(theta_arr))
        for idx in range(len(theta_arr)):
            condition_T, x_T, _ = transverse_plane(x_data, y_data, dim_data, x_orbit_rad, y_orbit_rad, idx, coord = True)
            # condition to not go too far away. Important for theta = 0
            x_plane, y_plane, z_plane, den_plane = make_slices([x_data, y_data, z_data, den_data], condition_T)
            condition_x = np.abs(x_T) < 20
            x_plane, y_plane, z_plane, den_plane = make_slices([x_plane, y_plane, z_plane, den_plane], condition_x)
            idx_cm = np.argmax(den_plane)
            x_cm[idx], y_cm[idx], z_cm[idx] = x_plane[idx_cm], y_plane[idx_cm], z_plane[idx_cm]
        return x_cm, y_cm, z_cm

    def find_iterate_transverse_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt):
        """ Find the maxima density points in a plane (transverse).
        The only difference from the code in src is (the name and) that here you output the different orbits of all the steps."""
        x_orbit_rad, y_orbit_rad, z_orbit_rad = find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt)
        x_cmTR = np.zeros(len(theta_arr))
        y_cmTR = np.zeros(len(theta_arr))
        z_cmTR = np.zeros(len(theta_arr))
        for idx in range(len(theta_arr)):
            condition_T, x_T, _ = transverse_plane(x_data, y_data, dim_data, x_orbit_rad, y_orbit_rad, idx, coord = True)
            # condition to not go too far away. Important for theta = 0
            x_plane, y_plane, z_plane, den_plane = make_slices([x_data, y_data, z_data, den_data], condition_T)
            condition_x = np.abs(x_T) < 20
            x_plane, y_plane, z_plane, den_plane = make_slices([x_plane, y_plane, z_plane, den_plane], condition_x)
            idx_cm = np.argmax(den_plane)
            x_cmTR[idx], y_cmTR[idx], z_cmTR[idx] = x_plane[idx_cm], y_plane[idx_cm], z_plane[idx_cm]
        x_cm = np.zeros(len(theta_arr))
        y_cm = np.zeros(len(theta_arr))
        z_cm = np.zeros(len(theta_arr))
        dim_cm = np.zeros(len(theta_arr))  
        print('Iterating')
        for idx in range(len(theta_arr)):
            condition_T, x_T, _ = transverse_plane(x_data, y_data, dim_data, x_cmTR, y_cmTR, idx, coord = True)
            # condition to not go too far away. Important for theta = 0
            x_plane, y_plane, z_plane, dim_plane, den_plane = make_slices([x_data, y_data, z_data, dim_data, den_data], condition_T)
            condition_x = np.abs(x_T) < 20
            x_plane, y_plane, z_plane, dim_plane, den_plane = make_slices([x_plane, y_plane, z_plane, dim_plane, den_plane], condition_x)
            idx_cm = np.argmax(den_plane)
            x_cm[idx], y_cm[idx], z_cm[idx], dim_cm[idx] = x_plane[idx_cm], y_plane[idx_cm], z_plane[idx_cm], dim_plane[idx_cm]
        return [x_orbit_rad, y_orbit_rad, z_orbit_rad], [x_cmTR, y_cmTR, z_cmTR],[x_cm, y_cm, z_cm, dim_cm]

    #%% Data load
    data = make_tree(path, snap, is_tde, energy = False)
    dim_cell = data.Vol**(1/3) 
    midplane = np.abs(data.Z) < dim_cell
    X_midplane, Y_midplane, Z_midplane, dim_midplane, Mass_midplane, Den_midplane, Temp_midplane, = \
        make_slices([data.X, data.Y, data.Z, dim_cell, data.Mass, data.Den, data.Temp], midplane)


    #%% Plot
    theta_lim = np.pi
    step = 0.1
    wsize = 3
    wsize1 = 5
    wsize2 = 7
    wsize3 = 9
    theta_params = [-theta_lim, theta_lim, step]
    theta_arr = np.arange(*theta_params)

    """
    RADIAL MAX with different size
    """
    # x_orbit_0, y_orbit_0, z_orbit_0 = find_radial_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, window_size=0, med=True)
    ### To check that if window_size = 0, the orbit is the same as the one without smoothing. TRUE!
    # x_orbit_test, y_orbit_test, z_orbit_test = find_radial_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, med=False)
    # plt.plot(x_orbit_test, y_orbit_test, '--', c= 'red', label = 'Radial no smoothing')
    ###
    # x_orbit_1, y_orbit_1, z_orbit_1 = find_radial_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, window_size=wsize1, med=True)
    # x_orbit_3, y_orbit_3, z_orbit_3 = find_radial_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, window_size=wsize3, med=True)

    # plt.plot(x_orbit_0, y_orbit_0, c= 'b', label = '0 window')
    # plt.plot(x_orbit_1, y_orbit_1, c= 'orange', label = f'{wsize1} window')
    # plt.plot(x_orbit_3, y_orbit_3, c= 'green', label = f'{wsize3} window')
    # plt.legend()
    # plt.show()


    """RADIAL OR TRANSVERSE MAX"""
    # x_orbit_rad, y_orbit_rad, z_orbit_rad = find_radial_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, 0)
    # x_orbit0, y_orbit0, z_orbit0 = find_transverse_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, 0)
    # x_o7rbit, y_orbit, z_orbit = find_transverse_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, wsize)
    # x_orbit1, y_orbit1, z_orbit1 = find_transverse_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, wsize1)
    # x_orbit2, y_orbit2, z_orbit2 = find_transverse_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, wsize2)
    # x_orbit3, y_orbit3, z_orbit3 = find_transverse_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt, wsize3)

    # plt.plot(x_orbit3, y_orbit3, c= 'orange', label = f'Ncell smooothing: {wsize3}')
    # plt.plot(x_orbit2, y_orbit2, c= 'green', label = f'Ncell smooothing: {wsize2}')
    # plt.plot(x_orbit1, y_orbit1, '--', c= 'b', label = f'Ncell smooothing: {wsize1}')
    # plt.plot(x_orbit, y_orbit, '--', c= 'b',  label = f'Ncell smooothing: {wsize}')
    # plt.plot(x_orbit_rad, y_orbit_rad, c= 'black', label = 'Radial')
    # plt.plot(x_orbit0, y_orbit0, '--', c= 'red', label = 'No smooothing')
    # plt.xlim(-300, 30)
    # plt.ylim(-60,60)
    # plt.legend()
    # plt.grid()
    # plt.savefig('Figs/orbit_radORtran.png')
    # plt.show()

    ########## ITERATE
    rad, trans, trans_iter = find_iterate_transverse_maximum(data.X, data.Y, data.Z, dim_cell, data.Den, theta_arr, Rt)
    x_rad, y_rad, z_rad = rad[0], rad[1], rad[2]
    x_trans, y_trans, z_trans = trans[0], trans[1], trans[2]
    x_trans_it, y_trans_it, z_trans_it, dim_trans_it = trans_iter[0], trans_iter[1], trans_iter[2], trans_iter[3]

    plt.plot(x_rad, y_rad, c= 'black', label = 'Radial')
    plt.plot(x_trans, y_trans, '-.', c= 'b',  label = f'Transverse max')
    plt.plot(x_trans_it, y_trans_it, '--', c= 'red', label = 'Transverse max + new Tplane')
    plt.xlim(-300, 30)
    plt.ylim(-60,60)
    plt.legend()
    plt.grid()
    # plt.savefig('Figs/Test/orbit_iter.png')
    plt.show()

    plt.figure()
    plt.plot(z_trans_it, label = 'Z')
    plt.plot(dim_trans_it, label = 'dim cell')
    plt.legend()
    plt.grid()
    plt.show()

######### TEST THRESHOLD
if com:
    sim = f'{check}{snap}'
    width = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/last/maximaDen/width_time{time}_thr0.3.txt')[3]
    data = np.load(f'/Users/paolamartire/shocks/data/{folder}/last/maximaDen/stream_{sim}_0.02.npy')
    theta_arr, indeces_stream = data[0], data[1].astype(int)
    data = make_tree(path, snap, energy = False)
    R = np.sqrt(data.X**2 + data.Y**2)
    R_stream = R[indeces_stream]
    img = plt.scatter(R_stream, width, s=1, c = theta_arr, label = 'Width')
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\theta$', fontsize = 15)
    plt.plot(R_stream, 3*(R_stream/Rp)**(1/2), c = 'indigo', label = r'$\propto (R_{stream}/R_p)^{1/2}$', alpha = 0.5)
    plt.plot(R_stream, 3*(R_stream/Rp)**(1/3), c = 'red', label = r'$\propto (R_{stream}/R_p)^{1/3}$', alpha = 0.5)
    plt.plot(R_stream, 3*(R_stream/Rp)**(1/4), c = 'seagreen', label = r'$\propto (R_{stream}/R_p)^{1/4}$', alpha = 0.5)
    plt.xlabel(r'R$_{stream}$', fontsize = 15)
    plt.ylabel(r'Width $[R_\odot]$', fontsize = 15)
    plt.loglog()
    plt.legend()
    plt.title(f'Sim: {sim}', fontsize = 18)
    plt.savefig(f'Figs/Test/streamR{sim}.png', dpi = 300)
    plt.show()

    

