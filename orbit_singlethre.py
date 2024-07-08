import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.spatial import KDTree
from Utilities.sections import transverse_plane, make_slices, radial_plane

def find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt):
    """ Find the maxima density points in the radial plane for all the theta in theta_arr"""
    x_max = np.zeros(len(theta_arr))
    y_max = np.zeros(len(theta_arr))
    z_max = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        # Exclude points near center and find radial plane
        condition_distance = np.sqrt(x_data**2 + y_data**2) > Rt 
        condition_Rplane = radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, den_plane = make_slices([x_data, y_data, z_data, den_data], condition_Rplane)
        # Find and save the maximum density point
        idx_max = np.argmax(den_plane) 
        x_max[i] = x_plane[idx_max]
        y_max[i] = y_plane[idx_max]
        z_max[i] = z_plane[idx_max]
        
    return x_max, y_max, z_max    

def find_transverse_com(x_data, y_data, z_data, dim_data, mass_data, theta_arr, Rt, Rp):
    """ Find the centres of mass in a plane (transverse)"""
    indeces = np.arange(len(x_data))
    # Cut a bit the data for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 50, np.abs(y_data) < 200)
    x_cut, y_cut, z_cut, dim_cut, mass_cut, indeces_cut = \
        make_slices([x_data, y_data, z_data, dim_data, mass_data, indeces], cutting)
    # Find the radial maximum points to have a first guess of the stream (necessary for the threshold and the tg)
    x_orbit_rad, y_orbit_rad, z_orbit_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, theta_arr, Rt)
    indeces_cmTR = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, dim_cut, x_orbit_rad, y_orbit_rad, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, indeces_cut], condition_T)
        r_cm_rad= np.sqrt(x_orbit_rad[idx]**2+y_orbit_rad[idx]**2)
        thresh = 3 * Rstar * (r_cm_rad/Rp)**(1/3)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane], condition)
        x_com = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_com = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_com = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
        # search in the tree the closest point to the center of mass
        points = np.array([x_plane, y_plane, z_plane]).T
        tree = KDTree(points)
        _, idx_cm = tree.query([x_com, y_com, z_com])
        indeces_cmTR[idx]= indeces_plane[idx_cm]

    indeces_cm = np.zeros(len(theta_arr))
    x_cmTR = x_data[indeces_cmTR.astype(int)]
    y_cmTR = y_data[indeces_cmTR.astype(int)]
    for idx in range(len(theta_arr)):
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, dim_cut, x_cmTR, y_cmTR, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, indeces_cut], condition_T)
        r_cm_rad= np.sqrt(x_orbit_rad[idx]**2+y_orbit_rad[idx]**2)
        thresh = 3 * Rstar * (r_cm_rad/Rp)**(1/3)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane], condition)
        x_com = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_com = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_com = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
        # search in the tree the closest point to the center of mass
        points = np.array([x_plane, y_plane, z_plane]).T
        tree = KDTree(points)
        _, idx_cm = tree.query([x_com, y_com, z_com])
        indeces_cmTR[idx]= indeces_plane[idx_cm]
        indeces_cm[idx]= indeces_plane[idx_cm]
    indeces_cm = indeces_cm.astype(int)
    return indeces_cm

def bound_mass(x, check_data, mass_data, m_thres):
    # given a density x, compute the mass enclosed in the region where the density is greater than x to find the threshold m_thres
    # condition = check_data > x #if you use density
    condition = np.abs(check_data) < x #if you use x_T or Z
    mass = mass_data[condition]
    total_mass = np.sum(mass)
    return total_mass - m_thres

def find_single_boundaries(x_data, y_data, z_data, dim_data, mass_data, indeces_orbit, idx, params):
    Mbh, Rstar, mstar, beta = params[0], params[1], params[2], params[3]
    Rt = Rstar * (Mbh/mstar)**(1/3)
    Rp =  Rt / beta
    indeces = np.arange(len(x_data))
    x_orbit, y_orbit, z_orbit, dim_orbit = \
        x_data[indeces_orbit], y_data[indeces_orbit], z_data[indeces_orbit], dim_data[indeces_orbit]
    # Find the transverse plane 
    condition_T, x_Tplane, _ = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        make_slices([x_data, y_data, z_data, dim_data, mass_data, indeces], condition_T)
    # Restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
    r_cm = np.sqrt(x_orbit[idx]**2+y_orbit[idx]**2)
    thresh = 3 * Rstar * (r_cm/Rp)**(1/3)
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        make_slices([x_plane, x_Tplane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane], condition)
    x_com = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
    y_com = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
    z_com = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
    mass_to_reach = 0.5 * np.sum(mass_plane)

    # Find the threshold for x
    # contour = brentq(bound_mass, 0, den_orbit[idx], args=(den_plane, mass_plane, mass_to_reach))
    # condition_contour = den_plane > contour
    contour = brentq(bound_mass, 0, thresh, args=(x_Tplane, mass_plane, mass_to_reach))
    condition_contour = np.abs(x_Tplane) < contour
    x_T_denth, z_denth, dim_denth, indeces_denth = \
        make_slices([x_Tplane, z_plane, dim_plane, indeces_plane], condition_contour)

    idx_before = np.argmin(x_T_denth)
    idx_after = np.argmax(x_T_denth)
    x_T_low, idx_low = x_T_denth[idx_before], indeces_denth[idx_before]
    x_T_up, idx_up = x_T_denth[idx_after], indeces_denth[idx_after]
    width = x_T_up - x_T_low
    width = np.max([width, dim_orbit[idx]]) # to avoid 0 width
    dim_cell_mean = np.mean(dim_denth)
    ncells_w = np.round(width/dim_cell_mean, 0) # round to the nearest integer

    # Find the threshold for z
    contour = brentq(bound_mass, 0, thresh, args=(z_plane, mass_plane, mass_to_reach))
    condition_contour = np.abs(z_plane) < contour
    x_T_denth, z_denth, dim_denth, indeces_denth = \
        make_slices([x_Tplane, z_plane, dim_plane, indeces_plane], condition_contour)
    
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

    return indeces_boundary, x_T_width, w_params, h_params, thresh


def follow_the_stream(x_data, y_data, z_data, dim_data, mass_data, theta_arr, path, params):
    """ Find width and height all along the stream """
    # Find the center of mass of the stream for each theta
    stream = np.load(path) # you load the stream of that or of another resolution
    theta_arr, indeces_stream = stream[0], stream[1].astype(int)
    indeces_boundary = []
    x_T_width = []
    w_params = []
    h_params = []
    for i in range(len(theta_arr)):
        try: 
            indeces_boundary_i, x_T_width_i, w_params_i, h_params_i, _ = \
                find_single_boundaries(x_data, y_data, z_data, dim_data, mass_data, indeces_stream, i, params)
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
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
    snap = '164'
    path = f'TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    Rp = Rt / beta
    params = [Mbh, Rstar, mstar, beta]
    theta_lim = np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    data = make_tree(path, snap, is_tde = True, energy = False)
    dim_cell = data.Vol**(1/3)

    make_stream = False
    compare = True
    TZslice = False

    if make_stream:
        midplane = np.abs(data.Z) < dim_cell
        X_midplane, Y_midplane, Den_midplane, Mass_midplane = make_slices([data.X, data.Y, data.Den, data.Mass], midplane)

        if compare:
            den_stream = np.load(f'data/{folder}/last/stream_{check}{snap}_{step}.npy')
            den_indeces_stream = den_stream[1].astype(int)
            den_x_stream, den_y_stream = data.X[den_indeces_stream], data.Y[den_indeces_stream]
            print('old done')

        indeces_stream = find_transverse_com(data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass, theta_arr, Rt, Rp)
        # stream = np.load(f'data/{folder}/last/stream_{check}{snap}_{step}.npy')
        # indeces_stream = stream[1].astype(int)
        x_stream, y_stream = data.X[indeces_stream], data.Y[indeces_stream]
        np.save(f'data/{folder}/stream_{check}{snap}_{step}', [theta_arr, indeces_stream])

        plt.figure(figsize = (12,6))
        img = plt.scatter(X_midplane, Y_midplane, c = Mass_midplane, s = 1, cmap = 'viridis', vmin = 5e-10, vmax = 8e-10, alpha = 0.2)
        cbar = plt.colorbar(img)
        cbar.set_label(r'Mass', fontsize = 16)
        plt.plot(x_stream[30:230], y_stream[30:230], c = 'k', label = 'COM stream')
        plt.xlim(-300,20)
        plt.ylim(-60,60)
        plt.grid()
        if compare:
            plt.plot(den_x_stream, den_y_stream, c = 'r', label = 'Density stream')
            plt.legend()
            plt.savefig(f'Figs/Test/NewOldStream{snap}_mass.png')
        plt.show()  


    if TZslice:
        indeces = [30, 80, 120, 160, 200]
        stream = np.load(f'data/{folder}/stream_Low{snap}_{step}.npy')
        indeces_orbit = stream[1].astype(int)
        x_orbit, y_orbit, z_orbit, den_orbit = data.X[indeces_orbit], data.Y[indeces_orbit], data.Z[indeces_orbit], data.Den[indeces_orbit]
        
        plt.plot(x_orbit, y_orbit, c = 'k')
        plt.scatter([x_orbit[i] for i in indeces], [y_orbit[i] for i in indeces], c = 'r')
        plt.xlim(-300,30)
        plt.ylim(-80,80)
        plt.grid()
        plt.savefig(f'Figs/{folder}/{check}/stream{snap}_selectedpoint.png')
        plt.show()  

        for idx in indeces:
            condition_tra, x_onplane, x0 = transverse_plane(data.X, data.Y, dim_cell, x_orbit, y_orbit, idx, coord= True)
            X_tra, Y_tra, Z_tra, Den_tra = \
                make_slices([data.X, data.Y, data.Z, data.Den], condition_tra)
            indeces_boundary, x_T_width, w_params, h_params, thresh = find_single_boundaries(data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass, indeces_orbit, idx)

            img1 = plt.scatter(x_onplane, Z_tra, c = Den_tra,  cmap = 'viridis', s = 27, vmin = 0, vmax = 2e-8)#den_orbit[idx])
            cbar1 = plt.colorbar(img1)
            cbar1.set_label(r'Density', fontsize = 16)
            # plt.tricontour(x_onplane, Z_tra, Den_tra, levels=[den_orbit/3], linewidths=2, colors='red')
            plt.scatter(0, z_orbit[idx], marker = 'x', s = 37, c = 'k', alpha = 1)

            plt.axvline(x_T_width[0], c = 'r')
            plt.axvline(x_T_width[1], c = 'r') # T coordinates for width
            plt.axvline(-thresh, c = 'k', alpha = 0.8)
            plt.axvline(thresh, c = 'k', alpha = 0.8)
            # plt.text(thresh, -2.5*thresh, f'cutoff', fontsize = 15, rotation = 'vertical', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))#,ha='center')
            plt.axvline(-2*thresh, c = 'k', alpha = 0.8)
            plt.axvline(2*thresh, c = 'k', alpha = 0.8)
            # plt.text(2*thresh, -2.5*thresh, f'2 cutoff', fontsize = 15, rotation = 'vertical', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))# ,ha='center')
            plt.xlim(-2*thresh, 2*thresh)
            # plt.text(2.8*thresh, -2.5*thresh, f'3 cutoff', fontsize = 15, rotation = 'vertical', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))#, ha='center')

            plt.axhline(data.Z[indeces_boundary[2]], c = 'r')
            plt.axhline(data.Z[indeces_boundary[3]], c = 'r')
            plt.axhline(-thresh, c = 'k', alpha = 0.8)
            plt.axhline(thresh, c = 'k', alpha = 0.8)
            # plt.text(-2.5*thresh, thresh, f'cutoff', fontsize = 15, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))#,ha='center')
            plt.ylim(-2*thresh, 2*thresh)

            plt.ylabel(r'Z [$R_\odot$]', fontsize = 18) 
            plt.suptitle(f'{check}, cells W: {int(w_params[1])}, cells H: {int(h_params[1])}, theta: {np.round(theta_arr[idx], 2)}', fontsize = 16)
            plt.tight_layout()
            plt.savefig(f'Figs/{folder}/{check}/TZslice{snap}_{idx}.png')
            plt.show()

    if compare:
        from Utilities.time_extractor import days_since_distruption
        file = f'data/{folder}/stream_{check}{snap}_{step}.npy' 
        indeces_stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, data.Mass, theta_arr, path = file, params = params)
        cm_x, cm_y = data.X[indeces_stream], data.Y[indeces_stream]
        low_x, low_y = data.X[indeces_boundary[:,0]] , data.Y[indeces_boundary[:,0]]
        up_x, up_y = data.X[indeces_boundary[:,1]] , data.Y[indeces_boundary[:,1]]
        print('Low done')

        data1 = make_tree(f'TDE/{folder}{check1}/{snap}', snap, is_tde = True, energy = False)
        dim_cell1 = data1.Vol**(1/3)
        file1 = f'data/{folder}/stream_{check1}{snap}_{step}.npy'
        indeces_stream1, indeces_boundary1, x_T_width1, w_params1, h_params1, theta_arr1  = follow_the_stream(data1.X, data1.Y, data1.Z, dim_cell1, data1.Mass, theta_arr, path = file1, params = params)
        cm_x1, cm_y1 = data1.X[indeces_stream1], data1.Y[indeces_stream1]
        low_x1, low_y1 = data1.X[indeces_boundary1[:,0]] , data1.Y[indeces_boundary1[:,0]]
        up_x1, up_y1 = data1.X[indeces_boundary1[:,1]] , data1.Y[indeces_boundary1[:,1]]
        print('HiRes done')

        check2 = 'Res20'
        snap2 = '169'
        data2 = make_tree(f'TDE/{folder}{check2}/{snap2}', snap2, is_tde = True, energy = False)
        dim_cell2 = data2.Vol**(1/3)
        file2 = f'data/{folder}/stream_{check2}{snap2}_{step}.npy'
        indeces_stream2, indeces_boundary2, x_T_width2, w_params2, h_params2, theta_arr2  = follow_the_stream(data2.X, data2.Y, data2.Z, dim_cell2, data2.Mass, theta_arr, path = file2, params = params)
        print('Res20 done')

        tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')
        with open(f'data/{folder}/width_time{np.round(tfb,1)}.txt','a') as file:
                # if file exist
                file.write(f'# theta \n')
                file.write((' '.join(map(str, theta_arr)) + '\n'))
                file.write(f'# {check}, snap {snap} width \n')
                file.write((' '.join(map(str, w_params[0])) + '\n'))
                file.write(f'# {check}, snap {snap} Ncells \n')
                file.write((' '.join(map(str, w_params[1])) + '\n'))
                file.write(f'################################ \n')
                file.write(f'# {check1}, snap {snap} width \n')
                file.write((' '.join(map(str, w_params1[0])) + '\n'))
                file.write(f'# {check1}, snap {snap} Ncells \n')
                file.write((' '.join(map(str, w_params1[1])) + '\n'))
                file.write(f'################################ \n')
                file.write(f'# {check2}, snap {snap2} width \n')
                file.write((' '.join(map(str, w_params2[0])) + '\n'))
                file.write(f'# {check2}, snap {snap2} Ncells \n')
                file.write((' '.join(map(str, w_params2[1])) + '\n'))
                file.write(f'################################ \n')
                file.close()
        
        with open(f'data/{folder}/height_time{np.round(tfb,1)}.txt','a') as file:
                # if file exist
                file.write(f'# theta \n')
                file.write((' '.join(map(str, theta_arr)) + '\n'))
                file.write(f'# {check}, snap {snap} height \n')
                file.write((' '.join(map(str, h_params[0])) + '\n'))
                file.write(f'# {check}, snap {snap} Ncells \n')
                file.write((' '.join(map(str, h_params[1])) + '\n'))
                file.write(f'################################ \n')
                file.write(f'# {check1}, snap {snap} height \n')
                file.write((' '.join(map(str, h_params1[0])) + '\n'))
                file.write(f'# {check1}, snap {snap} Ncells \n')
                file.write((' '.join(map(str, h_params1[1])) + '\n'))
                file.write(f'################################ \n')
                file.write(f'# {check2}, snap {snap2} height \n')
                file.write((' '.join(map(str, h_params2[0])) + '\n'))
                file.write(f'# {check2}, snap {snap2} Ncells \n')
                file.write((' '.join(map(str, h_params2[1])) + '\n'))
                file.write(f'################################ \n')
                file.close()

        #%%
        plt.plot(cm_x, cm_y,  c = 'k', label = 'Orbit')
        plt.plot(low_x, low_y, '--', c = 'k',label = 'Lower tube')
        plt.plot(up_x, up_y, '-.', c = 'k', label = 'Upper tube')
        plt.plot(cm_x1, cm_y1,  c = 'r', label = 'Orbit1')
        plt.plot(low_x1, low_y1, '--', c = 'r',label = 'Lower tube1')
        plt.plot(up_x1, up_y1, '-.', c = 'r', label = 'Upper tube1')
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.xlim(-100,30)
        plt.ylim(-100,100)
        plt.legend()
        plt.show()

        plt.plot(theta_arr, w_params[0], c= 'k', label = 'Low')
        plt.plot(theta_arr1, w_params1[0], c='r', label = 'Middle')
        plt.plot(theta_arr2, w_params2[0], c='b', label = 'High')
        plt.ylabel(r'Width [$R_\odot$]', fontsize = 18)
        plt.xlabel(r'$\theta$', fontsize = 18)
        plt.legend()
        plt.ylim(0,10)
        plt.grid()
        plt.show()

        