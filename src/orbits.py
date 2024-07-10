""" 
Find different kind of orbits (and their derivatives) for TDEs. 
Identify the stream using the centre of mass.
Measure the width and the height of the stream.
"""
import sys
sys.path.insert(0,'/Users/paolamartire/shocks/')
import numpy as np
import Utilities.prelude
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import trapezoid
from scipy.spatial import KDTree
from Utilities.sections import transverse_plane, make_slices, radial_plane

def make_cfr(R, x0=0, y0=0):
    x = np.linspace(-R, R, 100)
    y = np.linspace(-R, R, 100)
    xcfr, ycfr = np.meshgrid(x,y)
    cfr = (xcfr-x0)**2 + (ycfr-y0)**2 - R**2
    return xcfr, ycfr, cfr

def keplerian_orbit(theta, a, Rp, ecc=1):
    # we expect theta as from the function to_cylindric, i.e. clockwise. 
    # You have to mirror it to get the angle for the usual polar coordinates.
    theta = -theta
    if ecc == 1:
        p = 2 * Rp
    else:
        p = a * (1 - ecc**2) 
    radius = p / (1 + ecc * np.cos(theta))
    return radius

# def parameters_orbit(p, a, Mbh, c=1, G=1):
#     Rs = 2 * G * Mbh / c**2
#     En = G * Mbh * (p**2 * (a-Rs) - a**2 * (p-Rs)) / ((a**2-p**2) * (p-Rs) * (a-Rs))
#     L = np.sqrt(2 * a**2 * (En + G*Mbh/(a-Rs)))
#     return En, L

# def solvr(x, theta):
#     _, L = parameters_orbit(Rp, apo)
#     u,y = x
#     res =  np.array([y, (-u + G * Mbh / ((1 - Rs*u) * L)**2)])
#     return res

# def Witta_orbit(theta_data):
#     u,y = odeint(solvr, [0, 0], theta_data).T 
#     r = 1/u
#     return r

def deriv_an_orbit(theta, a, Rp, ecc, choose):
    # we need the - in front of theta to be consistent with the function to_cylindric, where we change the orientation of the angle
    theta = -theta
    if choose == 'Keplerian':
        if ecc == 1:
            p = 2 * Rp
        else:
            p = a * (1 - ecc**2)
        dr_dtheta = p * ecc * np.sin(theta)/ (1 + ecc * np.cos(theta))**2
    # elif choose == 'Witta':
    return dr_dtheta

def find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, R0):
    """ Find the maxima density points in the radial plane for all the thetas in theta_arr"""
    x_max = np.zeros(len(theta_arr))
    y_max = np.zeros(len(theta_arr))
    z_max = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        # Exclude points inside the smoothing lenght and find radial plane
        condition_distance = np.sqrt(x_data**2 + y_data**2) > R0 
        condition_Rplane = radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, den_plane = make_slices([x_data, y_data, z_data, den_data], condition_Rplane)
        # Find and save the maximum density point
        idx_max = np.argmax(den_plane) 
        x_max[i] = x_plane[idx_max]
        y_max[i] = y_plane[idx_max]
        z_max[i] = z_plane[idx_max]
        
    return x_max, y_max, z_max    

def find_transverse_com(x_data, y_data, z_data, dim_data, den_data, mass_data, theta_arr, params):
    """ Find the centres of mass in the transverse plane"""
    Mbh, Rstar, mstar, beta = params[0], params[1], params[2], params[3]
    Rt = Rstar * (Mbh/mstar)**(1/3)
    R0 = 0.6 * (Rt / beta) 
    indeces = np.arange(len(x_data))
    # Cut a bit the data for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 50, np.abs(y_data) < 200)
    x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut, indeces_cut = \
        make_slices([x_data, y_data, z_data, dim_data, den_data, mass_data, indeces], cutting)
    # Find the radial maximum points to have a first guess of the stream (necessary for the threshold and the tg)
    x_stream_rad, y_stream_rad, z_stream_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, den_cut, theta_arr, R0)
    print('radial done')
    r_stream_rad = np.sqrt(x_stream_rad**2 + y_stream_rad**2)

    # First iteration: find the center of mass of each transverse plane corresponding to maxima-density stream
    # indeces_cmTR = np.zeros(len(theta_arr))
    x_cmTR = np.zeros(len(theta_arr))
    y_cmTR = np.zeros(len(theta_arr))
    z_cmTR = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        # Find the transverse plane
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_stream_rad, y_stream_rad, z_stream_rad, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, indeces_cut], condition_T)
        # Restrict the points to not keep points too far away.
        thresh = 8 * Rstar * (r_stream_rad[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane], condition)
        # Find the center of mass
        x_cmTR[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cmTR[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cmTR[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)

    print('Iteration radial-transverse done')
    r_cmTR= np.sqrt(x_cmTR**2 + y_cmTR**2)
    # Second iteration: find the center of mass of each transverse plane corresponding to COM stream
    # indeces_cm = np.zeros(len(theta_arr))
    x_cm = np.zeros(len(theta_arr))
    y_cm = np.zeros(len(theta_arr))
    z_cm = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        print(idx)
        # Find the transverse plane
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_cmTR, y_cmTR, z_cmTR, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, indeces_cut], condition_T)
        # Restrict the points to not keep points too far away.
        thresh = 8 * Rstar * (r_cmTR[idx]/Rp)**(1/2)
        condition_x = np.abs(x_T) < thresh
        condition_z = np.abs(z_plane) < thresh
        condition = condition_x & condition_z
        x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane = \
            make_slices([x_plane, x_T, y_plane, z_plane, mass_plane, indeces_plane], condition)
        # Find the center of mass
        x_cm[idx] = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
        y_cm[idx]= np.sum(y_plane * mass_plane) / np.sum(mass_plane)
        z_cm[idx] = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
    #     x_com = np.sum(x_plane * mass_plane) / np.sum(mass_plane)
    #     y_com = np.sum(y_plane * mass_plane) / np.sum(mass_plane)
    #     z_com = np.sum(z_plane * mass_plane) / np.sum(mass_plane)
    #     # Search in the tree the closest point to the center of mass
    #     points = np.array([x_plane, y_plane, z_plane]).T
    #     tree = KDTree(points)
    #     _, idx_cm = tree.query([x_com, y_com, z_com])
    #     indeces_cm[idx]= indeces_plane[idx_cm]
    # indeces_cm = indeces_cm.astype(int)

    return x_cm, y_cm, z_cm

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
    Rp =  Rt / beta
    indeces = np.arange(len(x_data))
    x_stream, y_stream, z_stream = stream[0], stream[1], stream[2]
    # Find the transverse plane 
    condition_T, x_Tplane, _ = transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        make_slices([x_data, y_data, z_data, dim_data, mass_data, indeces], condition_T)
    # Restrict to not keep points too far away.
    r_cm = np.sqrt(x_stream[idx]**2 + y_stream[idx]**2)
    thresh = 8 * Rstar * (r_cm/Rp)**(1/2)
    if (r_cm-thresh)< 0.6*Rp:
        print(f'The threshold to cut the TZ plane is too broad: you overcome R0 at point #{idx} of the stream')
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        make_slices([x_plane, x_Tplane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane], condition)
    mass_to_reach = 0.5 * np.sum(mass_plane)
    # Find the threshold for x
    contourT = brentq(bound_mass, 0, thresh, args=(x_Tplane, mass_plane, mass_to_reach))
    condition_contourT = np.abs(x_Tplane) < contourT
    x_T_contourT, indeces_contourT = make_slices([x_Tplane, indeces_plane], condition_contourT)

    idx_before = np.argmin(x_T_contourT)
    idx_after = np.argmax(x_T_contourT)
    x_T_low, idx_low = x_T_contourT[idx_before], indeces_contourT[idx_before]
    x_T_up, idx_up = x_T_contourT[idx_after], indeces_contourT[idx_after]
    width = x_T_up - x_T_low
    # width = np.max([width, dim_stream[idx]]) # to avoid 0 width

    # Find the threshold for z
    contourZ = brentq(bound_mass, 0, thresh, args=(z_plane, mass_plane, mass_to_reach))
    condition_contourZ = np.abs(z_plane) < contourZ
    z_contourZ, indeces_contourZ = make_slices([z_plane, indeces_plane], condition_contourZ)
    
    idx_before = np.argmin(z_contourZ)
    idx_after = np.argmax(z_contourZ)
    z_low, idx_low_h = z_contourZ[idx_before], indeces_contourZ[idx_before]
    z_up, idx_up_h = z_contourZ[idx_after], indeces_contourZ[idx_after]
    height = z_up - z_low
    # height = np.max([height, dim_stream[idx]]) # to avoid 0 height

    # Compute the number of cells in width and height using the cells in the rectangle
    rectangle = condition_contourT & condition_contourZ
    dim_cell_mean = np.mean(dim_plane[rectangle])
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
    theta_arr, stream = stream[0], [stream[1], stream[2], stream[3]]
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

def deriv_maxima(theta_arr, x_orbit, y_orbit):
    # Find the idx where the orbit starts to decrease too much (i.e. the stream is not there anymore)
    idx = len(y_orbit)
    # for i in range(len(y_orbit)):
    #     if y_orbit[i] <= y_orbit[i-1] - 5:
    #         idx = i
    #         break
    # theta_arr, x_orbit, y_orbit = theta_arr[:idx], x_orbit[:idx], y_orbit[:idx]
    # Find the numerical derivative of r with respect to theta. 
    # Shift theta to start from 0 and compute the arclenght
    theta_shift = theta_arr-theta_arr[0] 
    r_orbit = np.sqrt(x_orbit**2 + y_orbit**2)
    dr_dtheta = np.gradient(r_orbit, theta_shift)

    return dr_dtheta, idx

def find_arclenght(theta_arr, orbit, params, choose):
    """ 
    Compute the arclenght of the orbit
    Parameters
    ----------
    theta_arr : array
        The angles of the orbit
    orbit : array
        The orbit. 
        If choose is Keplerian/Witta, it is the radius(thetas). 
        If choose is Maxima, it is the x and y of the maxima.
    a, Rp, ecc : float
        The parameters of the orbit
    choose : str
        The kind of orbit. It can be 'Keplerian', 'Witta', 'Maxima'
    Returns
    -------
    s : array
        The arclenght of the orbit
    """
    if choose == 'maxima':
        drdtheta, idx = deriv_maxima(theta_arr, orbit[0], orbit[1])
        r_orbit = np.sqrt(orbit[0][:idx]**2 + orbit[1][:idx]**2)
    elif choose == 'Keplerian' or choose == 'Witta':
        a, Rp, ecc = params[0], params[1], params[2]
        drdtheta = deriv_an_orbit(theta_arr, a, Rp, ecc, choose)
        r_orbit = orbit
        idx = len(r_orbit) # to keep the whole orbit

    ds = np.sqrt(r_orbit**2 + drdtheta**2)
    theta_arr = theta_arr[:idx]-theta_arr[0] # to start from 0
    s = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        s[i] = trapezoid(ds[:i+1], theta_arr[:i+1])

    return s, idx

def orbital_energy(r, v_xy, G, M):
    # no angular momentum??
    potential = -G * M / r
    energy = 0.5 * v_xy**2 + potential
    return energy

if __name__ == '__main__':
    from Utilities.operators import make_tree, Ryan_sampler
    import matplotlib.pyplot as plt

    G = 1
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'Res20'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
    snap = '169'
    path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    Rp = Rt / beta
    params = [Mbh, Rstar, mstar, beta]
    theta_lim = np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    data = make_tree(path, snap, is_tde = True, energy = False)
    print('Tree done')
    dim_cell = data.Vol**(1/3)

    make_stream = False
    make_width = True
    compare = False
    TZslice = False
    test_s = False

    if test_s:
        params = None
        theta_arr = np.linspace(-np.pi, 0, 100)
        # x^2+y^2 = 1
        x = np.cos(theta_arr)
        y = np.sin(theta_arr)
        r = np.sqrt(x**2 + y**2)
        s, idx = find_arclenght(theta_arr, [x, y], params, choose = 'maxima')
        # arclenght of a circle with keplerian
        r_kep = keplerian_orbit(theta_arr, 1, 1, ecc = 0)
        x_kep = r_kep * np.cos(theta_arr)
        y_kep = r_kep * np.sin(theta_arr)
        s_kep, _ = find_arclenght(theta_arr, r, params = [1, 1, 0], choose = 'Keplerian')
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.plot(x,y, c = 'r')
        ax1.plot(x_kep, y_kep, '--')
        ax2.plot(theta_arr[:idx], s, c = 'r', label = 'maxima')
        ax2.plot(theta_arr, s_kep, '--', label = 'Keplerian')
        ax2.legend()
        ax2.grid()
        plt.show()

    if make_stream:
        x_stream, y_stream, z_stream = find_transverse_com(data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass, theta_arr, params)
        np.save(f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy', [theta_arr, x_stream, y_stream, z_stream])
        # streamold = np.load(f'/Users/paolamartire/shocks/data/{folder}/DIMstream_{check}{snap}.npy' )
        # theta_arr, x_streamold, y_streamold = streamold[0], streamold[1], streamold[2]

        plt.plot(x_stream, y_stream, c = 'b', label = 'COM fix width TZ plane')
        # plt.plot(x_streamold, y_streamold, c = 'orange', label = 'COM variable width TZ plane')
        plt.xlim(-300,20)
        plt.ylim(-60,60)
        plt.grid()
        plt.legend()
        # plt.savefig(f'/Users/paolamartire/shocks/Figs/FixTZStream{snap}.png')
        plt.show()  

    if make_width:
        midplane = np.abs(data.Z) < dim_cell
        X_midplane, Y_midplane, Den_midplane, Mass_midplane = make_slices([data.X, data.Y, data.Den, data.Mass], midplane)
        file = f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy' 
        stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, data.Mass, path = file, params = params)
        cm_x, cm_y, cm_z = stream[1], stream[2], stream[3]
        low_x, low_y = data.X[indeces_boundary[:,0]] , data.Y[indeces_boundary[:,0]]
        up_x, up_y = data.X[indeces_boundary[:,1]] , data.Y[indeces_boundary[:,1]]

        #%%
        plt.figure(figsize = (12,4))
        img = plt.scatter(X_midplane, Y_midplane, c = Den_midplane, s = 1, cmap = 'viridis', alpha = 0.2, vmin = 2e-8, vmax = 6e-8)
        cbar = plt.colorbar(img)
        cbar.set_label(r'Density', fontsize = 16)
        plt.plot(cm_x, cm_y,  c = 'k')
        plt.plot(low_x, low_y, '--', c = 'k')
        plt.plot(up_x, up_y, '-.', c = 'k', label = 'Upper tube')
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.xlim(-200,30)
        plt.ylim(-40,60)
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{check}/stream{snap}.png', dpi=300)
        plt.show()