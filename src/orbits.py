""" 
Find different kind of orbits (and their derivatives) for TDEs. 
Identify the stream using the centre of mass.
Measure the width and the height of the stream.
"""
import sys
sys.path.insert(0,'/Users/paolamartire/shocks/')
import numpy as np
import numba
import Utilities.prelude
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import odeint
from scipy.integrate import trapezoid
from Utilities.sections import transverse_plane, make_slices, radial_plane

def make_cfr(R, x0=0, y0=0):
    x = np.linspace(-R, R, 100)
    y = np.linspace(-R, R, 100)
    xcfr, ycfr = np.meshgrid(x,y)
    cfr = (xcfr-x0)**2 + (ycfr-y0)**2 - R**2
    return xcfr, ycfr, cfr

def keplerian_energy(Mbh, G, t):
    # specific orbital energy of a Keplerian orbit
    energy = (np.pi * G * Mbh / (2 * t))**(2/3)
    return energy

def Mdot_fb(Mbh, G, t, dmdE):
    """ Mass fallback rate according to Keplers law for t and dM/dE from numerical simualtion data."""
    Mdot = -2/3 * dmdE * (np.pi * G * Mbh / 2)**(2/3) * t**(-5/3)
    return Mdot

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

def R_grav(Mbh, c, G):
    """ Gravitational radius of the black hole."""
    Rg = G * Mbh /c**2
    return Rg

def tidal_radius(Rstar, mstar, Mbh):
    Rt = Rstar * (Mbh/mstar)**(1/3)
    return Rt

def semimajor_axis(Rstar, mstar, Mbh, G):
    """ Semimajor axis of the most bound debris """
    E = energy_mb(Rstar, mstar, Mbh, G)
    a = G * Mbh / (2*E)
    return a

def pericentre(Rstar, mstar, Mbh, beta):
    Rt = tidal_radius(Rstar, mstar, Mbh)
    print(Rt)
    Rp = Rt/beta
    return Rp

def apocentre(Rstar, mstar, Mbh, beta):
    # comes from Ra=a(1+e), a=Rt^2/2Rstar, e=1-2*Rstar/(beta*Rt)
    Rt = Rstar * (Mbh/mstar)**(1/3)
    apo = Rt**2/Rstar - Rt/beta 
    return apo

def eccentricity(Rstar, mstar, Mbh, beta):
    # eccentricity of most bound derbis. It comes from Rp = a(1-e), a = Rt^2/2Rstar
    Rt = Rstar * (Mbh/mstar)**(1/3)
    ecc = 1-2*Rstar/(beta*Rt)
    return ecc

def energy_mb(Rstar, mstar, Mbh, G):
    """ Specific orbital energy of most bound debris """
    Rt = Rstar * (Mbh/mstar)**(1/3)
    En = G * Mbh * Rstar / Rt**2
    return En

def parameters_orbit(Rp, Ra, Mbh, c, G ):
    # Rp, Ra, Mbh, c, G = params_orb[0], params_orb[1], params_orb[2], params_orb[3], params_orb[4]
    Rs = 2 * G * Mbh / c**2
    En = G * Mbh * (Rp**2 * (Ra-Rs) - Ra**2 * (Rp-Rs)) / ((Ra**2-Rp**2) * (Rp-Rs) * (Ra-Rs))
    L = np.sqrt(2 * Ra**2 * (En + G*Mbh/(Ra-Rs)))
    return Rs, En, L

def solvr(x, theta, Rp, Ra, Mbh, c, G ):
    Rs, _, L = parameters_orbit(Rp, Ra, Mbh, c, G)
    u,y = x
    res =  np.array([y, (-u + G * Mbh / ((1 - Rs*u) * L)**2)])
    return res

def Witta_orbit(theta_data, Rp, Ra, Mbh, c, G):
    theta_data += np.pi
    u,y = odeint(solvr, [0, 0], theta_data, args = (Rp, Ra, Mbh, c, G)).T 
    r = 1/u
    return r

def precession_angle(Rstar, mstar, Mbh, beta, c, G):
    a = semimajor_axis(Rstar, mstar, Mbh, beta)
    e = eccentricity(Rstar, mstar, Mbh, beta)
    theta = 6 * np.pi * G * Mbh / (c**2 * a * (1-e**2))
    return theta

def R_selfinter(Rstar, mstar, Mbh, beta, c, G):
    theta = precession_angle(Rstar, mstar, Mbh, beta, c, G)
    e_mb = eccentricity(Rstar, mstar, Mbh, beta)
    Rt = tidal_radius(Rstar, mstar, Mbh)
    R = (1+e_mb) * Rt / (beta * (1-e_mb * np.cos(theta/2)))
    return R

def orbital_energy(r, vel, mass, G, c, M, R0):
    # no angular momentum??
    Rs = 2*G*M/c**2
    potential = np.where( r <= R0, -G * M * r**2 / (2 * R0 * (R0 - Rs)**2), -G * M / (r - Rs))
    energy = mass * (0.5 * vel**2 + potential)
    return energy

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
        condition_Rplane = radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane, y_plane, z_plane, den_plane = make_slices([x_data, y_data, z_data, den_data], condition_Rplane)
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
    apo = apocentre(Rstar, mstar, Mbh, beta)
    # indeces = np.arange(len(x_data))
    # Cut a bit the data for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 100, np.abs(y_data) < 10*apo)
    x_cut, y_cut, z_cut, dim_cut, den_cut, mass_cut = \
        make_slices([x_data, y_data, z_data, dim_data, den_data, mass_data], cutting)
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
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_stream_rad, y_stream_rad, z_stream_rad, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, den_plane, dim_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, den_cut, dim_cut], condition_T)
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
            make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
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
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, z_cut, dim_cut, x_cmTR, y_cmTR, z_cmTR, idx, coord = True)
        x_plane, y_plane, z_plane, mass_plane, den_plane, dim_plane = \
            make_slices([x_cut, y_cut, z_cut, mass_cut, den_cut, dim_cut], condition_T)
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
            make_slices([x_plane, y_plane, z_plane, mass_plane], condition)
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
    condition_T, x_Tplane, _ = transverse_plane(x_data, y_data, z_data, dim_data, x_stream, y_stream, z_stream, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, mass_plane, indeces_plane = \
        make_slices([x_data, y_data, z_data, dim_data, mass_data, indeces], condition_T)
    # Restrict to not keep points too far away.
    r_spherical_plane = np.sqrt(x_plane**2 + y_plane**2 + z_plane**2)
    # thresh = get_threshold(x_Tplane, z_plane, r_plane, mass_plane, dim_plane, R0) #8 * Rstar * (r_cm/Rp)**(1/2)
    thresh = thresh_stream[idx]
    condition_x = np.abs(x_Tplane) < thresh
    condition_z = np.abs(z_plane) < thresh
    condition = condition_x & condition_z
    x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, mass_plane, indeces_plane = \
        make_slices([x_plane, x_Tplane, y_plane, z_plane, r_spherical_plane, dim_plane, mass_plane, indeces_plane], condition)
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
    x_T_contourTpositive, indeces_contourTpositive = make_slices([x_Tplanepositive, indices_planepositive], condition_contourTpositive)

    contourTnegative = brentq(bound_mass, 0, thresh, args=(x_Tplanenegative, mass_planenegative, mass_to_reach))
    condition_contourTnegative = np.abs(x_Tplanenegative) < contourTnegative
    x_T_contourTnegative, indeces_contourTnegative = make_slices([x_Tplanenegative, indices_planenegative], condition_contourTnegative)

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
    z_contourZpositive, indeces_contourZpositive = make_slices([z_planepositive, indices_planepositive], condition_contourZpositive)

    contourZnegative = brentq(bound_mass, 0, thresh, args=(z_planenegative, mass_planenegative, mass_to_reach))
    condition_contourZnegative = np.abs(z_planenegative) < contourZnegative
    z_contourZnegative, indeces_contourZnegative = make_slices([z_planenegative, indices_planenegative], condition_contourZnegative)

    # contourZ = brentq(bound_mass, 0, thresh, args=(z_plane, mass_plane, mass_to_reach))
    # condition_contourZ = np.abs(z_plane) < contourZ
    # z_contourZ, indeces_contourZ = make_slices([z_plane, indeces_plane], condition_contourZ)
    
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


if __name__ == '__main__':
    from Utilities.operators import make_tree, Ryan_sampler
    import matplotlib.pyplot as plt
    
    G = 1
    G_SI = 6.6743e-11
    Msol = 2e30 #1.98847e30 # kg
    Rsol = 7e8 #6.957e8 m
    t = np.sqrt(Rsol**3 / (Msol*G_SI ))
    c = 3e8 / (Rsol/t)
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = ''
    compton = 'Compton'

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
    snap = '164'
    path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    Rp = Rt / beta
    Ra = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)
    a = Ra/2
    # params_orb = [Rp, Ra, Mbh, c, G]
    params = [Mbh, Rstar, mstar, beta]
    theta_lim = np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    theta_arr = theta_arr[:250]
    data = make_tree(path, snap, energy = False)
    dim_cell = data.Vol**(1/3)
    cut = data.Den>1e-19
    X, Y, Z, dim_cell, Den, Mass = \
        make_slices([data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass], cut)

    make_stream = True
    make_width = True
    test_s = False
    test_orbit = False

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
    
    if test_orbit:
        from Utilities.operators import from_cylindric
        r_Witta = Witta_orbit(theta_arr, Rp, Ra, Mbh, c, G)
        x_Witta, y_Witta = from_cylindric(theta_arr, r_Witta)
        r_kepler = keplerian_orbit(theta_arr, a, Rp, ecc = 1)
        x_kepler, y_kepler = from_cylindric(theta_arr, r_kepler)
        plt.plot(x_Witta, y_Witta, c = 'r', label = 'Witta')
        plt.plot(x_kepler, y_kepler, '--', c = 'b', label = 'Keplerian')
        plt.xlim(-300,20)
        plt.ylim(-60,60)
        plt.legend()
        plt.grid()
        plt.show()

    if make_stream:
        x_stream, y_stream, z_stream, thresh_cm, x_cmTR, y_cmTR, z_cmTR, x_stream_rad, y_stream_rad, z_stream_rad = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr, params, test = True)
        np.save(f'/Users/paolamartire/shocks/data/{folder}/streamRad_{check}{snap}.npy', [theta_arr, x_stream_rad, y_stream_rad, z_stream_rad])
        np.save(f'/Users/paolamartire/shocks/data/{folder}/streamcmTR_{check}{snap}.npy', [theta_arr, x_cmTR, y_cmTR, z_cmTR])
        np.save(f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy', [theta_arr, x_stream, y_stream, z_stream, thresh_cm])

        #%%
        plt.figure(figsize = (10,6))
        plt.plot(x_stream_rad, y_stream_rad, c = 'b', label = 'Max density stream')
        plt.plot(x_cmTR, y_cmTR, c = 'r', label = 'COM TZ plane of max density points')
        plt.plot(x_stream, y_stream, c = 'k', label = 'COM TZ plane of previous COM points')
        plt.xlim(-300,20)
        plt.ylim(-60,60)
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.grid()
        plt.legend()
        plt.savefig(f'/Users/paolamartire/shocks/Figs/WH/TZStream{snap}.png')
        plt.show() 

        plt.figure(figsize = (6,6))
        plt.plot(x_stream_rad, y_stream_rad, c = 'b', label = 'Max density stream')
        plt.plot(x_cmTR, y_cmTR, c = 'r', label = 'COM TZ plane of max density points')
        plt.plot(x_stream, y_stream, c = 'k', label = 'COM TZ plane of previous COM points')
        plt.xlim(-60,20)
        plt.ylim(-50,50)
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.grid()
        plt.legend()
        plt.show() 
    #%%
    if make_width:
        file = f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy' 
        stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = follow_the_stream(X, Y, Z, dim_cell, Mass, path = file, params = params)
        cm_x, cm_y, cm_z = stream[0], stream[1], stream[2]
        low_x, low_y = X[indeces_boundary[:,0]] , Y[indeces_boundary[:,0]]
        up_x, up_y = X[indeces_boundary[:,1]] , Y[indeces_boundary[:,1]]

        #%%
        # midplane = Z < dim_cell
        # X_midplane, Y_midplane, Den_midplane, Mass_midplane = make_slices([X, Y, Den, Mass], midplane)
        plt.figure(figsize = (12,8))
        # img = plt.scatter(X_midplane, Y_midplane, c = Den_midplane, s = 1, cmap = 'viridis', alpha = 0.2, vmin = 2e-8, vmax = 6e-8)
        # cbar = plt.colorbar(img)
        # cbar.set_label(r'Density', fontsize = 16)
        # plt.plot(cm_x, cm_y,  c = 'k')
        plt.plot(low_x[:-10], low_y[:-10], '--', c = 'k')
        plt.plot(up_x[:-10], up_y[:-10], '-.', c = 'k', label = 'Upper tube')
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.xlim(-300,30)
        plt.ylim(-75,75)
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{check}/stream{snap}.png', dpi=300)
        plt.show()
# %%
