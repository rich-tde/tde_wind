""" 
Find different kind of orbits (and their derivatives) for TDEs. 
Identify the stream using density maxima points.
Measure the width and the height of the stream.
NB: the densiy data have to be smoothed!
"""
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, odeint
from Utilities.sections import make_slices, radial_plane, transverse_plane
from Utilities.operators import sort_list#, average_array

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
        dr_dtheta = p * np.sin(theta)/ (1 + ecc * np.cos(theta))**2
    # elif choose == 'Witta':
    return dr_dtheta

def find_radial_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt):
    """ Find the maxima density points in a plane (midplane)"""
    x_cm = np.zeros(len(theta_arr))
    y_cm = np.zeros(len(theta_arr))
    z_cm = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        condition_Rplane = radial_plane(x_data, y_data, dim_data, theta_arr[i])
        condition_distance = np.sqrt(x_data**2 + y_data**2) > Rt # to avoid the center
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane = x_data[condition_Rplane]
        y_plane = y_data[condition_Rplane]
        z_plane = z_data[condition_Rplane]
        den_plane = den_data[condition_Rplane]
        idx_cm = np.argmax(den_plane) 
        x_cm[i] = x_plane[idx_cm]
        y_cm[i] = y_plane[idx_cm]
        z_cm[i] = z_plane[idx_cm]
        
    return x_cm, y_cm, z_cm    

def find_transverse_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt):
    """ Find the maxima density points in a plane (transverse)"""
    indeces = np.arange(len(x_data))
    # Cut a bit the data for computational reasons
    cutting = np.logical_and(np.abs(z_data) < 50, np.abs(y_data) < 200)
    x_cut, y_cut, z_cut, dim_cut, den_cut, indeces_cut = \
        make_slices([x_data, y_data, z_data, dim_data, den_data, indeces], cutting)
    x_orbit_rad, y_orbit_rad, z_orbit_rad = find_radial_maximum(x_cut, y_cut, z_cut, dim_cut, den_cut, theta_arr, Rt)
    indeces_cmTR = np.zeros(len(theta_arr))
    for idx in range(len(theta_arr)):
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, dim_cut, x_orbit_rad, y_orbit_rad, idx, coord = True)
        # condition to not go too far away. Important for theta = 0
        den_plane, indeces_plane = make_slices([den_cut, indeces_cut], condition_T)
        condition_x = np.abs(x_T) < 20
        den_plane, indeces_plane = make_slices([den_plane, indeces_plane], condition_x)
        idx_cm = np.argmax(den_plane)
        indeces_cmTR[idx]= indeces_plane[idx_cm]
    indeces_cm = np.zeros(len(theta_arr))
    x_cmTR = x_data[indeces_cmTR.astype(int)]
    y_cmTR = y_data[indeces_cmTR.astype(int)]
    for idx in range(len(theta_arr)):
        condition_T, x_T, _ = transverse_plane(x_cut, y_cut, dim_cut, x_cmTR, y_cmTR, idx, coord = True)
        # condition to not go too far away. Important for theta = 0
        den_plane, indeces_plane = make_slices([den_cut, indeces_cut], condition_T)
        condition_x = np.abs(x_T) < 20
        den_plane, indeces_plane = make_slices([den_plane, indeces_plane], condition_x)
        idx_cm = np.argmax(den_plane)
        indeces_cm[idx]= indeces_plane[idx_cm]
    indeces_cm = indeces_cm.astype(int)
    return indeces_cm

def find_single_boundaries(x_data, y_data, z_data, dim_data, den_data, indeces_orbit, idx, threshold):
    indeces = np.arange(len(x_data))
    x_orbit, y_orbit, z_orbit = x_data[indeces_orbit], y_data[indeces_orbit], z_data[indeces_orbit]
    # Find the transverse plane 
    condition_T, x_Tplane, x0 = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, den_plane, indeces_plane = \
        make_slices([x_data, y_data, z_data, dim_data, den_data, indeces], condition_T)
    # Find transverse orbital line (transverse intersect midplane)
    orbital = np.abs(z_plane-z_orbit[idx]) < dim_plane
    x_orb, x_T_orb, y_orb, z_orb, dim_orb, den_orb, indeces_orb = \
        make_slices([x_plane, x_Tplane, y_plane, z_plane, dim_plane, den_plane, indeces_plane], orbital)
    # Restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
    condition_x = np.abs(x_T_orb) < 20
    x_orb, x_T_orb, y_orb, z_orb, dim_orb, den_orb, indeces_orb = \
        make_slices([x_orb, x_T_orb, y_orb, z_orb, dim_orb, den_orb, indeces_orb], condition_x)

    # Sort in x_T
    x_T_orb = list(x_T_orb)
    x_Torb_sorted = sorted(x_T_orb)
    x_orb_sorted = sort_list(x_orb, x_T_orb)
    dim_orb_sorted = sort_list(dim_orb, x_T_orb)
    den_orb_sorted = sort_list(den_orb, x_T_orb)
    indeces_orb_sorted = sort_list(indeces_orb, x_T_orb)
    # Find the idx of the cm of the plane
    idx_cm = np.argmin(np.abs(x_orb_sorted-x_orbit[idx]))
    # print('Check if x_orbit[idx] is the min of x_min_sorted: difference = ', x_orb_sorted[idx_cm]-x_orbit[idx])
    den_cm = den_orb_sorted[idx_cm]
    
    # Walk before and after the cm till you find the threshold density 
    # Lower boundary
    idx_step = idx_cm
    den_tube = den_cm

    while den_tube > threshold and idx_step > 0:
        idx_step -= 1
        # den_tube =  den_aver_orb_sorted[idx_step] 
        den_tube = den_orb_sorted[idx_step]
    idx_before = idx_step + 1 # because the last step has density < threshold
    idx_low, x_T_low = indeces_orb_sorted[idx_before], x_Torb_sorted[idx_before]

    # Upper boundary
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold and idx_step < len(den_orb_sorted) - 1:
        idx_step += 1
        den_tube = den_orb_sorted[idx_step]
    idx_after = idx_step - 1 # because the last step has density < threshold
    idx_up, x_T_up = indeces_orb_sorted[idx_after], x_Torb_sorted[idx_after]
    
    # Compute the width
    width = x_T_up - x_T_low
    width = np.max([width, dim_orb_sorted[idx_cm]]) # to avoid 0 width
    dim_cell_mean = np.mean(dim_orb_sorted[idx_before:idx_after+1])
    ncells_w = np.round(width/dim_cell_mean, 0) # round to the nearest integer

    # Find transverse vertical line (transverse intersect xT=0)
    # print('x0 in follow_the_stream = ', x0)
    vertical = np.abs(x_Tplane-x0) < dim_plane #or x_plane-x_cm
    z_vert, dim_vert, den_vert, indeces_vert = \
        make_slices([z_plane, dim_plane, den_plane, indeces_plane], vertical)

    # Sort 
    z_vert = list(z_vert)
    z_vert_sorted = sorted(z_vert)
    # x_vert_sorted = sort_list(x_plane_vert, z_vert)
    dim_vert_sorted = sort_list(dim_vert, z_vert)
    den_vert_sorted = sort_list(den_vert, z_vert)
    indeces_vert_sorted = sort_list(indeces_vert, z_vert)

    # Find the cm of the plane (you are assuming that the cm found before is at z=0)
    idx_cm = np.argmin(np.abs(z_vert_sorted-z_orbit[idx]))    

    # Lower boundary in height
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold and idx_step > 0:
        idx_step -= 1
        den_tube = den_vert_sorted[idx_step]
    idx_before = idx_step + 1
    # x_low = x_vert_sorted[idx_before]
    z_low, idx_low_h = z_vert_sorted[idx_before], indeces_vert_sorted[idx_before]

    # Upper boundary in height
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold and idx_step < len(den_vert_sorted) - 1:
        idx_step += 1
        den_tube = den_vert_sorted[idx_step] 
    idx_after = idx_step - 1
    z_up, idx_up_h = z_vert_sorted[idx_after], indeces_vert_sorted[idx_after]

    # Compute the height
    height = z_up - z_low 
    height = np.max([height, dim_vert_sorted[idx_cm]]) # to avoid 0 height
    dim_cell_mean = np.mean(dim_vert_sorted[idx_before:idx_after+1])
    ncells_h = np.round(height/dim_cell_mean, 0) # round to the nearest integer

    # Wrap. Assumption: z=0 for upper/lower tube for width, xplane=x0~0 for upper/lower tube for height
    # print('x_T_cm: ', x_T_cm)
    # cm = np.array([x_cm, x_T_cm, y_cm, z_cm, den_cm])
    indeces_boundary = np.array([idx_low, idx_up, idx_low_h, idx_up_h]).astype(int)
    x_T_width = np.array([x_T_low, x_T_up])
    w_params = np.array([width, ncells_w])
    h_params = np.array([height, ncells_h])

    return indeces_boundary, x_T_width, w_params, h_params

def follow_the_stream(x_data, y_data, z_data, dim_data, den_data, theta_arr, den_thresh, Rt, path = 'none'):
    """ Find width and height all along the stream """
    # Find the center of mass of the stream for each theta
    try:
        streamLow = np.load(path)
        print('existing file.')
        theta_arr, indeces_stream = streamLow[0], streamLow[1].astype(int)
    except:
        print('Computing orbit')
        indeces_stream = find_transverse_maximum(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt)
    indeces_boundary = []
    x_T_width = []
    w_params = []
    h_params = []
    for i in range(len(indeces_stream)):
        print(i)
        indeces_boundary_i, x_T_width_i, w_params_i, h_params_i = \
            find_single_boundaries(x_data, y_data, z_data, dim_data, den_data, indeces_stream, i, den_thresh[i])
        indeces_boundary.append(indeces_boundary_i)
        x_T_width.append(x_T_width_i)
        w_params.append(w_params_i)
        h_params.append(h_params_i)
    indeces_boundary = np.array(indeces_boundary).astype(int)
    x_T_width = np.array(x_T_width)
    w_params = np.transpose(np.array(w_params)) # line 1: width, line 2: ncells
    h_params = np.transpose(np.array(h_params)) # line 1: height, line 2: ncells
    return indeces_stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr

def deriv_maxima(theta_arr, x_orbit, y_orbit):
    # Find the idx where the orbit starts to decrease too much (i.e. the stream is not there anymore)
    for i in range(len(y_orbit)):
        if y_orbit[i] <= y_orbit[i-1] - 5:
            idx = i
    theta_arr, x_orbit, y_orbit = theta_arr[:idx], x_orbit[:idx], y_orbit[:idx]
    # Find the numerical derivative of r with respect to theta. 
    # Shift theta to start from 0 and compute the arclenght
    theta_shift = theta_arr-theta_arr[0] 
    r_orbit = np.sqrt(x_orbit**2 + y_orbit**2)
    dr_dtheta = np.zeros(len(theta_arr))
    for i in range(len(theta_shift)):
        dr_dtheta[i] = trapezoid(r_orbit[:i+1], theta_shift[:i+1])

    return dr_dtheta, idx

def find_arclenght(theta_arr, orbit, a, Rp, ecc, choose):
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
    if choose == 'Maxima':
        drdtheta, idx = deriv_maxima(theta_arr, orbit[0], orbit[1])
        r_orbit = np.sqrt(orbit[0][:idx]**2 + orbit[1][:idx]**2)
    elif choose == 'Keplerian' or choose == 'Witta':
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
    check = 'HiRes'
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
    snap = '164'
    snapL = '164'
    path = f'TDE/{folder}{check}/{snap}'
    Rt = Rstar * (Mbh/mstar)**(1/3)
    theta_lim = np.pi
    step = 0.02
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)
    xcfr, ycfr, cfr = make_cfr(Rt)
    data = make_tree(path, snap, is_tde = True, energy = False)
    dim_cell = data.Vol**(1/3)

    path = f'data/{folder}/stream_{check}{snap}_{step}.npy' 
    density = np.load(f'TDE/{folder}{check}/{snap}/smoothed_Den_{snap}.npy')

    streamL = np.load(f'data/{folder}/stream_Low{snapL}_{step}.npy')
    indeces_orbitL = streamL[1].astype(int)
    DenL = np.load(f'TDE/{folder}Low/{snapL}/smoothed_Den_{snapL}.npy')
    stream_thresh = DenL[indeces_orbitL]/3

    # indeces_stream, indeces_boundary, x_T_width, w_params, h_params  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, density, theta_arr, Rt, threshold=1/3)
    indeces_stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = follow_the_stream(data.X, data.Y, data.Z, dim_cell, density, theta_arr, stream_thresh, Rt, path = path, other = True)

    cm_x, cm_y = data.X[indeces_stream], data.Y[indeces_stream]
    low_x, low_y = data.X[indeces_boundary[:,0]] , data.Y[indeces_boundary[:,0]]
    up_x, up_y = data.X[indeces_boundary[:,1]] , data.Y[indeces_boundary[:,1]]
    plt.plot(cm_x[30:230], cm_y[30:230],  c = 'k', label = 'Orbit')
    plt.plot(low_x[30:230], low_y[30:230], '--', c = 'k',label = 'Lower tube')
    plt.plot(up_x[30:230], up_y[30:230], '-.', c = 'k', label = 'Upper tube')
    plt.xlim(-300,30)
    plt.ylim(-80,80)
    plt.legend()
    plt.show()

    