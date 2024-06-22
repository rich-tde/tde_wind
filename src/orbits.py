""" 
Find different kind of orbits for TDEs. 
Measure the width and the height of the stream.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# from scipy.ndimage import gaussian_filter1d
from Utilities.sections import make_slices, radial_plane, transverse_plane
from Utilities.operators import sort_list, median_array, from_cylindric, to_cylindric

def make_cfr(R, x0=0, y0=0):
    x = np.linspace(-R, R, 100)
    y = np.linspace(-R, R, 100)
    xcfr, ycfr = np.meshgrid(x,y)
    cfr = (xcfr-x0)**2 + (ycfr-y0)**2 - R**2
    return xcfr, ycfr, cfr

def keplerian_orbit(theta, apo, a, ecc=1):
    # we expect theta as from the function to_cylindric, i.e. clockwise. 
    # You have to mirror it to get the angle for the usual polar coordinates.
    theta = -theta
    if ecc == 1:
        p = 2 * a
        radius = p / (1 + ecc * np.cos(theta))
    else:
        radius = apo * (1 - ecc**2) / (1 + ecc * np.cos(theta))
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

def find_maximum(x_mid, y_mid, dim_mid, den_mid, theta_arr, Rt, window_size=3):
    """ Find the maxima density points in a plane (midplane)"""
    x_cm = np.zeros(len(theta_arr))
    y_cm = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        condition_Rplane = radial_plane(x_mid, y_mid, dim_mid, theta_arr[i])
        condition_distance = np.sqrt(x_mid**2 + y_mid**2) > Rt # to avoid the center
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        x_plane = x_mid[condition_Rplane]
        y_plane = y_mid[condition_Rplane]
        # Order for radial distance to smooth the density
        r_plane = list(np.sqrt(x_plane**2 + y_plane**2))
        den_plane_sorted = sort_list(den_mid[condition_Rplane], r_plane)
        x_plane_sorted = sort_list(x_plane, r_plane)
        y_plane_sorted = sort_list(y_plane, r_plane)
        den_median = median_array(den_plane_sorted, window_size)

        idx_cm = np.argmax(den_median) 
        x_cm[i] = x_plane_sorted[idx_cm]
        y_cm[i] = y_plane_sorted[idx_cm]
        
    return x_cm, y_cm

def find_stream_boundaries(x_data, y_data, z_data, dim_data, den_data, x_orbit, y_orbit, idx, threshold = 0.33):
    # Find the transverse plane 
    condition_T, x_Tplane, x0 = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, idx, coord = True)
    x_plane, y_plane, z_plane, dim_plane, den_plane = make_slices([x_data, y_data, z_data, dim_data, den_data], condition_T)
    # Find transverse line (on midplane)
    midplane = np.abs(z_plane) < dim_plane
    x_mid, x_T_mid, y_mid, dim_mid, den_mid = make_slices([x_plane, x_Tplane, y_plane, dim_plane, den_plane], midplane)

    # Restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
    condition_x = np.abs(x_T_mid) < 20
    x_mid = x_mid[condition_x]
    x_T_mid = x_T_mid[condition_x]
    y_mid = y_mid[condition_x]
    dim_mid = dim_mid[condition_x]
    den_mid = den_mid[condition_x]

    # Sort in x_T
    x_T_mid = list(x_T_mid)
    x_Tmid_sorted = sorted(x_T_mid)
    x_mid_sorted = sort_list(x_mid, x_T_mid)
    y_mid_sorted = sort_list(y_mid, x_T_mid)
    dim_mid_sorted = sort_list(dim_mid, x_T_mid)
    den_mid_sorted = sort_list(den_mid, x_T_mid)
    den_median_mid_sorted = median_array(den_mid_sorted)

    # find the cm of the plane
    idx_cm = np.argmin(np.abs(x_mid_sorted-x_orbit[idx]))#np.argmax(den_mid_sorted) 
    print('x cm difference: ', x_mid_sorted[idx_cm]-x_orbit[idx])
    x_cm, y_cm, den_cm = x_mid_sorted[idx_cm], y_mid_sorted[idx_cm], den_mid_sorted[idx_cm]
    
    # Walk before and after the cm till you find a density 3 times smaller
    # Lower boundary
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step > 0:
        idx_step -= 1
        den_tube = den_median_mid_sorted[idx_step] #den_mid_sorted[idx_step]
    idx_before = idx_step+1
    x_low, x_T_low, y_low, den_low_w = x_mid_sorted[idx_before], x_Tmid_sorted[idx_before], y_mid_sorted[idx_before], den_tube

    # Upper boundary
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step < len(den_mid_sorted) - 1:
        idx_step += 1
        den_tube  = den_median_mid_sorted[idx_step] #den_mid_sorted[idx_step]
    idx_after = idx_step-1
    x_up, x_T_up, y_up, den_up_w = x_mid_sorted[idx_after], x_Tmid_sorted[idx_after], y_mid_sorted[idx_after], den_tube
    
    # Compute the width
    width = x_T_up - x_T_low
    dim_cell_mean = (dim_mid_sorted[idx_after] + dim_mid_sorted[idx_before]) / 2
    # ncells_w = idx_after - idx_before + 1 # +1 because ideces start from 0
    ncells_w = int(width / dim_cell_mean) # round to the nearest integer

    # Z direction
    # take point on the vertical axis
    # x_Tplane_vert, z_vert, den_vert = make_slices([x_Tplane, z_plane, den_plane], vertical)
    vertical = np.abs(x_Tplane-x0) < dim_plane #or x_plane-x_cm
    z_vert, dim_vert, den_vert = make_slices([z_plane, dim_plane, den_plane], vertical)

    # sort 
    z_vert = list(z_vert)
    z_vert_sorted = sorted(z_vert)
    # x_vert_sorted = sort_list(x_plane_vert, z_vert)
    dim_vert_sorted = sort_list(dim_vert, z_vert)
    den_vert_sorted = sort_list(den_vert, z_vert)
    den_median_vert_sorted = median_array(den_vert_sorted)

    # Find the cm of the plane (you are assuming that the cm found before is at z=0)
    idx_cm = np.argmin(np.abs(z_vert_sorted))    

    # Lower boundary in height
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step > 0:
        idx_step -= 1
        den_tube = den_median_vert_sorted[idx_step] #den_vert_sorted[idx_step]
    idx_before = idx_step + 1
    # x_low = x_vert_sorted[idx_before]
    z_low, den_low_h = z_vert_sorted[idx_before], den_tube

    # Upper boundary in height
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step < len(den_vert_sorted) - 1:
        idx_step += 1
        den_tube  = den_median_vert_sorted[idx_step] #den_vert_sorted[idx_step]
    idx_after = idx_step-1
    z_up, den_up_h = z_vert_sorted[idx_after], den_tube

    # Compute the height
    height = z_up - z_low 
    #ncells_h = idx_after - idx_before + 1 # +1 because ideces start from 0
    dim_cell_mean = (dim_vert_sorted[idx_after] + dim_vert_sorted[idx_before]) / 2
    ncells_h = int(height / dim_cell_mean) # round to the nearest integer
    # Wrap. Assumption: z=0 for upper/lower tube for width, xplane=x0~0 for upper/lower tube for height
    cm = np.array([x_cm, y_cm, den_cm])
    lower_tube_w = np.array([x_low, x_T_low, y_low, den_low_w])
    upper_tube_w = np.array([x_up, x_T_up, y_up, den_up_w])
    lower_tube_h = np.array([z_low, den_low_h])
    upper_tube_h = np.array([z_up, den_up_h])
    w_params = np.array([width, ncells_w])
    h_params = np.array([height, ncells_h])

    return cm, lower_tube_w, upper_tube_w, lower_tube_h, upper_tube_h, w_params, h_params

def follow_the_stream(x_data, y_data, z_data, dim_data, den_data, theta_arr, Rt, threshold = 1./3):
    midplane = np.abs(z_data) < dim_data
    x_mid, y_mid, dim_mid, den_mid = make_slices([x_data, y_data, dim_data, den_data], midplane)
    x_orbit, y_orbit = find_maximum(x_mid, y_mid, dim_mid, den_mid, theta_arr, Rt)
    cm = []
    lower_tube_w = []
    upper_tube_w = []
    lower_tube_h = []
    upper_tube_h = []
    w_params = []
    h_params = []
    for i in range(len(theta_arr)):
        cm_i, lower_tube_w_i, upper_tube_w_i, lower_tube_h_i, upper_tube_h_i, w_params_i, h_params_i = \
            find_stream_boundaries(x_data, y_data, z_data, dim_data, den_data, x_orbit, y_orbit, i, threshold)
        cm.append(cm_i)
        lower_tube_w.append(lower_tube_w_i)
        upper_tube_w.append(upper_tube_w_i)
        lower_tube_h.append(lower_tube_h_i)
        upper_tube_h.append(upper_tube_h_i)
        w_params.append(w_params_i)
        h_params.append(h_params_i)
    cm = np.array(cm)
    lower_tube_w = np.array(lower_tube_w)
    upper_tube_w = np.array(upper_tube_w)
    lower_tube_h = np.array(lower_tube_h)
    upper_tube_h = np.array(upper_tube_h)
    w_params = np.transpose(np.array(w_params)) # line 1: width, line 2: ncells
    h_params = np.transpose(np.array(h_params)) # line 1: height, line 2: ncells
    return cm, lower_tube_w, upper_tube_w, lower_tube_h, upper_tube_h, w_params, h_params

def orbital_energy(r, v_xy, G, M):
    # no angular momentum??
    potential = -G * M / r
    energy = 0.5 * v_xy**2 + potential
    return energy


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test clockwise polar coordinates 
    R0 = 1
    xcfr0, ycfr0, cfr0 = make_cfr(R0)
    theta = np.array([-np.pi, -np.pi/4, 0, np.pi/4, np.pi/2, 3])
    colors = ['b', 'g', 'r', 'orchid', 'y', 'c']
    x, y = from_cylindric(theta, R0)
    plt.xlim(-2*R0, 2*R0)
    plt.ylim(-2*R0, 2*R0)
    plt.scatter(x,y, c = colors)
    plt.contour(xcfr0, ycfr0, cfr0, [0], linestyles = 'dotted', colors = 'k')
    plt.title('To cartesian coordinates')
    plt.show()

    # Test from polar to cartesian
    plt.figure()
    x2 = np.array([1, 0, -1, 0])
    y2 = np.array([0, 1, 0, -1])
    colors = ['b', 'g', 'r', 'orchid']
    theta2, r2 = to_cylindric(x2,y2)
    plt.scatter(theta2, r2,  c=colors)
    plt.title('To polar coordinates')
    plt.show()