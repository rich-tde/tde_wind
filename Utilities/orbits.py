""" 
Find different kind of orbits for TDEs. 
Measure the width and the height of the stream.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Utilities.sections import radial_plane, transverse_plane
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

def find_maximum(x_mid, y_mid, dim_mid, den_mid, theta_params):
    # find the orbit given by maxima density points
    # It's better to give theta_params than theta_arr, so that you can adjust the step
    theta_arr = np.arange(*theta_params)
    step = theta_params[2]
    if step < 0.1:
        step = 0.1
    radius_arr = np.zeros(len(theta_arr))
    for i,theta_cm in enumerate(theta_arr):
        condition_Rplane = radial_plane(x_mid, y_mid, dim_mid, theta_cm)
        idx_cm = np.argmax(den_mid[condition_Rplane])
        x_cm = x_mid[condition_Rplane][idx_cm]
        y_cm = y_mid[condition_Rplane][idx_cm]
        radius_arr[i] = np.sqrt(x_cm**2 + y_cm**2)
    # r_smooth = gaussian_filter1d(radius_arr, 3)
    return theta_arr, radius_arr#, r_smooth

def find_width_boundaries(x_data, y_data, dim_data, den_data, x_orbit, y_orbit, theta_cm, radius_cm, threshold = 0.33):
    # find the transverse plane 
    condition_coord, x_onplane, _ = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_cm, radius_cm, coord = True)
    x_plane = x_data[condition_coord]
    y_plane = y_data[condition_coord]
    den_plane = den_data[condition_coord]

    # restrict to not keep points too far away
    condition_x = np.abs(x_onplane) < 20
    x_onplane = x_onplane[condition_x]
    x_plane = x_plane[condition_x]
    y_plane = y_plane[condition_x]
    den_plane = den_plane[condition_x]

    # sort 
    x_onplane = list(x_onplane)
    x_onplane_sorted = sorted(x_onplane)
    x_plane_sorted = sort_list(x_plane, x_onplane)
    y_plane_sorted = sort_list(y_plane, x_onplane)
    den_plane_sorted = sort_list(den_plane, x_onplane)
    den_median_plane_sorted = median_array(den_plane_sorted)

    # find the cm of the plane
    idx_cm = np.argmax(den_plane_sorted)    
    x_cm = x_plane_sorted[idx_cm]
    y_cm = y_plane_sorted[idx_cm]
    den_cm = den_plane_sorted[idx_cm]
    
    # walk before and after the cm till you find a density 3 times smaller
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step > 0:
        idx_step -= 1
        den_tube = den_median_plane_sorted[idx_step] #den_plane_sorted[idx_step]
    idx_before = idx_step+1
    x_low = x_plane_sorted[idx_before]
    y_low = y_plane_sorted[idx_before]
    den_low = den_tube

    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step < len(den_plane_sorted) - 1:
        idx_step += 1
        den_tube  = den_median_plane_sorted[idx_step] #den_plane_sorted[idx_step]
    idx_after = idx_step-1
    x_high = x_plane_sorted[idx_after]
    y_high = y_plane_sorted[idx_after]
    den_high = den_tube
    width = x_onplane_sorted[idx_after] - x_onplane_sorted[idx_before]
    ncells = idx_after - idx_before + 1 # +1 because ideces start from 0

    # plot to check
    # vminrho = -8
    # vmaxrho = -7.2
    # x_chosen, y_chosen = from_cylindric(theta_cm, radius_cm)
    # plt.figure(figsize = (12,8))
    # img = plt.scatter(x_mid, y_mid, c = np.log10(den_mid), s = .1, cmap = 'viridis', vmin = vminrho, vmax = vmaxrho)
    # #cbar = plt.colorbar(img)
    # plt.scatter(0,0,s=40, c= 'k')
    # plt.plot(x_orbit, y_orbit, c = 'r')
    # plt.scatter([x_low, x_high], [y_low, y_high], c = ['b', 'r'], s = 4)
    # plt.scatter(x_plane, y_plane, s = 0.1, c = 'k')
    # plt.scatter(x_chosen, y_chosen, marker = 'x', s = 27, c = 'b')
    # plt.xlim(-200,30)
    # plt.ylim(-60,70)
    # #plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
    # plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
    # plt.show()
    return x_cm, y_cm, den_cm, x_low, y_low, den_low, x_high, y_high, den_high, width, ncells

def find_height_boundaries(x_data, y_data, z_data, dim_data, den_data, x_orbit, y_orbit, theta_cm, radius_cm, den_cm, threshold = 0.33):
    # find the transverse plane 
    condition_coord, x_onplane, x0  = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_cm, radius_cm, coord = True)
    x_plane = x_data[condition_coord]
    z_plane = z_data[condition_coord]
    den_plane = den_data[condition_coord]
    dim_plane = dim_data[condition_coord]

    # restrict to not keep points too far away
    condition_z = np.abs(z_plane) < 10
    # take point on the vertical axis
    condition_x = np.abs(x_onplane-x0) < dim_plane
    condition_vertical = condition_z & condition_x

    x_onplane_vert = x_onplane[condition_vertical]
    z_vert = z_plane[condition_vertical]
    den_vert = den_plane[condition_vertical]

    # sort 
    z_vert = list(z_vert)
    z_vert_sorted = sorted(z_vert)
    x_vert_sorted = sort_list(x_onplane_vert, z_vert)
    den_vert_sorted = sort_list(den_vert, z_vert)
    den_median_vert_sorted = median_array(den_vert_sorted)

    # find the cm of the plane
    idx_cm = np.argmin(np.abs(z_vert_sorted))    
    
    # walk before and after the cm till you find a density 3 times smaller
    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step > 0:
        idx_step -= 1
        den_tube = den_median_vert_sorted[idx_step] #den_vert_sorted[idx_step]
    idx_before = idx_step+1
    x_low = x_vert_sorted[idx_before]
    z_low = z_vert_sorted[idx_before]
    den_low = den_tube

    idx_step = idx_cm
    den_tube = den_cm
    while den_tube > threshold * den_cm and idx_step < len(den_vert_sorted) - 1:
        idx_step += 1
        den_tube  = den_median_vert_sorted[idx_step] #den_vert_sorted[idx_step]
    idx_after = idx_step-1
    x_high = x_vert_sorted[idx_after]
    z_high = z_vert_sorted[idx_after]
    den_high = den_tube
    height = z_high - z_low 
    ncells = idx_after - idx_before + 1 # +1 because ideces start from 0

    # plot to check
    # vmaxrho = 5e-8
    # vminrho = threshold*vmaxrho -1e-8 # -1 to see the points with density lower than threshold
    # plt.figure(figsize = (6,4))
    # plt.scatter(x_onplane_vert, z_vert, s = 8, marker='s', c = 'k', alpha = 0.8)
    # img = plt.scatter(x_onplane, z_plane, c = den_plane, s = 5, cmap = 'viridis', vmin = vminrho, vmax = vmaxrho)
    # cbar = plt.colorbar(img)
    # plt.axhline(z_low, c = 'k', alpha = 0.5)
    # plt.axhline(z_high, c = 'k', alpha = 0.5)
    # plt.scatter(x0, 0, marker = 'x', s = 10, c = 'b')
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    # plt.xlabel(r'T [$R_\odot$]', fontsize = 18)
    # plt.ylabel(r'Z [$R_\odot$]', fontsize = 18)
    # plt.show()
    return x0, den_cm, x_low, z_low, den_low, x_high, z_high, den_high, height, ncells

def find_width_stream(x_data, y_data, dim_data, den_data, theta_params, threshold = 0.33):
    theta_arr, r_orbit = find_maximum(x_data, y_data, dim_data, den_data, theta_params)
    x_orbit, y_orbit = from_cylindric(theta_arr, r_orbit) 
    cm = np.zeros((3,len(theta_arr)))
    upper_tube = np.zeros((3,len(theta_arr)))
    lower_tube = np.zeros((3,len(theta_arr)))
    width = np.zeros(len(theta_arr))
    ncells = np.zeros(len(theta_arr))
    for i,theta in enumerate(theta_arr):
        x_cm, y_cm, den_cm, x_low, y_low, den_low, x_high, y_high, den_high, w, num = \
            find_width_boundaries(x_data, y_data, dim_data, den_data, x_orbit, y_orbit, theta, r_orbit[i], threshold)
        cm[0][i], cm[1][i], cm[2][i] = x_cm, y_cm, den_cm
        lower_tube[0][i], lower_tube[1][i], lower_tube[2][i] = x_low, y_low, den_low
        upper_tube[0][i], upper_tube[1][i], upper_tube[2][i] = x_high, y_high, den_high
        width[i] = w
        ncells[i] = num
    # upper_tube = gaussian_filter1d(upper_tube, 6)
    # lower_tube = gaussian_filter1d(lower_tube, 6)
    return theta_arr, cm, upper_tube, lower_tube, width, ncells

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