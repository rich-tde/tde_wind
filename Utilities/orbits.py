""" 
Find different kind of orbits for TDEs. 
Measure the width and the height of the stream.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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

def find_maximum(x_mid, y_mid, dim_mid, den_mid, theta_params, Rt):
    # find the orbit given by maxima density points
    # It's better to give theta_params than theta_arr, so that you can adjust the step
    theta_arr = np.arange(*theta_params)
    step = theta_params[2]
    if step < 0.1:
        step = 0.1
    radius_arr = np.zeros(len(theta_arr))
    for i,theta_cm in enumerate(theta_arr):
        condition_Rplane = radial_plane(x_mid, y_mid, dim_mid, theta_cm)
        condition_distance = np.sqrt(x_mid**2 + y_mid**2) > Rt # to avoid the center
        condition_Rplane = np.logical_and(condition_Rplane, condition_distance)
        idx_cm = np.argmax(den_mid[condition_Rplane])
        x_cm = x_mid[condition_Rplane][idx_cm]
        y_cm = y_mid[condition_Rplane][idx_cm]
        radius_arr[i] = np.sqrt(x_cm**2 + y_cm**2)
    # r_smooth = gaussian_filter1d(radius_arr, 3)
    return theta_arr, radius_arr#, r_smooth

# def find_width_boundaries(x_data, y_data, dim_data, den_data, x_orbit, y_orbit, theta_cm, radius_cm, threshold = 0.33):
#     # find the transverse plane 
#     condition_coord, x_onplane, _ = previous_transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_cm, radius_cm, coord = True)
#     x_plane = x_data[condition_coord]
#     y_plane = y_data[condition_coord]
#     den_plane = den_data[condition_coord]

#     # restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
#     condition_x = np.abs(x_onplane) < 20
#     x_onplane = x_onplane[condition_x]
#     x_plane = x_plane[condition_x]
#     y_plane = y_plane[condition_x]
#     den_plane = den_plane[condition_x]

#     # sort 
#     x_onplane = list(x_onplane)
#     x_onplane_sorted = sorted(x_onplane)
#     x_plane_sorted = sort_list(x_plane, x_onplane)
#     y_plane_sorted = sort_list(y_plane, x_onplane)
#     den_plane_sorted = sort_list(den_plane, x_onplane)
#     den_median_plane_sorted = median_array(den_plane_sorted)

#     # find the cm of the plane
#     idx_cm = np.argmax(den_plane_sorted)    
#     x_cm = x_plane_sorted[idx_cm]
#     y_cm = y_plane_sorted[idx_cm]
#     den_cm = den_plane_sorted[idx_cm]
#     print('den_cm', den_cm)
    
#     # walk before and after the cm till you find a density 3 times smaller
#     idx_step = idx_cm
#     den_tube = den_cm
#     while den_tube > threshold * den_cm and idx_step > 0:
#         idx_step -= 1
#         den_tube = den_median_plane_sorted[idx_step] #den_plane_sorted[idx_step]
#     idx_before = idx_step+1
#     x_low = x_plane_sorted[idx_before]
#     y_low = y_plane_sorted[idx_before]
#     den_low = den_tube

#     idx_step = idx_cm
#     den_tube = den_cm
#     while den_tube > threshold * den_cm and idx_step < len(den_plane_sorted) - 1:
#         idx_step += 1
#         den_tube  = den_median_plane_sorted[idx_step] #den_plane_sorted[idx_step]
#     idx_after = idx_step-1
#     x_high = x_plane_sorted[idx_after]
#     y_high = y_plane_sorted[idx_after]
#     den_high = den_tube
#     width = x_onplane_sorted[idx_after] - x_onplane_sorted[idx_before]
#     ncells = idx_after - idx_before + 1 # +1 because ideces start from 0

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
    # plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
    # plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
    # plt.show()
    # return x_cm, y_cm, den_cm, x_low, y_low, den_low, x_high, y_high, den_high, width, ncells

# def find_height_boundaries(x_data, y_data, z_data, dim_data, den_data, x_orbit, y_orbit, theta_cm, radius_cm, den_cm, threshold = 0.33):
#     # find the transverse plane 
#     condition_coord, x_onplane, x0  = previous_transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_cm, radius_cm, coord = True)
#     x_plane = x_data[condition_coord]
#     z_plane = z_data[condition_coord]
#     den_plane = den_data[condition_coord]
#     dim_plane = dim_data[condition_coord]

#     # restrict to not keep points too far away
#     condition_z = np.abs(z_plane) < 10
#     # take point on the vertical axis
#     condition_x = np.abs(x_onplane-x0) < dim_plane
#     condition_vertical = condition_z & condition_x

#     x_onplane_vert = x_onplane[condition_vertical]
#     z_vert = z_plane[condition_vertical]
#     den_vert = den_plane[condition_vertical]

#     # sort 
#     z_vert = list(z_vert)
#     z_vert_sorted = sorted(z_vert)
#     x_vert_sorted = sort_list(x_onplane_vert, z_vert)
#     den_vert_sorted = sort_list(den_vert, z_vert)
#     den_median_vert_sorted = median_array(den_vert_sorted)

#     # find the cm of the plane
#     idx_cm = np.argmin(np.abs(z_vert_sorted))    
    
#     # walk before and after the cm till you find a density 3 times smaller
#     idx_step = idx_cm
#     den_tube = den_cm
#     while den_tube > threshold * den_cm and idx_step > 0:
#         idx_step -= 1
#         den_tube = den_median_vert_sorted[idx_step] #den_vert_sorted[idx_step]
#     idx_before = idx_step+1
#     x_low = x_vert_sorted[idx_before]
#     z_low = z_vert_sorted[idx_before]
#     den_low = den_tube

#     idx_step = idx_cm
#     den_tube = den_cm
#     while den_tube > threshold * den_cm and idx_step < len(den_vert_sorted) - 1:
#         idx_step += 1
#         den_tube  = den_median_vert_sorted[idx_step] #den_vert_sorted[idx_step]
#     idx_after = idx_step-1
#     x_high = x_vert_sorted[idx_after]
#     z_high = z_vert_sorted[idx_after]
#     den_high = den_tube
#     height = z_high - z_low 
#     ncells = idx_after - idx_before + 1 # +1 because ideces start from 0

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
    # return x0, den_cm, x_low, z_low, den_low, x_high, z_high, den_high, height, ncells

def find_stream_boundaries(x_data, y_data, z_data, dim_data, den_data, x_orbit, y_orbit, theta_cm, radius_cm, threshold = 0.33):
    # find the transverse plane 
    condition_T, x_Tplane, x0 = transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_cm, radius_cm, coord = True)
    x_plane, y_plane, z_plane, dim_plane, den_plane = make_slices([x_data, y_data, z_data, dim_data, den_data], condition_T)
    # find transverse line (on midplane)
    midplane = np.abs(z_plane) < dim_plane
    x_mid, x_T_mid, y_mid, den_mid = make_slices([x_plane, x_Tplane, y_plane, den_plane], midplane)

    # restrict to not keep points too far away. Important for theta=0 or you take the stream at apocenter
    condition_x = np.abs(x_T_mid) < 20
    x_mid = x_mid[condition_x]
    x_T_mid = x_T_mid[condition_x]
    y_mid = y_mid[condition_x]
    den_mid = den_mid[condition_x]

    # sort 
    x_T_mid = list(x_T_mid)
    x_Tmid_sorted = sorted(x_T_mid)
    x_mid_sorted = sort_list(x_mid, x_T_mid)
    y_mid_sorted = sort_list(y_mid, x_T_mid)
    den_mid_sorted = sort_list(den_mid, x_T_mid)
    den_median_mid_sorted = median_array(den_mid_sorted)

    # find the cm of the plane
    idx_cm = np.argmax(den_mid_sorted) #den_median_mid_sorted
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
    ncells_w = idx_after - idx_before + 1 # +1 because ideces start from 0

    # Z direction
    # take point on the vertical axis
    # x_Tplane_vert, z_vert, den_vert = make_slices([x_Tplane, z_plane, den_plane], vertical)
    vertical = np.abs(x_Tplane-x0) < dim_plane #or x_plane-x_cm
    z_vert, den_vert = make_slices([z_plane, den_plane], vertical)

    # sort 
    z_vert = list(z_vert)
    z_vert_sorted = sorted(z_vert)
    # x_vert_sorted = sort_list(x_plane_vert, z_vert)
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
    ncells_h = idx_after - idx_before + 1 # +1 because ideces start from 0

    # Wrap. Assumption: z=0 for upper/lower tube for width, xplane=x0~0 for upper/lower tube for height
    cm = np.array([x_cm, y_cm, den_cm])
    lower_tube_w = np.array([x_low, x_T_low, y_low, den_low_w])
    upper_tube_w = np.array([x_up, x_T_up, y_up, den_up_w])
    lower_tube_h = np.array([z_low, den_low_h])
    upper_tube_h = np.array([z_up, den_up_h])
    w_params = np.array([width, ncells_w])
    h_params = np.array([height, ncells_h])

    return cm, lower_tube_w, upper_tube_w, lower_tube_h, upper_tube_h, w_params, h_params

def follow_the_stream(x_data, y_data, z_data, dim_data, den_data, theta_params, threshold = 0.33):
    theta_arr, r_orbit = find_maximum(x_data, y_data, dim_data, den_data, theta_params)
    x_orbit, y_orbit = from_cylindric(theta_arr, r_orbit) 
    cm = []
    lower_tube_w = []
    upper_tube_w = []
    lower_tube_h = []
    upper_tube_h = []
    w_params = []
    h_params = []
    for i,theta in enumerate(theta_arr):
        cm_i, lower_tube_w_i, upper_tube_w_i, lower_tube_h_i, upper_tube_h_i, w_params_i, h_params_i = \
            find_stream_boundaries(x_data, y_data, z_data, dim_data, den_data, x_orbit, y_orbit, theta, r_orbit[i], threshold)
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
    return theta_arr, cm, lower_tube_w, upper_tube_w, lower_tube_h, upper_tube_h, w_params, h_params

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