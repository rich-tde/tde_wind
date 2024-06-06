""" 
Obtain polar coordinates for the orbital plane that go clockwise:
from -pi in -x to second, first, fourth and third (pi) quadrant.
Find orbits for TDEs. 
"""

import numpy as np
from Utilities.sections import radial_plane, transverse_plane
from Utilities.operators import sort_list, median_array

def to_cylindric(x,y):
    radius = np.sqrt(x**2+y**2)
    if np.abs(x.any()) > 1e-5: # numerical version of x.any()!= 0:
        theta_coord = np.arctan2(y,x)
    else:
        if np.abs(y.any()) < 1e-5:
            theta_coord = 0
        elif y.any()>0:
            theta_coord = np.pi/2
        else:
            theta_coord = -np.pi/2
    # theta_coord go from -pi to pi with negative values in the 3rd and 4th quadrant. You want to mirror 
    theta_broadcasted = -theta_coord
    return theta_broadcasted, radius

def from_cylindric(theta, r):
    # we expect theta as from the function to_cylindric, i.e. clockwise. 
    # You have to mirror it to get the angle for the usual polar coordinates.
    theta = -theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

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
    theta0, thetaf, step = theta_params[0], theta_params[1], theta_params[2]
    if step < 0.1:
        step = 0.1
    theta_arr = np.arange(theta0, thetaf, step)
    radius_arr = np.zeros(len(theta_arr))
    for i,theta_cm in enumerate(theta_arr):
        condition_Rplane = radial_plane(x_mid, y_mid, dim_mid, theta_cm)
        idx_cm = np.argmax(den_mid[condition_Rplane])
        x_cm = x_mid[condition_Rplane][idx_cm]
        y_cm = y_mid[condition_Rplane][idx_cm]
        radius_arr[i] = np.sqrt(x_cm**2 + y_cm**2)
    return theta_arr, radius_arr

def find_stream_boundaries(x_mid, y_mid, dim_mid, den_mid, x_orbit, y_orbit, theta_cm, radius_cm, threshold = 0.33):
    # find the normal plane 
    condition_coord, x_onplane, _ = transverse_plane(x_mid, y_mid, dim_mid, x_orbit, y_orbit, theta_cm, radius_cm, coord = True)
    x_plane = x_mid[condition_coord]
    y_plane = y_mid[condition_coord]
    den_plane = den_mid[condition_coord]

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
    return x_cm, y_cm, den_cm, x_low, y_low, den_low, x_high, y_high, den_high, width

def find_width_stream(x_mid, y_mid, dim_mid, den_mid, theta_params):
    theta_arr, r_orbit = find_maximum(x_mid, y_mid, dim_mid, den_mid, theta_params)
    x_orbit, y_orbit = from_cylindric(theta_arr, r_orbit) 
    cm = np.zeros((3,len(theta_arr)))
    upper_tube = np.zeros((3,len(theta_arr)))
    lower_tube = np.zeros((3,len(theta_arr)))
    width = np.zeros(len(theta_arr))
    for i,theta in enumerate(theta_arr):
        x_cm, y_cm, den_cm, x_low, y_low, den_low, x_high, y_high, den_high, w = \
            find_stream_boundaries(x_mid, y_mid, dim_mid, den_mid, x_orbit, y_orbit, theta, r_orbit[i])
        cm[0][i], cm[1][i], cm[2][i] = x_cm, y_cm, den_cm
        lower_tube[0][i], lower_tube[1][i], lower_tube[2][i] = x_low, y_low, den_low
        upper_tube[0][i], upper_tube[1][i], upper_tube[2][i] = x_high, y_high, den_high
        width[i] = w
    return theta_arr, cm, upper_tube, lower_tube, width

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