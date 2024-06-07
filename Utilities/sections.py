import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from Utilities.operators import from_cylindric

def make_slices(all_data, condition):
    """
    For each array of data, it slices it according to a condition.
    """
    return [data[condition] for data in all_data]

def rotation(x_data, y_data, m):
    theta = np.arctan2(m,1)
    theta = -theta # we need the - in front of theta to be consistent with the function to_cylindric, where we change the orientation of the angle
    matrix_rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) 
    x_onplane, y_onplane = np.dot(matrix_rotation, np.array([x_data, y_data])) 
    return x_onplane, y_onplane

def tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coeff_ang = False):
    # Search where you are in the orbit to then find the previous point
    x_chosen, y_chosen = from_cylindric(theta_chosen, radius_chosen)
    idx_chosen = np.argmin(np.abs(y_chosen - y_orbit)) # orizontal parabola --> unique y values
    if idx_chosen == 0:
        idx_chosen += 1
    elif idx_chosen == len(y_orbit)-1:
        idx_chosen -= 1
    if np.abs(theta_chosen)<1e-4: #i.e. x=Rchosen, which should be pericenter --> draw vertical line
        m = 1e10 #infty
        condition_coord = np.abs(x_data-radius_chosen) < dim_data 
    else:
        m = (y_orbit[idx_chosen+1] - y_orbit[idx_chosen-1]) / (x_orbit[idx_chosen+1] - x_orbit[idx_chosen-1])
        ideal_y_value = y_chosen + m * (x_data-x_chosen)
        condition_coord = np.abs(y_data - ideal_y_value) < dim_data
    if coeff_ang:
        return condition_coord, m
    else:
        return condition_coord

def transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coord = False):
    # Find the transverse plane to the orbit with respect to the tangent plane at the chosen point
    x_chosen, y_chosen = from_cylindric(theta_chosen, radius_chosen)
    _, m = tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coeff_ang = True)
    inv_m = -1/m
    ideal_y_value = y_chosen + inv_m * (x_data-x_chosen)
    condition_coord = np.abs(y_data - ideal_y_value) < dim_data
    if np.abs(theta_chosen) < 1e-4: # if theta is close to 0
        condition_coord = np.logical_and(condition_coord, x_data >= 0)
    elif np.logical_and(theta_chosen > 0, theta_chosen != np.pi): # if theta is in the third or fourth quadrant
        condition_coord = np.logical_and(condition_coord, y_data < 0)
    else:
        condition_coord = np.logical_and(condition_coord, y_data > 0)
    if coord:
        x_trasl = x_data[condition_coord] - x_chosen
        y_trasl = y_data[condition_coord] - y_chosen
        x_onplane = rotation(x_trasl, y_trasl, inv_m)[0]
        # just a check when you'll plot to be sure that x_chosen is at the origin, i.e. x0 = 0
        idx = np.argmin(np.abs(x_data[condition_coord]-x_chosen)) #should be the index of x_chosen
        x0 = x_onplane[idx]
        return condition_coord, x_onplane, x0
    else:
        return condition_coord

def radial_plane(x_data, y_data, dim_data, theta_chosen):
    if np.abs(theta_chosen - np.pi/2) < 0.1: # if theta is close to pi/2
        condition_coord = np.abs(x_data) < dim_data  # vertical line
        condition_coord = np.logical_and(condition_coord, y_data < 0) # only the lower part
    elif np.abs(theta_chosen + np.pi/2) < 0.1: # if theta is close to -pi/2
        condition_coord = np.abs(x_data) < dim_data  # vertical line
        condition_coord = np.logical_and(condition_coord, y_data > 0) # only the upper part
    else:
        # we need the - in front of theta to be consistent with the function to_cylindric, where we change the orientation of the angle
        m = np.tan(-theta_chosen)
        condition_coord = np.abs(y_data - m * x_data) < dim_data 
        if theta_chosen == 0:
            condition_coord = np.logical_and(condition_coord, x_data >= 0)
        elif np.abs(theta_chosen) == np.pi:
            condition_coord = np.logical_and(condition_coord, x_data < 0)
        elif np.logical_and(theta_chosen > 0, theta_chosen != np.pi): # if theta is in the third or fourth quadrant
            condition_coord = np.logical_and(condition_coord, y_data < 0)
        else:
            condition_coord = np.logical_and(condition_coord, y_data > 0)
    return condition_coord
