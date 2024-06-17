import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
from Utilities.operators import from_cylindric

def make_slices(all_data, condition):
    """
    For each array of data, it slices it according to a condition.
    """
    return [data[condition] for data in all_data]

def rotation(x_data, y_data, param, angle = False):
    """ Rotate your data so that the line y = m*x is horizontal.
    We have to take into account the fact that the angle given by arctan2 goes from -pi to pi, but is counterclockwise, while our orbits are clockwise."""
    if angle:
        theta = -param # you have the angles, but it's clockwise (from the function to_cylindric)
    else:
        theta = np.arctan2(param,1)
        theta = -theta # to be consistent with our orbit (function to_cylindric)
    x_onplane = x_data * np.cos(theta) - y_data * np.sin(theta)
    y_onplane = x_data * np.sin(theta) + y_data * np.cos(theta)
    return x_onplane, y_onplane

def tangent_vector(x_orbit, y_orbit, theta_chosen, radius_chosen):
    # find the components of the tangent vector to the orbit at the chosen point
    _, y_chosen = from_cylindric(theta_chosen, radius_chosen)
    idx_chosen = np.argmin(np.abs(y_chosen - y_orbit)) # orizontal parabola --> unique y values
    if idx_chosen == 0:
        idx_chosen += 1
    elif idx_chosen == len(y_orbit)-1:
        idx_chosen -= 1
    x_coord_tg = x_orbit[idx_chosen+1] - x_orbit[idx_chosen-1]
    y_coord_tg = y_orbit[idx_chosen+1] - y_orbit[idx_chosen-1]
    vec_tg = np.array([x_coord_tg, y_coord_tg])
    vers_tg = vec_tg / np.linalg.norm(vec_tg)
    return vers_tg
        
def transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coord = False):
    """
    Find the transverse plane to the orbit at the chosen point.
    If coord is True, it returns the coordinates of the data in the plane with respect to the new coordinates system,
    i.e. the one that have versor (tg_vect, Zhat X tg_vect, Zhat).
    """
    vers_tg = tangent_vector(x_orbit, y_orbit, theta_chosen, radius_chosen)
    # Find the points orthogonal to the tangent vector at the chosen point
    x_chosen, y_chosen = from_cylindric(theta_chosen, radius_chosen)
    x_data_trasl = x_data - x_chosen
    y_data_trasl = y_data - y_chosen
    data_trasl = np.transpose([x_data_trasl, y_data_trasl])
    #data_trasl /= np.linalg.norm(data_trasl)
    dot_product = np.dot(data_trasl, vers_tg) #that's the projection of the data on the tangent vector
    condition_tra = np.abs(dot_product) < dim_data 

    if theta_chosen < 0:
        conditon_y = y_data>0
    else:
        conditon_y = y_data<0
    condition_coord = np.logical_and(condition_tra, conditon_y)
    if coord:
        #Find the versors in 3D
        zhat = np.array([0,0,1])
        yRhat = np.array([vers_tg[0], vers_tg[1],0]) # points in the direction of the orbit
        xRhat = np.cross(zhat, vers_tg) # points outwards
        xRhat /= np.linalg.norm(xRhat)
        vers_norm = np.array([xRhat[0], xRhat[1]])
        #Find the coordinates of the data in the new system
        x_onplane = np.dot(data_trasl[condition_coord], vers_norm)
        # y_onplane = np.dot(data_trasl[condition_coord], vers_tg)
        x0 = x_onplane[np.argmin(np.abs(x_onplane))]
        return condition_coord, x_onplane, x0
    else:
        return condition_coord

def tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coeff_ang = False):
    # Search where you are in the orbit to then find the previous point
    x_chosen, y_chosen = from_cylindric(theta_chosen, radius_chosen)
    idx_chosen = np.argmin(np.abs(y_chosen - y_orbit)) # orizontal parabola --> unique y values
    if idx_chosen == 0:
        idx_chosen += 1
    elif idx_chosen == len(y_orbit)-1:
        idx_chosen -= 1
    if np.abs(theta_chosen)<0.1: #i.e. x=Rchosen, which should be pericenter --> draw vertical line
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

def busy_transverse_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coord = False):
    # Find the transverse plane to the orbit with respect to the tangent plane at the chosen point
    x_chosen, y_chosen = from_cylindric(theta_chosen, radius_chosen)
    _, m = tangent_plane(x_data, y_data, dim_data, x_orbit, y_orbit, theta_chosen, radius_chosen, coeff_ang = True)
    inv_m = -1/m
    ideal_y_value = y_chosen + inv_m * (x_data-x_chosen)
    condition_coord = np.abs(y_data - ideal_y_value) < dim_data
    if np.abs(m) < 0.1: # if m is close to 0, take a vertical line
        print('m is close to 0, theta: ', theta_chosen)
        condition_coord = np.abs(x_data-x_chosen) < dim_data #condition_coord = np.logical_and(condition_coord, y_data >= 0)
        if theta_chosen < 0:
            condition_coord = np.logical_and(condition_coord, y_data > 0)
        elif theta_chosen >0:
            condition_coord = np.logical_and(condition_coord, y_data < 0)
    elif np.logical_and(theta_chosen > 0, theta_chosen != np.pi): # if theta is in the third or fourth quadrant
        condition_coord = np.logical_and(condition_coord, y_data < 0)
    elif theta_chosen<0:
        condition_coord = np.logical_and(condition_coord, y_data > 0)
    if coord:
        # early = False
        x_trasl = x_data[condition_coord] - x_chosen
        y_trasl = y_data[condition_coord] - y_chosen
        idx = np.argmin(np.abs(x_data[condition_coord] - x_chosen)) #should be the index of x_chosen
        x_trasl[idx] = 0 #if u can avoid it, better
        y_trasl[idx] = 0 #if u can avoid it, better
        x_onplane = rotation(x_trasl, y_trasl, inv_m)[0]
        # just a check when you'll plot to be sure that x_chosen is at the origin, i.e. x0 = 0
        x0 = x_onplane[idx]
        return condition_coord, x_onplane, x0
    else:
        return condition_coord


def radial_plane(x_data, y_data, dim_data, theta_chosen):
    if np.abs(theta_chosen - np.pi/2) < 1e-4: # if theta is close to pi/2
        condition_coord = np.abs(x_data) < dim_data  # vertical line
        condition_coord = np.logical_and(condition_coord, y_data < 0) # only the lower part
    elif np.abs(theta_chosen + np.pi/2) < 1e-4: # if theta is close to -pi/2
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
