# import sys
# sys.path.append('/Users/paolamartire/shocks')
import Utilities.prelude
import pickle 

import numpy as np
import matplotlib.pyplot as plt
import math
from Utilities.operators import make_tree, calc_grad, calc_div

gamma = 5/3
mach_min = 1.3
save = True

def temperature_bump(mach, gamma):
    """ T_post/ T_pre shock """
    Tbump =  (mach**2 * (gamma-1) + 2) * (2 * gamma * mach**2 - (gamma-1)) / (mach**2 * (gamma+1)**2)
    return Tbump

def pressure_bump(mach, gamma):
    """ P_post/ P_pre shock"""
    Pbump = (2 * gamma * mach**2 - (gamma-1)) / (gamma+1)
    return Pbump

def shock_direction(sim_tree, X, Y, Z, Temp, point, delta):
    """ Find shock direction according eq.(5) by Schaal14 in the point of coordinates indices idx.
    Parameters
    -----------
    sim_tree: nDarray.
            Tree where to search the point.
    X, Y, Z, Temp: arrays.
            Coordinates and temperature of the points of the tree.
    point: array.
            Starting point.
    delta: float.
            Step between 2 neighbours.
    Returns
    -----------
    ds: array.
        Shock direction (vector of 3 components).
    """
    grad = calc_grad(sim_tree, X, Y, Z, Temp, point, delta)
    magnitude = np.linalg.norm(grad)
    
    if np.logical_and(np.abs(grad[0])<0.1, np.logical_and(np.abs(grad[1])<0.1, np.abs(grad[2])<0.1)):
        ds = [0, 0, 0]
    else:
        ds = - np.divide(grad,magnitude)
    return ds

def condition3(sim_tree, Press, Temp, point, ds, mach_min, gamma, delta):
    """ Last condition fot shock zone by Schaal14 .
    Parameters
    -----------
    sim_tree: nDarray.
            Tree where to search the point.
    X, Y, Z, Press, Temp: arrays.
            Coordinates, pressure and temperature of the points of the tree.
    point: array.
            Starting point.
    ds: array (1x3)
        Shock direction.
    mach_min, gamma: floats.
                    Minimum mach number, adiabatic index.
    delta: float.
            Step between 2 neighbours.
    Returns
    -----------
    ds: bool.
        If condition is satisfied or not.
    """
    # Find and query the point in the pre/post shock region.
    post_shock = point - np.multiply(delta, ds)
    pre_shock = point + np.multiply(delta, ds)

    _, idxpost  = sim_tree.query(post_shock)
    Tpost = Temp[idxpost]
    Ppost = Press[idxpost]

    _, idxpre  = sim_tree.query(pre_shock)
    Tpre = Temp[idxpre]
    Ppre = Press[idxpre]

    # Last condition fot shock zone by Schaal14
    delta_logT = np.log(Tpost) - np.log(Tpre)
    Tjump = temperature_bump(mach_min, gamma)
    Tjump = np.log(Tjump)
    ratioT = delta_logT / Tjump 

    delta_logP = np.log(Ppost)-np.log(Ppre)
    Pjump = pressure_bump(mach_min, gamma)
    Pjump = np.log(Pjump)
    ratioP = delta_logP / Pjump 
    
    if np.logical_and(ratioT >= 1, ratioP >= 1): 
        return True
    else:
        return False
    
def shock_zone(sim_tree, X, Y, Z, Den, Temp, Vx, Vy, Vz, point, delta, cond3, check_cond = '3'):
    """ Find the shock zone according conditions in Sec. 2.3.2 of Schaal14. 
    In order to test the code, with "check_con" you can decide if checking all or some of the conditions."""
    cond1 = calc_div(sim_tree, X, Y, Z, Vx, Vy, Vz, point, delta)
    gradT = calc_grad(sim_tree, X, Y, Z, Temp, point, delta)
    gradden = calc_grad(sim_tree, X, Y, Z, Den, point, delta)

    if check_cond == '1' or check_cond == '2':
        cond3 = True #so you don't check it
        if check_cond == '2':
            cond2 = np.dot(gradT, gradden)
        else:
            cond2 = 10
    else:
        cond2 = np.dot(gradT, gradden)

    if np.logical_and(cond1<0, np.logical_and(cond2 > 0, cond3 == True)):
        return cond1
    else:
        return False