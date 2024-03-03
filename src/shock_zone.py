import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle 
import Utilities.prelude
from Utilities.operators import make_tree, calc_grad, calc_div

##
# PARAMETERS
##

gamma_chosen = 5/3
mach_min = 1.3
save = True

##
# FUNCTIONS
##

def temperature_bump(mach, gamma):
    """ T_post/ T_pre shock according to RH conditions."""
    Tbump =  (mach**2 * (gamma-1) + 2) * (2 * gamma * mach**2 - (gamma-1)) / (mach**2 * (gamma+1)**2)
    return Tbump

def pressure_bump(mach, gamma):
    """ P_post/ P_pre shock according to RH conditions."""
    Pbump = (2 * gamma * mach**2 - (gamma-1)) / (gamma+1)
    return Pbump

def shock_direction(sim_tree, X, Y, Z, Temp, point, delta):
    """ Find shock direction according eq.(5) by Schaal14 in the point of coordinates indices idx.
    Parameters
    -----------
    sim_tree: tree.
            Simulation points.
    X, Y, Z, Temp: arrays.
            Coordinates and temperature of the points of the tree.
    point: array.
            Starting point.
    delta: float.
            Step between 2 neighbours.
    Returns
    -----------
    grad: array.
        Gradient of temperature(vector of 3 components).
    ds: array.
        Shock direction (vector of 3 components).
    """
    grad = calc_grad(sim_tree, X, Y, Z, Temp, point, delta)
    magnitude = np.linalg.norm(grad)
    
    if np.logical_and(np.abs(grad[0])<1, np.logical_and(np.abs(grad[1])<1, np.abs(grad[2])<1)):
        ds = np.zeros(3)
    else:
        ds = - np.divide(grad,magnitude)
    return grad, ds

def find_prepost(sim_tree, X, Y, Z, point, ds, delta, direction):
    """
    Parameters
    -----------
    point: array.
        Starting point.
    ds: array (1x3)
        Shock direction.
    delta: float.
        Step to do to search.
    direction: str.
        Choose pre or post shock.
    Returns
    -----------
    idx: int.
        Tree index of the pre/post point.
    new_point: array.
        Pre/post shock point
    """
    if direction == 'post':
        delta = - delta

    k = 1
    # check that you are not taking the same point as the one given
    distance = 0
    while distance == 0:
        new_point = point + k * delta * ds 
        _, idx  = sim_tree.query(new_point)
        check_point = np.array([X[idx], Y[idx], Z[idx]])
        distance = math.dist(point, check_point)
        k += 0.2
    idx = int(idx)
    
    return idx, new_point

def condition3(sim_tree, X, Y, Z, Press, Temp, point, ds, mach_min, gamma, delta):
    """ Last condition fot shock zone by Schaal14 .
    Parameters
    -----------
    sim_tree: tree.
        Simulation points.
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
    bool.
        If condition is satisfied or not.
    """
    # Find (the index in the tree of) the point in the pre/post shock region.
    idxpost, _ = find_prepost(sim_tree, X, Y, Z, point, ds, delta, direction = 'post') 
    idxpre, _ = find_prepost(sim_tree, X, Y, Z, point, ds, delta, direction = 'pre') 

    # Store data from the tree
    Tpost = Temp[idxpost]
    Ppost = Press[idxpost]
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
    
def shock_zone(idx, divv, gradT, gradrho, cond3, check_cond = '3'):
    """ Find the shock zone according conditions in Sec. 2.3.2 of Schaal14. 
    In order to test the code, with "check_con" you can decide if checking all or some of the conditions."""
    cond2 = np.dot(gradT, gradrho)
    if np.logical_and(divv<0, np.logical_and(cond2 > 0, cond3 == True)):
        return True
    else:
        return False

if __name__ == '__main__':
    sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, P, Temp = make_tree('data_sim')

    # Step for the gradient in the shock direction
    dim_cell = (3*Vol/(4*np.pi))**(1/3)
    step_grad = 2 * np.mean(dim_cell)

    # Step for the shock zone/surface
    step_zone = 2 * np.min(dim_cell)

    # Find cells in the shock zone. Store the data 
    are_u_shock = np.zeros(len(X), dtype = bool)
    shock_dirx = []
    shock_diry = []
    shock_dirz = []
    X_shock = []
    Y_shock = []
    Z_shock = []
    div_shock = []
    T_shock = []

    for i in range(len(X)):
        # print(i)
        point = np.array([X[i],Y[i],Z[i]])
        gradT, ds = shock_direction(sim_tree, X, Y, Z, Temp, point, step_grad)

        # Fundamental! (check notes 04/03/24)
        if np.linalg.norm(ds) == 0:
            are_u_shock[i] = False
            continue

        div_vel = calc_div(sim_tree, X, Y, Z, VX, VY, VZ, point, step_grad)
        grad_temp = calc_grad(sim_tree, X, Y, Z, Temp, point, step_grad)
        grad_den = calc_grad(sim_tree, X, Y, Z, Den, point, step_grad)
        
        cond3 = condition3(sim_tree, X, Y, Z, P, Temp, point, ds, mach_min, gamma_chosen, step_zone)

        shock = shock_zone(i, div_vel, grad_temp, grad_den, cond3)
        are_u_shock[i] = shock

        if shock == True:
            X_shock.append(X[i])
            Y_shock.append(Y[i])
            Z_shock.append(Z[i])
            shock_dirx.append(ds[0])
            shock_diry.append(ds[1])
            shock_dirz.append(ds[2])
            div_shock.append(div_vel)
            T_shock.append(Temp[i])
    
    X_shock = np.array(X_shock)
    Y_shock = np.array(Y_shock)
    Z_shock = np.array(Z_shock)

    if save == True:
        with open(f'Tshockzone.txt', 'w') as file:
            file.write(f'# Coordinates of the points in the shock zone, mach_min = {mach_min} \n# X \n') 
            file.write(' '.join(map(str, X_shock)) + '\n')
            file.write('# Y \n') 
            file.write(' '.join(map(str, Y_shock)) + '\n')
            file.write('# Z \n') 
            file.write(' '.join(map(str, Z_shock)) + '\n')
            file.write('# div v \n') 
            file.write(' '.join(map(str, div_shock)) + '\n')
            file.write('# T \n') 
            file.write(' '.join(map(str, T_shock)) + '\n')
            file.close()
        with open(f'Tshockdir.txt', 'w') as fileds:
            fileds.write('# shock x direction \n') 
            fileds.write(' '.join(map(str, shock_dirx)) + '\n')
            fileds.write('# shock y direction \n') 
            fileds.write(' '.join(map(str, shock_diry)) + '\n')
            fileds.write('# shock z direction \n') 
            fileds.write(' '.join(map(str, shock_dirz)) + '\n')
            fileds.close()
        with open('Tshockbool.pkl', 'wb') as filebool:
            pickle.dump(are_u_shock, filebool)
    
    plt.figure(figsize=(10,10))
    plt.plot(X_shock[np.abs(Z_shock)<0.02], Y_shock[np.abs(Z_shock)<0.02], 'ks', c = 'k', markerfacecolor='k', ms=5, markeredgecolor='k')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('X', fontsize = 18)
    plt.ylabel('Y', fontsize = 18)
    plt.legend()
    plt.grid()
    plt.title('Shock zone, z=0', fontsize = 18)
    if save == True: 
        plt.savefig('Figs/T4shockzone.png')
    plt.show()