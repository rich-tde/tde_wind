import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle 
import Utilities.prelude
from Utilities.operators import make_tree, calc_multiple_grad, calc_div
from Utilities.time_extractor import days_since_distruption

##
# PARAMETERS
##

gamma = 5/3
mach_min = 1.3
save = True
snap = 'final'
folder = 'sedov'
path = f'{folder}/{snap}'
if folder == 'TDE':
    is_tde = False
    m = 6
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)
    delta_cs = 2
else:
    is_tde = False
    apocenter = 1
    delta_cs = 0.02


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

def shock_direction(grad_temp):
    """ Find shock direction according eq.(5) by Schaal14 in the point of coordinates indices idx.
    Parameters
    -----------
    grad_temp: array.
            Gradient of temperature of a cell (vector of 3 components).
    Returns
    -----------
    grad: array.
        Gradient of temperature (vector of 3 components).
    ds: array.
        Shock direction (vector of 3 components).
    """
    magnitude = np.linalg.norm(grad_temp)
    
    # Avoid numerical issues. If grad_temp is small, set ds = [0,0,0]
    if magnitude<1e-3:
        ds = np.zeros(3)
    else:
        ds = - np.divide(grad_temp,magnitude)

    ds = np.array(ds)
    return ds
 
def find_prepost(sim_tree, X, Y, Z, point, ds, delta, direction):
    """ Find the previous/next point along the shock direction.
    Parameters
    -----------
    point: array.
        Starting point.
    ds: array (1x3)
        Shock direction.
    delta: float.
        Step to do to search.
    direction: str.
        Choose if you want to move towards the 'pre' or 'post' shock region.
    Returns
    -----------
    idx: int.
        Tree index of the previous/next point along the shock direction.
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
        k += 0.1

    return idx

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
    idxpost = find_prepost(sim_tree, X, Y, Z, point, ds, delta, direction = 'post') 
    idxpre = find_prepost(sim_tree, X, Y, Z, point, ds, delta, direction = 'pre') 

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
    
def shock_zone(divv, gradT, gradrho, cond3):
    """ Find the shock zone according conditions in Sec. 2.3.2 of Schaal14. 
    In order to test the code, with "check_con" you can decide if checking all or some of the conditions."""

    cond2 = np.dot(gradT, gradrho)

    if np.logical_and(divv<0, np.logical_and(cond2 > 0, cond3 == True)):
        return True
    else:
        return False
    
##
# MAIN
##
    
if __name__ == '__main__':
    if folder == 'TDE':
        _, days = days_since_distruption(f'{path}/snap_{snap}.h5', 'tfb')
    else:
        time, _ = days_since_distruption(f'{path}/snap_{snap}.h5')
    sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde)
    print('Tree built')
    dim_cell = (3*Vol/(4*np.pi))**(1/3)

    shock_dirx = []
    shock_diry = []
    shock_dirz = []
    X_shock = []
    Y_shock = []
    Z_shock = []
    div_shock = []
    T_shock = []
    are_u_shock = np.zeros(len(X), dtype = bool)
    idx_tree = []

    for i in range(len(X)):
        # print(i)
        point = np.array([X[i],Y[i],Z[i]])

        # exclude point with nothing
        if Den[i] < 1e-15:
            are_u_shock[i] = False
            continue
        
        # compute gradient of temperature and density 
        step = 2*dim_cell[i]
        gradients = calc_multiple_grad(sim_tree, X, Y, Z, [Temp, Den], point, step)
        gradT = np.array(gradients[0])
        grad_den = np.array(gradients[1])
    
        ds = shock_direction(gradT)

        # exclude point with nothing
        if np.linalg.norm(ds) == 0:
            are_u_shock[i] = False
            continue
        
        div_vel = calc_div(sim_tree, X, Y, Z, VX, VY, VZ, point, step)

        # exclude points fow which you mess up computing the gradient 
        # (if you improve it, you delete this if)
        if math.isnan(np.linalg.norm(gradT)) or math.isnan(np.linalg.norm(div_vel)):
            are_u_shock[i] = False
            print(f'nan in grad/div of cell {i}. Skip.')
            continue 

        cond3 = condition3(sim_tree, X, Y, Z, Press, Temp, point, ds, mach_min, gamma, step)
        shock = shock_zone(div_vel, gradT, grad_den, cond3)
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
            idx_tree.append(i)

    X_shock = np.array(X_shock)
    Y_shock = np.array(Y_shock)
    Z_shock = np.array(Z_shock)

    # Save data 
    if save == True:
        with open(f'data/{snap}/shockzone_{snap}.txt', 'w') as file:
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
            file.write('# index in the tree \n') 
            file.write(' '.join(map(str, idx_tree)) + '\n')
            file.close()
        with open(f'data/{snap}/shockdir_{snap}.txt', 'w') as fileds:
            fileds.write('# shock x direction \n') 
            fileds.write(' '.join(map(str, shock_dirx)) + '\n')
            fileds.write('# shock y direction \n') 
            fileds.write(' '.join(map(str, shock_diry)) + '\n')
            fileds.write('# shock z direction \n') 
            fileds.write(' '.join(map(str, shock_dirz)) + '\n')
            fileds.close()
        with open(f'data/{snap}/shockbool_{snap}.pkl', 'wb') as filebool:
            pickle.dump(are_u_shock, filebool)
        
    cross_sec = 0

    # Plotting
    plt.figure(figsize=(14,7))
    plt.plot(X_shock[np.abs(Z_shock-cross_sec)<delta_cs]/apocenter, Y_shock[np.abs(Z_shock-cross_sec)<delta_cs]/apocenter, 'ks',  markerfacecolor='none', ms=5, markeredgecolor='k', label = 'shock zone')
    if folder == 'sedov':
        plt.xlabel(r'X [x]', fontsize = 18)
        plt.ylabel(r'Y [y]', fontsize = 18)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.title(r'Shock zone (projection) t= ' + f'{np.round(time,5)}', fontsize = 18)
    else:
        img = plt.scatter(X[::200]/apocenter, Y[::200]/apocenter, c = np.log10(Temp[::200]), alpha = 0.5)#, vmin = 2, vmax = 8)
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\log_{10}$Temperature', fontsize = 18)
        plt.xlabel(r'X [x/$R_a$]', fontsize = 18)
        plt.ylabel(r'Y [y/$R_a$]', fontsize = 18)
        plt.xlim(-1,0.05)
        plt.ylim(-0.3, 0.2)
        plt.title(r'Shock zone (projection) t/t$_{fb}$= ' + f'{np.round(days,5)}', fontsize = 18)
    plt.grid()
    if save:
        plt.savefig(f'Figs/{snap}/shockzone_{snap}.png')

    plt.show()