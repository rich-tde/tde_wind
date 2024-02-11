import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt

from grid_maker import make_grid
from Utilities.operators import calc_div, calc_grad, zero_interpolator, tree_interpolator
import Utilities.prelude


##
# PARAMETERS
##
gamma = 5/3
mach_min = 1.3
num = 45
cross_section = True

##
# FUNCTIONS
##
def temperature_bump(mach, gamma):
    """ T_post/ T_pre shock """
    Tbump =  (mach**2 * (gamma-1) + 2) * (2 * gamma * mach**2 - (gamma-1)) / (mach**2 * (gamma+1)**2)
    return Tbump

def pressure_bump(mach, gamma):
    """ P_post/ P_pre shock"""
    Pbump = (2 * gamma * mach**2 - (gamma-1)) / (gamma+1)
    return Pbump

def shock_direction(x_array, y_array, z_array, Tgrid, idx):
    """ Find shock direction according eq.(5) by Schaal14 in the point of coordinates indices idx"""
    gradT = calc_grad(x_array, y_array, z_array, Tgrid, idx, kind_info = 'idx')
    magnitude = np.linalg.norm(gradT)
    ds = - np.array(gradT) / magnitude 
    return ds

def condition3(x_array, y_array, z_array, Tgrid, Pgrid, idx, ds, deltax, mach_min, gamma):
    i = idx[0]
    j = idx[1]
    k = idx[2]
    xyz = [x_array[i], y_array[j], z_array[k]]
    # You suppose that the cells have the same dimension is x,y,z
    post = xyz + ds*deltax
    pre = xyz - ds*deltax

    Tpost = zero_interpolator(x_array, y_array, z_array, Tgrid, post)
    Tpre = zero_interpolator(x_array, y_array, z_array, Tgrid, pre)
    Ppost = zero_interpolator(x_array, y_array, z_array, Pgrid, post)
    Ppre = zero_interpolator(x_array, y_array, z_array, Pgrid, pre)

    delta_logT = np.log(Tpost)-np.log(Tpre)
    Tjump = temperature_bump(mach_min, gamma)
    Tjump = np.log(Tjump)
    ratioT = - delta_logT / Tjump # we take the - because we are in the lab (NOT shock) frame

    delta_logP = np.log(Ppost)-np.log(Ppre)
    Pjump = pressure_bump(mach_min, gamma)
    Pjump = np.log(Pjump)
    ratioP = - delta_logP / Pjump # we take the - because we are in the lab (NOT shock) frame
    
    if np.logical_and(ratioT >= 1, ratioP >= 1): 
        return True
    else:
        return False
    
def shock_zone(x_array, y_array, z_array, Vx_grid, Vy_grid, Vz_grid, T_grid, den_grid, idx, cond3):
    """ Find the shock zone according conditions in Sec. 2.3.2 of Schaal14. """
    cond1 = calc_div(x_array, y_array, z_array, Vx_grid, Vy_grid, Vz_grid, idx, 'idx')
    gradT = calc_grad(x_array, y_array, z_array, T_grid, idx, 'idx')
    gradP = calc_grad(x_array, y_array, z_array, den_grid, idx, 'idx')

    cond2 = np.dot(gradT, gradP)

    if np.logical_and(cond1<0, np.logical_and(cond2 > 0, cond3 == True)):
        return cond1
    else:
        return False

if __name__ == '__main__':
    # make the grid
    _, gridded_den, gridded_T, gridded_P, gridded_Vx, gridded_Vy, gridded_Vz, gridded_V, gridded_Rcell, x_radii, y_radii, z_radii = make_grid(num)

    # check the dimension of simulations cells: it has to be smaller than the one of your grid
    Rcell_min = np.min(gridded_Rcell)
    Rcell_max = np.max(gridded_Rcell)
    dx = (x_radii[-1]-x_radii[0])/num
    print(f'our grid: {np.round(dx,3)}, R_cell: {np.round(Rcell_min,3)}-{np.round(Rcell_max,3)}') #to check is is higher than min(Rcell)

    middle_zidx = int(len(z_radii)/2)
    shock_dirx = []
    shock_diry = []
    shock_dirz = []
    div_shock = []
    X_shock = []
    Y_shock = []
    Z_shock = []

    for i in range(1, len(x_radii)-1):
        for j in range(1,len(y_radii)-1):
            for k in range(middle_zidx, middle_zidx+1):#len(1, z_radii)-1):
                # we don't consider the first and last point of the tree
                idx = [i,j,k]
                ds = shock_direction(x_radii, y_radii, z_radii, gridded_T, idx)
                cond3 = condition3(x_radii, y_radii, z_radii, gridded_T, gridded_P, idx, ds, dx, mach_min, gamma) 
                shock = shock_zone(x_radii, y_radii, z_radii, gridded_Vx, gridded_Vy, gridded_Vz, gridded_T, gridded_den, idx, cond3)

                if shock != False:
                    X_shock.append(x_radii[i])
                    Y_shock.append(y_radii[j])
                    Z_shock.append(z_radii[k])
                    shock_dirx.append(ds[0])
                    shock_diry.append(ds[1])
                    shock_dirz.append(ds[2])
                    div_shock.append(shock)


    if cross_section == True:
        with open(f'shockzone_num{num}.txt', 'a') as file:
            file.write(f'# Coordinates of the points in the shock zone at z = {z_radii[middle_zidx]}, num = {num}, mach_min = {mach_min} \n#X \n') 
            file.write(' '.join(map(str, X_shock)) + '\n')
            file.write('# Y \n') 
            file.write(' '.join(map(str, Y_shock)) + '\n')
            file.write('# Z \n') 
            file.write(' '.join(map(str, Z_shock)) + '\n')
            file.write('# div v \n') 
            file.write(' '.join(map(str, div_shock)) + '\n')
            file.close()

        with open(f'shockdir_num{num}.txt', 'a') as fileds:
            fileds.write('# shock  x direction \n') 
            fileds.write(' '.join(map(str, shock_dirx)) + '\n')
            fileds.write('# shock  y direction \n') 
            fileds.write(' '.join(map(str, shock_diry)) + '\n')
            fileds.write('# shock z direction \n') 
            fileds.write(' '.join(map(str, shock_dirz)) + '\n')
            fileds.close()
    
    plt.figure(figsize = (10,10))
    plt.plot(X_shock, Y_shock, 'ks', c = 'k', markerfacecolor='k', ms=4, markeredgecolor='k')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('X', fontsize = 18)
    plt.ylabel('Y', fontsize = 18)
    plt.title(f'z = {np.round(z_radii[middle_zidx],3)}, '+ r'$N_{cell}$'+ f'= {num} in each direction', fontsize = 16)
    # plt.savefig('Figs/shockzone.png')
    plt.show()