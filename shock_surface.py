import sys
sys.path.append('/Users/paolamartire/shocks') 

import numpy as np
import k3match
import pickle 
import math 
from scipy.spatial import KDTree

mach_min = 1.3
gamma = 5/3
save = True
folder = 'sedov'
snap = '100'

def make_tree(snap):
    """ Load data from simulation and build the tree. """
    X = np.load(f'sedov/{snap}/CMx_{snap}.npy')
    Y = np.load(f'sedov/{snap}/CMy_{snap}.npy')
    Z = np.load(f'sedov/{snap}/CMz_{snap}.npy')
    #Star = np.load(f'sedov/{snap}/Star_{snap}.npy')
    Den = np.load(f'sedov/{snap}/Den_{snap}.npy')
    P = np.load(f'sedov/{snap}/P_{snap}.npy')
    T = P/Den #np.load(f'data/{snap}/T_{snap}.npy')
    #Den[Star<0.999] = 0

    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of shape (number_points, 3)
    sim_tree = KDTree(sim_value) 

    return sim_tree, X, Y, Z, Den, P, T

def shock_direction(grad):
    magnitude = np.linalg.norm(grad, axis=1, keepdims=True)
    small_magnitude_indices = magnitude < 1e-3
    # For small magnitudes, set ds to zero
    ds = np.where(small_magnitude_indices, np.zeros_like(grad), -grad / magnitude)

    return ds

def find_Tmach(ratio, gamma):
    """ Find mach nuber from the temperature jump (ratio)."""
    a = 2*gamma*(gamma-1)
    minusb = gamma*2 - 6*gamma + ratio*(gamma+1)**2 + 1
    msquared = (minusb + np.sqrt(minusb**2 + 8*a*(gamma-1))) / (2*a)
    return np.sqrt(msquared)

def find_Pmach(ratio, gamma):
    """ Find mach nuber from the temperature jump (ratio)."""
    msquared = (ratio * (gamma+1) + gamma - 1) / (2*gamma)
    return np.sqrt(msquared)

def find_Denmach(ratio, gamma):
    """ Find mach nuber from the temperature jump (ratio)."""
    denom = gamma + 1 - ratio * (gamma-1)
    msquared = 2 * ratio / denom
    return np.sqrt(msquared)

def searching_tree(tree, point):
    _, idx  = tree.query(point)
    #idx = idx[0]
    return idx

def find_prepost(sim_tree, X, Y, Z, point, check_point, ds, delta, direction):
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
        idx  = searching_tree(sim_tree, new_point) # sim_tree.query(new_point)
        new_point = np.array([X[idx], Y[idx], Z[idx]])
        distance = math.dist(check_point, new_point)
        k += 0.1

    k -=0.1 #becuase you have added it in the end
    final_step = k*np.abs(delta)
    return idx, final_step

def ray_tracer(sim_tree, X, Y, Z, are_u_shock, x_zone, y_zone, z_zone, all_idx, idx, div, dir, direction):
    """ Start from one cell and walk along the shock direction till you go out the shock zone accoridng to Schaal14 (par 2.3.3).
    Parameters
    -----------
    sim_tree: tree.
        Simualation points. 
    X, Y, Z: arrays.
        Points coordinates.
    are_u_shock: bool array.
        Says if a simulation cell is in the shock zone.
    x_zone, y_zone, z_zone: arrays.
        Shock zone points coordinates.
    all_idx: array.
        Tree indeces identifying the cells in the shock zone.
    idx: int.
        Index of the chosen point between the one in the shock zone.
    div: array.
        Velocity divergence of the shock zone points.
    dir: 3D-array.
        Shock direction of the shock zone points.
    delta: float.
        Step you do from your chosen point. It has to be positive!
    direction: str.
        Choose if you want to move towards the 'pre' or 'post' shock region.
    Returns:
    -----------
    final_tree_index: int.
        Tree index of the pre/post shock cell corresponding to the starting one.
    """
    # Take the necessary info of your starting point 
    x_point = x_zone[idx]
    y_point = y_zone[idx]
    z_point = z_zone[idx]   

    point = np.array([x_point, y_point, z_point])
    r_point = np.linalg.norm(point)
    div_v = div[idx]
    dir_sh = np.array(dir[idx])
    
    _, _, dist = k3match.cartesian(X, Y, Z, x_point, y_point, z_point, 2*r_point)
    dist = np.delete(dist, np.argmin(dist))
    delta = np.min(dist)
    
    # Walk till you go out the shock zone
    check_zone = True 
    check_point = point
    while check_zone == True:
        # Find the next point
        i_tree, final_step = find_prepost(sim_tree, X, Y, Z, point, check_point, dir_sh, delta, direction)
        # check if it's in the shock zone
        check_zone = are_u_shock[i_tree]

        if check_zone == True:
            # there will be a index in all_idx equal to i_tree. Find it. 
            idx_zone = np.argmin(np.abs(i_tree-all_idx))  
            # Take the div and dir of that (shock zone) point.
            div_next = div[idx_zone]
            dir_next = dir[idx_zone]

            # if lower div v, you discard the ray.
            if div_next < div_v:
                return False # and then you don't take this cell

            # if opposite direction in shocks, you turn/stop.
            if np.dot(dir_sh, dir_next) < 0:
                check_zone = False # so you exit from the while
        
        check_point = np.array([X[i_tree], Y[i_tree], Z[i_tree]])
        delta = final_step
 
    final_tree_index = i_tree
    
    return final_tree_index

def shock_surface(sim_tree, X, Y, Z, Temp, Press, Den, are_u_shock, x_zone, y_zone, z_zone, all_idx, div, dir):
    """ 
    Find among the cells in the shock zone the one in the shock surface 
    (output: indeces referring to the shockzonefile) 
    and its pre/post shock cells (output: tree indeces).
    """
    surface_Tmach = []
    surface_Pmach = []
    surface_Denmach = []

    #indeces referring to the shockzone file: you use them on xyz_zone and dir
    indeces = [] 
    # indeces referring to the list of ALL simulation cells: you use them on XYZ
    indeces_pre = []
    indeces_post = []

    # loop over all the cells in the shock zone
    for idx in range(len(x_zone)):  
        if np.linalg.norm(dir[idx]) == 0:
            continue

        print(idx)  
        post_tree_index = ray_tracer(sim_tree, X, Y, Z, are_u_shock, x_zone, y_zone, z_zone, 
                                     all_idx, idx, div, dir, direction = 'post')
        if post_tree_index == False:
            continue
        else:
            Tpost = Temp[post_tree_index]
            pre_tree_index = ray_tracer(sim_tree, X, Y, Z, are_u_shock, x_zone, y_zone, z_zone, 
                                        all_idx, idx, div, dir, direction = 'pre')
            if pre_tree_index == False:
                continue
            else:
                Tpre = Temp[pre_tree_index]

                Tbump = Tpost/Tpre
                # check if the Tbump is in the same direction of ds 
                if Tbump < 1:
                    continue 

                # We should also avoid the cells with Tbump< than the one (1.2921782544378697) inferred from M=M_min... So condition 2 doesn't work in shock zone?
                # This already happen using Elad's gradient, not with my approximation
                # if Tbump < 1.292:
                #     continue 

                indeces.append(idx)
                indeces_pre.append(pre_tree_index)
                indeces_post.append(post_tree_index)
                
                Ppre = Press[pre_tree_index]
                Ppost = Press[post_tree_index]
                Pbump = Ppost/Ppre
                Denpre = Den[pre_tree_index]
                Denpost = Den[post_tree_index]
                Denbump = Denpost/Denpre
                
                Tmach = find_Tmach(Tbump, gamma)
                Pmach = find_Pmach(Pbump, gamma)
                Denmach = find_Denmach(Denbump, gamma)

                surface_Tmach.append(Tmach)
                surface_Pmach.append(Pmach)
                surface_Denmach.append(Denmach)
            
    surface_Tmach = np.array(surface_Tmach)
    surface_Pmach = np.array(surface_Pmach)
    surface_Denmach = np.array(surface_Denmach)
    indeces = np.array(indeces)
    indeces_pre = np.array(indeces_pre)
    indeces_post = np.array(indeces_post)

    return surface_Tmach, surface_Pmach, surface_Denmach, indeces, indeces_pre, indeces_post

##
# MAIN
##

sim_tree, X, Y, Z, Den, Press, Temp = make_tree(snap)
# Import data
div = np.load(f'sedov/{snap}/DivV_{snap}.npy')
Eladx_rho = np.load(f'sedov/{snap}/DrhoDx_{snap}.npy')
Elady_rho = np.load(f'sedov/{snap}/DrhoDy_{snap}.npy')
Eladz_rho = np.load(f'sedov/{snap}/DrhoDz_{snap}.npy')
Eladx_p = np.load(f'sedov/{snap}/DpDx_{snap}.npy')
Elady_p = np.load(f'sedov/{snap}/DpDy_{snap}.npy')
Eladz_p = np.load(f'sedov/{snap}/DpDz_{snap}.npy')
with open(f'data/{snap}/areushock_{snap}.pkl', 'rb') as filebool:
    are_u_shock = pickle.load(filebool)
idx_tree = np.loadtxt(f'data/{snap}/shockzone_{snap}.txt')

gradP_all = np.array([Eladx_p, Elady_p, Eladz_p]).T
gradden_all = np.array([Eladx_rho, Elady_rho, Eladz_rho]).T
gradT_all = gradP_all / Den[:, np.newaxis] - Press[:, np.newaxis] * gradden_all / (Den[:, np.newaxis] ** 2)
ds_all = shock_direction(gradT_all)

idx_tree = np.array([int(i) for i in idx_tree])
x_zone = X[idx_tree]
y_zone = Y[idx_tree]
z_zone = Z[idx_tree]
Tzone = Temp[idx_tree]

surface_Tmach, surface_Pmach, surface_Denmach, indeces, indeces_pre, indeces_post = shock_surface(sim_tree, X, Y, Z, Temp, Press, Den, are_u_shock, 
                                                                                                      x_zone, y_zone, z_zone, idx_tree, div, ds_all)

with open(f'shocksurface_{snap}.txt', 'w') as file:
    file.write(f'# Indeces of the shock zone points that are in the shock surface (I.E. they refer to the shockzone file) \n') 
    file.write(' '.join(map(str, indeces)) + '\n')
    file.write('# mach number according to T jump\n') 
    file.write(' '.join(map(str, surface_Tmach)) + '\n')
    file.write('# mach number according to P jump\n') 
    file.write(' '.join(map(str, surface_Pmach)) + '\n')
    file.write('# mach number according to Den jump\n') 
    file.write(' '.join(map(str, surface_Denmach)) + '\n')
    file.write(f'# Tree indeces of the pre shock zone points corresponding to the shock surface points \n') 
    file.write(' '.join(map(str, indeces_pre)) + '\n')
    file.write(f'# Tree indeces of the post shock zone points corresponding to the shock surface points \n') 
    file.write(' '.join(map(str, indeces_post)) + '\n')
    file.close()
