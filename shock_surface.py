import Utilities.prelude
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import k3match
from tqdm import tqdm
import pickle 
import math 
import matplotlib
import numba
from scipy.spatial import KDTree
from multiprocessing import Pool
from Utilities.operators import make_tree

z_chosen = 0
mach_min = 1.3
gamma = 5/3
save = True
folder = 'TDE'
snap = '196'
m = 5
path = f'{folder}/{snap}'

if folder == 'TDE':
    is_tde = True
elif folder == 'sedov':
    is_tde = False

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
    _, idx = tree.query(point)
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
    
    _, _, dist = k3match.cartesian(X,Y,Z, x_point, y_point, z_point, 2*r_point)
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

def parallel_tracer(idx, sim_tree, X, Y, Z, Temp, are_u_shock, x_zone, y_zone, z_zone, all_idx, div, dir):    
    if np.linalg.norm(dir[idx]) == 0:
        return None

    post_tree_index = ray_tracer(sim_tree, X, Y, Z, are_u_shock, x_zone, y_zone, z_zone, 
                                    all_idx, idx, div, dir, direction = 'post')
    if post_tree_index == False:
        return None
    else:
        Tpost = Temp[post_tree_index]
        pre_tree_index = ray_tracer(sim_tree, X, Y, Z, are_u_shock, x_zone, y_zone, z_zone, 
                                    all_idx, idx, div, dir, direction = 'pre')
        if pre_tree_index == False:
            return None
        else:
            Tpre = Temp[pre_tree_index]

            Tbump = Tpost/Tpre
            # check if the Tbump is in the same direction of ds 
            if Tbump < 1:
                return None 
            
    return idx, pre_tree_index, post_tree_index
    
def shock_surface(sim_tree, X, Y, Z, Temp, Press, Den, are_u_shock, x_zone, y_zone, z_zone, all_idx, div, dir):
    """ 
    Find among the cells in the shock zone the one in the shock surface 
    (output: indeces referring to the shockzonefile) 
    and its pre/post shock cells (output: tree indeces).
    """
    print((len(x_zone)))
    # loop over all the cells in the shock zone
    with Pool() as pool:
        args = [(i, sim_tree, X, Y, Z, Temp, are_u_shock, x_zone, y_zone, z_zone, all_idx, div, dir) for i in range(len(x_zone))]
        results = pool.starmap(parallel_tracer, args)

    results = [i for i in results if i != None]
    results = np.transpose(results)
    indeces = results[0]
    indeces_pre = results[1]
    indeces_post = results[2]

    # Tempre = Temp[indeces_pre]
    # Tempost = Temp[indeces_post]
    # Tbump = Tempost/Tempre 
    # Ppre = Press[indeces_pre]
    # Ppost = Press[indeces_post]
    # Pbump = Ppost/Ppre
    # Denpre = Den[indeces_pre]
    # Denpost = Den[indeces_post]
    # Denbump = Denpost/Denpre
    
    # Tmach = find_Tmach(Tbump, gamma)
    # Pmach = find_Pmach(Pbump, gamma)
    # Denmach = find_Denmach(Denbump, gamma)

    # surface_Tmach.append(Tmach)
    # surface_Pmach.append(Pmach)
    # surface_Denmach.append(Denmach)
            
    # surface_Tmach = np.array(surface_Tmach)
    # surface_Pmach = np.array(surface_Pmach)
    # surface_Denmach = np.array(surface_Denmach)
    # indeces = np.array(indeces)
    # indeces_pre = np.array(indeces_pre)
    # indeces_post = np.array(indeces_post)

    return indeces, indeces_pre, indeces_post

if __name__ == '__main__':
    sim_tree, X, Y, Z, Vol, VX, VY, VZ, Den, Press, Temp = make_tree(path, snap, is_tde)
    dim_cell = (3*Vol/(4*np.pi))**(1/3)

    # Import data
    Elad_div = np.load(f'{folder}/{snap}/DivV_{snap}.npy')
    Eladx_rho = np.load(f'{folder}/{snap}/DrhoDx_{snap}.npy')
    Elady_rho = np.load(f'{folder}/{snap}/DrhoDy_{snap}.npy')
    Eladz_rho = np.load(f'{folder}/{snap}/DrhoDz_{snap}.npy')
    Eladx_p = np.load(f'{folder}/{snap}/DpDx_{snap}.npy')
    Elady_p = np.load(f'{folder}/{snap}/DpDy_{snap}.npy')
    Eladz_p = np.load(f'{folder}/{snap}/DpDz_{snap}.npy')
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
    div_zone = Elad_div[idx_tree]
    ds_zone = ds_all[idx_tree]
    dim_cell_zone = dim_cell[idx_tree]
    
    indeces, indeces_pre, indeces_post = shock_surface(sim_tree, X, Y, Z, Temp, Press, Den, are_u_shock, 
                                                        x_zone, y_zone, z_zone, idx_tree, div_zone, ds_zone)
    
    surface_x = x_zone[indeces]
    surface_y = y_zone[indeces]
    surface_z = z_zone[indeces]
    epsilon_surf = dim_cell_zone[indeces]

    post_x = X[indeces_post]
    post_y = Y[indeces_post]
    post_z = Z[indeces_post]
    pre_x = X[indeces_pre]
    pre_y = Y[indeces_pre]
    pre_z = Z[indeces_post]

    epsilon_post = dim_cell[indeces_post]
    epsilon_pre = dim_cell[indeces_pre]

    if save == True:
        with open(f'TESTshocksurface_{snap}.txt', 'w') as file:
            file.write(f'# Indeces of the shock zone points that are in the shock surface (I.E. they refer to the shockzone file) \n') 
            file.write(' '.join(map(str, indeces)) + '\n')
            file.write(f'# Tree indeces of the pre shock zone points corresponding to the shock surface points \n') 
            file.write(' '.join(map(str, indeces_pre)) + '\n')
            file.write(f'# Tree indeces of the post shock zone points corresponding to the shock surface points \n') 
            file.write(' '.join(map(str, indeces_post)) + '\n')
            file.close()
    
    # Maybe for every cell of the shock surface you don't see pre/post because they are not on the same layers
    if folder == 'sedov':
        zone_cross_x = x_zone[np.abs(z_zone-z_chosen)< dim_cell_zone]
        zone_cross_y = y_zone[np.abs(z_zone-z_chosen)< dim_cell_zone]

        surface_cross_x = surface_x[np.abs(surface_z-z_chosen)< epsilon_surf]
        surface_cross_y = surface_y[np.abs(surface_z-z_chosen)< epsilon_surf]

        # cross_shockdirx = surface_dirx[np.abs(surface_z-z_chosen)< epsilon_surf]
        # cross_shockdiry = surface_diry[np.abs(surface_z-z_chosen)< epsilon_surf]

        zone_post_x = post_x[np.logical_and(np.abs(surface_z-z_chosen)<epsilon_surf, np.abs(post_z-z_chosen)<epsilon_post)] # post_x[np.abs(surface_z-z_chosen)<epsilon_surf] 
        zone_post_y = post_y[np.logical_and(np.abs(surface_z-z_chosen)<epsilon_surf, np.abs(post_z-z_chosen)<epsilon_post)] # post_y[np.abs(surface_z-z_chosen)<epsilon_surf]

        zone_pre_x = pre_x[np.logical_and(np.abs(surface_z-z_chosen)<epsilon_surf, np.abs(pre_z-z_chosen)<epsilon_pre)] # pre_x[np.abs(surface_z-z_chosen)<epsilon_surf]
        zone_pre_y = pre_y[np.logical_and(np.abs(surface_z-z_chosen)<epsilon_surf, np.abs(pre_z-z_chosen)<epsilon_pre)] # pre_y[np.abs(surface_z-z_chosen)<epsilon_surf]

        fig, ax = plt.subplots(1,1, figsize = (9,9))
        ax.plot(zone_cross_x, zone_cross_y, 'ks', markerfacecolor='none', ms = 5, markeredgecolor='coral', label = 'Shock zone')
        ax.plot(surface_cross_x, surface_cross_y, 'ks', markerfacecolor='maroon', ms = 4, markeredgecolor='none',  alpha = 0.8, label = 'Shock surface')

        ax.plot(zone_post_x, zone_post_y, 'ks', markerfacecolor='sandybrown', ms = 5, markeredgecolor='none', alpha = 0.8,  label = 'Post shock')
        ax.plot(zone_pre_x, zone_pre_y, 'ks', markerfacecolor='lightskyblue', ms = 5, markeredgecolor='none', alpha = 0.8,  label = 'Pre shock')
        # ax.quiver(surface_cross_x, surface_cross_y, cross_shockdirx, cross_shockdiry, color = 'k', angles='xy', scale_units='xy', width = 3e-3)#, scale = 20)

        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.set_xlabel('X', fontsize = 18)
        ax.set_ylabel('Y', fontsize = 18)
        ax.legend()
        ax.set_title(r'z = V$^{1/3}$', fontsize = 16)
        plt.grid()
        if save == True:
            plt.savefig(f'TEST{snap}.png')
        plt.show()
