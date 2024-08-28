import Utilities.prelude
import pickle 
import numpy as np
import math
import matplotlib.pyplot as plt
from Utilities.operators import make_tree
from Utilities.isalice import isalice
alice, plot = isalice()

abs_path = '/Users/paolamartire/shocks'

#
## PARAMETERS
#

#%%
# CONSTANTS
##

gamma = 5/3
mach_min = 1.3

#
## FUNCTIONS
#

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
        # Bound the search to the region of interest
        x_point, y_point, z_point = x_zone[idx], y_zone[idx], z_zone[idx]
        if x_point < 7 or x_point > 20 or np.abs(y_point) > 25 or np.abs(z_point)>1: 
            continue
        
        if np.linalg.norm(dir[idx]) == 0:
            continue

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

#%% 
# MAIN
##

save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
check = 'Low' # 'Compton' or 'ComptonHiRes' or 'ComptonRes20'
compton = 'Compton'
snap = '216'

Mbh = 10**m
Rt = Rstar * (Mbh/mstar)**(1/3)
Rp =  Rt / beta
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
if alice:
    if check == 'Low':
        check = ''
    path = f'/data1/martirep/shocks/{folder}{check}/{snap}'
else:
    path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'

# Load the data
data = make_tree(path, snap, energy = False)
dim_cell = data.Vol**(1/3)

Eladx_rho = np.load(f'{path}/DrhoDx_{snap}.npy')
Elady_rho = np.load(f'{path}/DrhoDy_{snap}.npy')
Eladz_rho = np.load(f'{path}/DrhoDz_{snap}.npy')
Eladx_p = np.load(f'{path}/DpDx_{snap}.npy')
Elady_p = np.load(f'{path}/DpDy_{snap}.npy')
Eladz_p = np.load(f'{path}/DpDz_{snap}.npy')
Elad_divV = np.load(f'{path}/DivV_{snap}.npy')
if alice:
    if check == '':
        check = 'Low'
    prepath = f'/data1/martirep/shocks/shock_capturing'
else: 
    prepath = f'/Users/paolamartire/shocks'
with open(f'{prepath}/data/{folder}/{check}/areushock_{snap}.pkl', 'rb') as filebool:
    are_u_shock = pickle.load(filebool)
idx_tree = np.loadtxt(f'{prepath}/data/{folder}/{check}/shockzone_{snap}.txt')

gradP_all = np.array([Eladx_p, Elady_p, Eladz_p]).T
gradden_all = np.array([Eladx_rho, Elady_rho, Eladz_rho]).T
gradT_all = gradP_all / data.Den[:, np.newaxis] - data.Press[:, np.newaxis] * gradden_all / (data.Den[:, np.newaxis] ** 2)
ds_all = shock_direction(gradT_all)

idx_tree = np.array([int(i) for i in idx_tree])
x_zone = data.X[idx_tree]
y_zone = data.Y[idx_tree]
z_zone = data.Z[idx_tree]
Tzone = data.Temp[idx_tree]
div_zone = Elad_divV[idx_tree]
ds_zone = ds_all[idx_tree]
dim_cell_zone = dim_cell[idx_tree]


surface_Tmach, surface_Pmach, surface_Denmach, indeces, indeces_pre, indeces_post = \
    shock_surface(data.sim_tree, data.X, data.Y, data.Z, data.Temp, data.Press, data.Den, are_u_shock,
                  x_zone, y_zone, z_zone, idx_tree, div_zone, ds_zone)

#%%
if save == True:
    with open(f'{prepath}/data/{folder}/{check}/shocksurface_{snap}.txt', 'w') as file:
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
    