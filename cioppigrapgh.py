import numpy as np
import h5py
import scipy
import pickle 

##
# Parameters
##
folder = 'TDE'
snap = '196'
gamma = 5/3
mach_min = 1.3

##
# FUNCTIONS
##

def voronoi_neigh(coordinates):
    vor = scipy.spatial.Voronoi(coordinates)
    num_centers = vor.points.shape[0]
    connections = vor.ridge_points  
    graph = {}
    for center_index in np.arange(num_centers):
        # A -> B
        interesting_connections_first = connections[connections[:,0]==center_index]
        neighbours_list_first = interesting_connections_first[:,1]
        # B -> A
        interesting_connections_second = connections[connections[:,1]==center_index]
        neighbours_list_second = interesting_connections_second[:,0]
        # unique neighbours list
        neighbours_list = np.concatenate([neighbours_list_first, neighbours_list_second])
        neighbours_list = np.unique(neighbours_list)
        graph[center_index] = {"neighbours_indeces": neighbours_list}
            
    return graph

def walkTheGraph(coordinates, graph, current_position, suggested_movement, direction):
    # Find neighbours
    neighbours_indeces = graph[current_position]['neighbours_indeces']
    neighbours_positions = np.array(coordinates[neighbours_indeces])
    distance_vectors_list = neighbours_positions - coordinates[current_position]

    suggested_movement = np.array([suggested_movement])
    if direction == 'post':
        suggested_movement = - suggested_movement
    # computation
    cosines_sim_list = np.squeeze(distance_vectors_list @ suggested_movement.T) / (np.linalg.norm(distance_vectors_list, axis=-1) * np.linalg.norm(suggested_movement))
    cosines_sim_list = 1 - cosines_sim_list
    chosen_neighbour = graph[current_position]['neighbours_indeces'][np.argmin(cosines_sim_list)]
    return chosen_neighbour

# Jump conditions
def temperature_bump(mach, gamma):
    """ T_post/ T_pre shock according to RH conditions."""
    Tbump =  (mach**2 * (gamma-1) + 2) * (2 * gamma * mach**2 - (gamma-1)) / (mach**2 * (gamma+1)**2)
    return Tbump

def pressure_bump(mach, gamma):
    """ P_post/ P_pre shock according to RH conditions."""
    Pbump = (2 * gamma * mach**2 - (gamma-1)) / (gamma+1)
    return Pbump

# Mach number from jump considtions
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

def shock_direction(grad):
    magnitude = np.linalg.norm(grad, axis=1, keepdims=True)
    small_magnitude_indices = magnitude < 1e-3
    # For small magnitudes, set ds to zero
    ds = np.where(small_magnitude_indices, np.zeros_like(grad), -grad / magnitude)

    return ds

def condition3(coordinates, graph, Press, Temp, current_position, initial_ds, mach_min, gamma):
    """ Last condition fot shock zone by Schaal14 .
    Parameters
    -----------
    graph: tree.
            Simulation points.
    coordinates, Press, Temp: (n)arrays.
            Coordinates, pressure and temperature of the points of the tree.
    current_position: array.
            Index of the starting point.
    import pickle : array (1x3)
        Shock direction.
    mach_min, gamma: floats.
                    Minimum mach number, adiabatic index.
    Returns
    -----------
    bool.
        If condition is satisfied or not.
    """
    # Find (the index in the tree of) the point in the pre/post shock region.
    idxpost = walkTheGraph(coordinates, graph, current_position, initial_ds, direction = 'post')
    idxpre = walkTheGraph(coordinates, graph, current_position, initial_ds, direction = 'pre')

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
    
def ray_tracer(coordinates, graph, div_v, ds, are_u_shock, initial_position, direction):
    """ Start from one cell and walk along the shock direction till you go out the shock zone accoridng to Schaal14 (par 2.3.3).
    Parameters
    -----------
    graph: tree.
        Simualation points. 
    div_v: array.
        Velocity divergence of the simulation points.
    ds: 3D-array.
        Shock direction of the simulation points.
    are_u_shock: bool array.
        Says if a simulation cell is in the shock zone.
    initial_position: int.
        (Graph) index of the chosen point.
    direction: str.
        Choose if you want to move towards the 'pre' or 'post' shock region.
    Returns:
    -----------
    outer_neigh: int.
        Graph index of the pre/post shock cell corresponding to the starting one.
    """
    # Walk till you go out the shock zone
    initial_ds = ds[initial_position]
    initial_divv = div_v[initial_position]

    current_position = initial_position
    check_zone = True 
    while check_zone == True:
        # Find the next point
        idx_next_neigh = walkTheGraph(coordinates, graph, current_position, initial_ds, direction)
        # check if it's in the shock zone
        inside_check = are_u_shock[idx_next_neigh]

        if inside_check == True:
            next_divv = div_v[idx_next_neigh]
            next_ds = ds[idx_next_neigh]

            # if lower div v, you discard the ray.
            if next_divv < initial_divv:
                return False # and then you don't take this cell

            # if opposite direction in shocks, you turn/stop.
            if np.dot(initial_ds, next_ds) < 0:
                check_zone = False # so you exit from the while
        
        current_position = idx_next_neigh
        check_zone = inside_check
    
    outer_neigh = current_position
    
    return outer_neigh

def shock_surface(coordinates, graph, Temp, Press, Den, div_v, ds, are_u_shock, indeces_zone):
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
    for i in range(len(indeces_zone)):
        initial_position = int(indeces_zone[i])
        post_index = ray_tracer(coordinates, graph, div_v, ds, are_u_shock, initial_position, direction = 'post')

        if post_index == False:
            continue

        else:
            Tpost = Temp[post_index]
            pre_index = ray_tracer(coordinates, graph, div_v, ds, are_u_shock, initial_position, direction = 'pre')

            if pre_index == False:
                continue
            else:
                Tpre = Temp[pre_index]

                Tbump = Tpost/Tpre
                # check if the Tbump is in the same direction of ds 
                if Tbump < 1:
                    continue 

                # We should also avoid the cells with Tbump< than the one (1.2921782544378697) inferred from M=M_min... So condition 2 doesn't work in shock zone?
                # This already happen using Elad's gradient, not with my approximation
                # if Tbump < 1.292:
                #     continue 

                indeces.append(initial_position)
                indeces_pre.append(pre_index)
                indeces_post.append(post_index)
                
                Ppre = Press[pre_index]
                Ppost = Press[post_index]
                Pbump = Ppost/Ppre
                Denpre = Den[pre_index]
                Denpost = Den[post_index]
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

# Main
# Load data
X = np.load(f'./data/CMx_{snap}.npy')
Y = np.load(f'./data/CMy_{snap}.npy')
Z = np.load(f'./data/CMz_{snap}.npy')
P = np.load(f'./data/P_{snap}.npy')
T = np.load(f'./data/T_{snap}.npy')
Den = np.load(f'./data/Den_{snap}.npy')
Star = np.load(f'./data/Star_{snap}.npy')
Elad_divV = np.load(f'./data/DivV_{snap}.npy')
Eladx_rho = np.load(f'./data/DrhoDx_{snap}.npy')
Elady_rho = np.load(f'./data/DrhoDy_{snap}.npy')
Eladz_rho = np.load(f'./data/DrhoDz_{snap}.npy')
Eladx_p = np.load(f'./data/DpDx_{snap}.npy')
Elady_p = np.load(f'./data/DpDy_{snap}.npy')
Eladz_p = np.load(f'./data/DpDz_{snap}.npy')

print('Data loaded')

# Making data ready to be used
Den[Star<0.999] = 0
coordinates = np.array([X,Y,Z]).T
gradP_all = np.array([Eladx_p, Elady_p, Eladz_p]).T
gradden_all = np.array([Eladx_rho, Elady_rho, Eladz_rho]).T
gradT_all = gradP_all / Den[:, np.newaxis] - P[:, np.newaxis] * gradden_all / (Den[:, np.newaxis] ** 2)
ds_all = shock_direction(gradT_all)

# Re-build Voronoi structure
graph = voronoi_neigh(coordinates)

print('Graph finished')

# save data
with open('graph.pkl', 'wb') as filegraph:
    pickle.dump(graph, filegraph)

# Compute shock zone
shock_dirx = []
shock_diry = []
shock_dirz = []
X_shock = []
Y_shock = []
Z_shock = []
are_u_shock = np.zeros(len(X), dtype = bool)
indeces_zone = []

for i in range(len(X)):
    if Den[i]<1e-9: # threshold is 1e-15.
        are_u_shock[i] = False
        continue

    gradP = gradP_all[i]
    gradDen = gradden_all[i]
    divV = Elad_divV[i] 
    gradT = gradT_all[i]
    ds = ds_all[i]
    
    # fondamentale!!
    if np.linalg.norm(ds) == 0:
        are_u_shock[i] = False
        continue

    cond3 = condition3(coordinates, graph, P, T, i, ds, mach_min, gamma) 
    shock = shock_zone(divV, gradT, gradDen, cond3)
    are_u_shock[i] = shock

    if shock == True:
        indeces_zone.append(i)

indeces_zone = np.array(indeces_zone)

print('Shock zone done')

# save data 
with open(f'areushock_{snap}.pkl', 'wb') as filebool:
        pickle.dump(are_u_shock, filebool)
with open(f'shockzone_{snap}.txt', 'w') as filezone:
        filezone.write('# Indeces in the graph \n') 
        filezone.write(' '.join(map(str, indeces_zone)) + '\n')
        filezone.close()
print('Shock zone saved')

surface_Tmach, surface_Pmach, surface_Denmach, indeces, indeces_pre, indeces_post = shock_surface(coordinates, graph, T, P, Den, Elad_divV, ds_all, are_u_shock, indeces_zone)

print('Shock surface done')

# save data
with open(f'shocksurface_{snap}.txt', 'w') as filesurf:
        filesurf.write(f'# Indeces of the shock surface points in the graph) \n') 
        filesurf.write(' '.join(map(str, indeces)) + '\n')
        filesurf.write(f'# Pre shock zone points \n') 
        filesurf.write(' '.join(map(str, indeces_pre)) + '\n')
        filesurf.write(f'# Post shock zone points \n') 
        filesurf.write(' '.join(map(str, indeces_post)) + '\n')
        filesurf.close()

print('Shock surface saved')