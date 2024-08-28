import Utilities.prelude
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import math
import k3match
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

def temperature_bump(mach, gamma):
    """ T_post/ T_pre shock according to RH conditions."""
    Tbump =  (mach**2 * (gamma-1) + 2) * (2 * gamma * mach**2 - (gamma-1)) / (mach**2 * (gamma+1)**2)
    return Tbump

def pressure_bump(mach, gamma):
    """ P_post/ P_pre shock according to RH conditions."""
    Pbump = (2 * gamma * mach**2 - (gamma-1)) / (gamma+1)
    return Pbump

def shock_direction(grad):
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
    magnitude = np.linalg.norm(grad)
    
    if np.linalg.norm(grad) <1e-3:
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
        Index of the pre/post shock point
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
    
def shock_zone(divv, gradT, gradrho, cond3, check_cond = '3'):
    """ Find the shock zone according conditions in Sec. 2.3.2 of Schaal14. 
    In order to test the code, with "check_con" you can decide if checking all or some of the conditions."""
    if check_cond == '1' or check_cond == '2':
        cond3 = True # so you don't check it
        if check_cond == '2':
            cond2 = np.dot(gradT, gradrho)
        else:
            cond2 = 10
    else:
        cond2 = np.dot(gradT, gradrho)

    if np.logical_and(divv<0, np.logical_and(cond2 > 0, cond3 == True)):
        return True
    else:
        return False

#
## MAIN
#
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
    path = f'/home/martirep/shocks/{folder}{check}/{snap}'
else:
    path = f'/Users/paolamartire/shocks/TDE/{folder}{check}/{snap}'

#%%
data = make_tree(path, snap, energy = False)
dim_cell = data.Vol**(1/3) # according to Elad
# tfb = days_since_distruption(f'{path}/snap_{snap}.h5', m, mstar, Rstar, choose = 'tfb')

#%% Load gradients
# Eladx_rholim = np.load(f'{folder}/{snap}/DrhoDxLimited_{snap}.npy')
# Elady_rholim = np.load(f'{folder}/{snap}/DrhoDyLimited_{snap}.npy')
# Eladz_rholim = np.load(f'{folder}/{snap}/DrhoDzLimited_{snap}.npy')
Eladx_rho = np.load(f'{folder}/{snap}/DrhoDx_{snap}.npy')
Elady_rho = np.load(f'{folder}/{snap}/DrhoDy_{snap}.npy')
Eladz_rho = np.load(f'{folder}/{snap}/DrhoDz_{snap}.npy')
# Eladx_plim = np.load(f'{folder}/{snap}/DpDxLimited_{snap}.npy')
# Elady_plim = np.load(f'{folder}/{snap}/DpDyLimited_{snap}.npy')
# Eladz_plim = np.load(f'{folder}/{snap}/DpDzLimited_{snap}.npy')
Eladx_p = np.load(f'{folder}/{snap}/DpDx_{snap}.npy')
Elady_p = np.load(f'{folder}/{snap}/DpDy_{snap}.npy')
Eladz_p = np.load(f'{folder}/{snap}/DpDz_{snap}.npy')

Elad_divVlim = np.load(f'{folder}/{snap}/divVLimited_{snap}.npy')
Elad_divV = np.load(f'{folder}/{snap}/divV_{snap}.npy')

# Compute shock zone
shock_dirx = []
shock_diry = []
shock_dirz = []
X_shock1 = []
Y_shock1 = []
Z_shock1 = []
X_shock2 = []
Y_shock2 = []
Z_shock2 = []
X_shock = []
Y_shock = []
Z_shock = []
div_shock = []
T_shock = []
are_u_shock = np.zeros(len(data.X), dtype = bool)
x_who = []
y_who = []
z_who = []
idx_tree = []
idx_tree1 = []
idx_tree2 = []

for i in range(len(data.X)):
    if i%10_000 == 0:
        print(i)
    point = np.array([data.X[i],data.Y[i],data.Z[i]])

    # if np.linalg.norm(point)>threshold:
    #     masked += 1
    #     are_u_shock[i] = False
    #     continue

    if data.Den[i]<1e-9: # threshold is 1e-15. but we increase to speed it up
        are_u_shock[i] = False
        continue

    step = 2*dim_cell[i]
    # grad_temp, ds = shock_direction(sim_tree, X, Y, Z, Temp, point, step)
    # grad_den = calc_grad(sim_tree, X, Y, Z, Den, point, step)
    # div_vel = calc_div(sim_tree, X, Y, Z, VX, VY, VZ, point, step)

    gradP = np.array([Eladx_p[i], Elady_p[i], Eladz_p[i]])
    grad_den = np.array([Eladx_rho[i], Elady_rho[i], Eladz_rho[i]]) 
    div_vel = Elad_divV[i] 
    
    grad_temp = np.zeros(3)
    for k in range(3):
        gradP_fords = np.array([Eladx_p[i], Elady_p[i], Eladz_p[i]])
        grad_den_fords = np.array([Eladx_rho[i], Elady_rho[i], Eladz_rho[i]]) 
        grad_temp[k] = gradP_fords[k]/data.Den[i] - data.Press[i] * grad_den_fords[k]/ (data.Den[i])**2

    _, ds = shock_direction(grad_temp)
    
    # fondamentale!!
    if np.linalg.norm(ds) == 0:
        are_u_shock[i] = False
        continue
    
    if math.isnan(np.linalg.norm(grad_temp)):
        # non dovrebbe succedere se prendi ds=2*max
        continue 

    if math.isnan(np.linalg.norm(div_vel)):
        # non dovrebbe succedere per come prendi step in ds=2*max
        are_u_shock[i] = False
        print('nan in div_v')

    cond3 = condition3(data.sim_tree, data.X, data.Y, data.Z, data.Press, data.Temp, point, ds, mach_min, gamma, step)
    shock1 = shock_zone(div_vel, grad_temp, grad_den, cond3, check_cond =  '1')
    shock2 = shock_zone(div_vel, grad_temp, grad_den, cond3, check_cond =  '2')
    shock = shock_zone(div_vel, grad_temp, grad_den, cond3, check_cond = '3')
    are_u_shock[i] = shock
    
    if shock1 == True:
        X_shock1.append(data.X[i])
        Y_shock1.append(data.Y[i])
        Z_shock1.append(data.Z[i])
        idx_tree1.append(i)

    if shock2 == True:
        X_shock2.append(data.X[i])
        Y_shock2.append(data.Y[i])
        Z_shock2.append(data.Z[i])
        idx_tree2.append(i)

    if shock == True:
        X_shock.append(data.X[i])
        Y_shock.append(data.Y[i])
        Z_shock.append(data.Z[i])
        shock_dirx.append(ds[0])
        shock_diry.append(ds[1])
        shock_dirz.append(ds[2])
        div_shock.append(div_vel)
        T_shock.append(data.Temp[i])
        idx_tree.append(i)

X_shock = np.array(X_shock)
Y_shock = np.array(Y_shock)
Z_shock = np.array(Z_shock)

X_shock1 = np.array(X_shock1)
Y_shock1 = np.array(Y_shock1)
Z_shock1 = np.array(Z_shock1)

X_shock2 = np.array(X_shock2)
Y_shock2 = np.array(Y_shock2)
Z_shock2 = np.array(Z_shock2)

if save == True:
    if alice:
        prepath = f'/data1/martirep/shocks/shock_capturing'
    else: 
        prepath = f'/Users/paolamartire/shocks'
    with open(f'{prepath}/data/{folder}/{check}/shockzone_{snap}.txt', 'w') as file:
        file.write(f'# Coordinates of the points in the shock zone, mach_min = {mach_min} \n# X \n') 
        file.write('# Index in the tree \n') 
        file.write(' '.join(map(str, idx_tree)) + '\n')
        file.close()

    with open(f'{prepath}/data/{folder}/{check}/areushock_{snap}.pkl', 'wb') as filebool:
        pickle.dump(are_u_shock, filebool)

if plot:
    cross_sec = 0
    dim_cell_zone = dim_cell[idx_tree]
    dim_cell_zone1 = dim_cell[idx_tree1]
    dim_cell_zone2 = dim_cell[idx_tree2]

    # Plot 3 conditions (cross section)
    if(len(X_shock1[np.abs(Z_shock1-cross_sec)<dim_cell_zone1])>1):
        if folder == 'sedov':
            plt.figure(figsize=(10,10))
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1,1.1)
            plt.xlabel('X', fontsize = 18)
            plt.ylabel('Y', fontsize = 18)
        else:
            plt.figure(figsize=(10,6))
            plt.xlabel(r'X[$R_\odot$]', fontsize = 18)
            plt.ylabel(r'Y[$R_\odot$]', fontsize = 18)
            plt.xlim(-150,100)
            plt.ylim(-150,100)
        plt.plot(X_shock1[np.abs(Z_shock1-cross_sec)<dim_cell_zone1], Y_shock1[np.abs(Z_shock1-cross_sec)<dim_cell_zone1], 'ks', markerfacecolor='none', ms=8, markeredgecolor='coral', label = '1')
        plt.plot(X_shock2[np.abs(Z_shock2-cross_sec)<dim_cell_zone2], Y_shock2[np.abs(Z_shock2-cross_sec)<dim_cell_zone2], 'ks', markerfacecolor='orange', ms=6, markeredgecolor='orange', label = '1,2')
        plt.plot(X_shock[np.abs(Z_shock-cross_sec)<dim_cell_zone], Y_shock[np.abs(Z_shock-cross_sec)<dim_cell_zone], 'ks', markerfacecolor='k', ms=4, markeredgecolor='k', label = '1,2,3')
        plt.legend(loc = 'upper left', fontsize = 18)
        plt.grid()
        plt.title(f'Conditions shock zone, z={cross_sec}', fontsize = 18)
        if save:    
            plt.savefig(f'Figs/{snap}/3XYshockzone_conditions_{snap}zoom.png')
        plt.show()