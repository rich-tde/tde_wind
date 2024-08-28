import sys
sys.path.append('/Users/paolamartire/shocks/')
import Utilities.prelude
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import math
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

gamma_ad = 5/3
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
    """ Find shock direction according eq.(5) by Schaal14.
    Parameters
    -----------
    grad: array (1x3)
        Gradient of temperature.
    Returns
    -----------
    grad: array (1x3)
        Gradient of temperature.
    ds: array (1x3)
        Shock direction.
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
    sim_tree, X, Y, Z: tree, arrays
        Simulation points tree and coordinates.
    point: array (1x3).
        Coordinates of the starting point.
    ds: array (1x3).
        Shock direction.
    delta: float.
        Step to do to search.
    direction: str.
        Choose pre or post shock.
    Returns
    -----------
    idx: int.
        Index of the pre('pre') / post('post') shock point.
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
        Simulation points tree.
    X, Y, Z, Press, Temp: arrays.
        Coordinates, pressure and temperature of the points of the tree.
    point: array.
        Coordinates of the starting point.
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
    
def shock_zone(divv, gradT, gradrho, cond3, check_cond = 'all'):
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

# Parameters
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

# TDE quantities and names
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

#%% Load data
data = make_tree(path, snap, energy = False)
dim_cell = data.Vol**(1/3) # according to Elad
Eladx_rho = np.load(f'{path}/DrhoDx_{snap}.npy')
Elady_rho = np.load(f'{path}/DrhoDy_{snap}.npy')
Eladz_rho = np.load(f'{path}/DrhoDz_{snap}.npy')
Eladx_p = np.load(f'{path}/DpDx_{snap}.npy')
Elady_p = np.load(f'{path}/DpDy_{snap}.npy')
Eladz_p = np.load(f'{path}/DpDz_{snap}.npy')
Elad_divV = np.load(f'{path}/DivV_{snap}.npy')
# Eladx_rholim = np.load(f'{path}/DrhoDxLimited_{snap}.npy')
# Elady_rholim = np.load(f'{path}/DrhoDyLimited_{snap}.npy')
# Eladz_rholim = np.load(f'{path}/DrhoDzLimited_{snap}.npy')
# Eladx_plim = np.load(f'{path}/DpDxLimited_{snap}.npy')
# Elady_plim = np.load(f'{path}/DpDyLimited_{snap}.npy')
# Eladz_plim = np.load(f'{path}/DpDzLimited_{snap}.npy')
# Elad_divVlim = np.load(f'{path}/divVLimited_{snap}.npy')

#%% Compute shock zone
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
    
    if math.isnan(np.linalg.norm(grad_temp)) or math.isnan(np.linalg.norm(div_vel)):
        are_u_shock[i] = False
        continue 

    cond3 = condition3(data.sim_tree, data.X, data.Y, data.Z, data.Press, data.Temp, point, ds, mach_min, gamma_ad, step)
    # shock1 = shock_zone(div_vel, grad_temp, grad_den, cond3, check_cond =  '1')
    # shock2 = shock_zone(div_vel, grad_temp, grad_den, cond3, check_cond =  '2')
    shock = shock_zone(div_vel, grad_temp, grad_den, cond3, check_cond = '3')
    are_u_shock[i] = shock
    
    # if shock1 == True:
    #     X_shock1.append(data.X[i])
    #     Y_shock1.append(data.Y[i])
    #     Z_shock1.append(data.Z[i])
    #     idx_tree1.append(i)

    # if shock2 == True:
    #     X_shock2.append(data.X[i])
    #     Y_shock2.append(data.Y[i])
    #     Z_shock2.append(data.Z[i])
    #     idx_tree2.append(i)

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

# X_shock1 = np.array(X_shock1)
# Y_shock1 = np.array(Y_shock1)
# Z_shock1 = np.array(Z_shock1)

# X_shock2 = np.array(X_shock2)
# Y_shock2 = np.array(Y_shock2)
# Z_shock2 = np.array(Z_shock2)

if save == True:
    if alice:
        if check == '':
            check = 'Low'
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

#%%
if plot:
    from Utilities.sections import make_slices
    from matplotlib import colors

    if alice:
        if check == '':
            check = 'Low'
        prepath = f'/data1/martirep/shocks/shock_capturing'
    else: 
        prepath = f'/Users/paolamartire/shocks'
        
    idx_tree = np.loadtxt(f'{prepath}/data/{folder}/{check}/shockzone_{snap}.txt')
    idx_tree = np.array([int(i) for i in idx_tree])
    X = np.load(f'{path}grad/CMx_{snap}.npy')
    Y = np.load(f'{path}grad/CMy_{snap}.npy')
    Z = np.load(f'{path}grad/CMz_{snap}.npy')
    Den = np.load(f'{path}grad/Den_{snap}.npy')
    Elad_divV = np.load(f'{path}grad/DivV_{snap}.npy')
    Vol = np.load(f'{path}grad/Vol_{snap}.npy')
    dim_cell = Vol**(1/3)
    
    x_zone, y_zone, z_zone, dim_cell_zone = X[idx_tree], Y[idx_tree], Z[idx_tree], dim_cell[idx_tree]
    midplane = np.abs(Z)<dim_cell
    X_midplane, Y_midplane, Z_midplane, Den_midplane, div_midplane = make_slices([X, Y, Z, Den, Elad_divV], midplane)
    midplane_zone = np.abs(z_zone)<dim_cell_zone
    x_zonemidplane, y_zonemidplane, z_zonemidplane = make_slices([x_zone, y_zone, z_zone], midplane_zone)

    fig, ax = plt.subplots(1,1,figsize=(12,10))
    img = ax.scatter(X_midplane, Y_midplane, c = div_midplane, s = 1, cmap = 'plasma', vmin = -5, vmax = 2)#, norm=colors.LogNorm(vmin = 1e-9, vmax = 1e-5))
    # img = ax.scatter(X_midplane, Y_midplane, c = Den_midplane, s = 2, cmap = 'inferno', norm=colors.LogNorm(vmin = 1e-9, vmax = 1e-5))
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(r'$\nabla \cdot \mathbf{v}$', fontsize = 20)
    ax.plot(x_zonemidplane, y_zonemidplane, 'ks', markerfacecolor='k', ms=4, markeredgecolor='k', label = 'Shock zone')
    ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
    ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
    ax.set_xlim(-10,30)
    ax.set_ylim(-20,20)
    ax.legend(loc = 'upper right', fontsize = 20)
    # if save:    
    #     plt.savefig(f'Figs/{snap}/3XYshockzone_conditions_{snap}zoom.png')
    plt.show()

# %%
