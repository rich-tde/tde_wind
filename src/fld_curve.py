from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'

import sys
sys.path.append(abspath)

# import gc
# import warnings
# warnings.filterwarnings('ignore')
# import csv

import numpy as np
import h5py
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from tqdm import tqdm
# from sklearn.neighbors import KDTree


import Utilities.prelude as c
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
# from Utilities.parser import parse
from Utilities.selectors_for_snap import select_snap, select_prefix


#%% Choose parameters -----------------------------------------------------------------
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
step = ''
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

check = 'Low' # 'Low' or 'HiRes'
save = True
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, step, time = True) #[100,115,164,199,216]
Lphoto_all = np.zeros(len(snaps))

#%% Opacities -----------------------------------------------------------------
# Freq range
f_min = c.Kb * 1e3 / c.h
f_max = c.Kb * 3e13 / c.h
f_num = 1_000
frequencies = np.logspace(np.log10(f_min), np.log10(f_max), f_num)

# Opacity Input
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

# Fill value none extrapolates
def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    z = np.array(z)
    #D = [di for di in D0]

    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
    #D.append(to_stack)
    return np.array(D), z

def pad_interp(x,y,V):
    Vn, xn = linearpad(V, x)
    Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = linearpad(Vn, y)
    Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn

T_cool2, Rho_cool2, rossland2 = pad_interp(T_cool, Rho_cool, rossland.T)
_, _, plank2 = pad_interp(T_cool, Rho_cool, plank.T)

# MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

if alice:
    if check == 'Low':
        check = ''
pre = select_prefix(m, check, mstar, Rstar, beta, n, compton, step)
print('we are in: ', pre)
for idx_s, snap in enumerate(snaps):
    print('\n Snapshot: ', snap, '\n')
    box = np.zeros(6)
    #%% Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
        T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy')
        Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
        with h5py.File(f'{pre}/snap_{snap}/snap_{snap}.h5', 'r') as fileh:
            for i in range(len(box)):
                box[i] = fileh['Box'][i]
            fileh.close()
        # day = np.loadtxt(f'{pre}/snap_{snap}/tbytfb_{snap}.txt')
    else:
        X = np.load(f'{pre}/{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/{snap}/CMz_{snap}.npy')
        T = np.load(f'{pre}/{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}/{snap}/Rad_{snap}.npy')
        Vol = np.load(f'{pre}/{snap}/Vol_{snap}.npy')
        with h5py.File(f'{pre}/{snap}/snap_{snap}.h5', 'r') as fileh:
            for i in range(len(box)):
                box[i] = fileh['Box'][i]
            fileh.close()
    Rad_den = np.multiply(Rad,Den) # now you have enrgy density
    del Rad   

    R = np.sqrt(X**2 + Y**2 + Z**2)
    #%% Cross dot -----------------------------------------------------------------
    observers_xyz = hp.pix2vec(c.NSIDE, range(192))
    # Line 17, * is matrix multiplication, ' is .T
    observers_xyz = np.array(observers_xyz).T
    cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
    cross_dot[cross_dot<0] = 0
    cross_dot *= 4/192

    #%% Tree ----------------------------------------------------------------------
    #from scipy.spatial import KDTree
    xyz = np.array([X, Y, Z]).T
    tree = eng.KDTreeSearcher(xyz)
    N_ray = 5_000

    # Flux?
    F_photo = np.zeros((192, f_num))
    F_photo_temp = np.zeros((192, f_num))

    # Dynamic Box -----------------------------------------------------------------
    reds = np.zeros(192)
    for i in tqdm(range(192)):
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # Box is for dynamic ray making
        if mu_x < 0:
            rmax = box[0] / mu_x
            # print('x-', rmax)
        else:
            rmax = box[3] / mu_x
            # print('x+', rmax)
        if mu_y < 0:
            rmax = min(rmax, box[1] / mu_y)
            # print('y-', rmax)
        else:
            rmax = min(rmax, box[4] / mu_y)
            # print('y+', rmax)

        if mu_z < 0:
            rmax = min(rmax, box[2] / mu_z)
            # print('z-', rmax)
        else:
            rmax = min(rmax, box[5] / mu_z)
            # print('z+', rmax)

        r = np.logspace( -0.25, np.log10(rmax), N_ray)
        alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
        dr = alpha * r

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        # del x, y, z
        # tree = KDTree(xyz, leaf_size=50)
        # _, idx = tree.query(xyz2, k=1)
        idx = eng.knnsearch(tree, xyz2)
        idx = [ int(idx[i][0] - 1) for i in range(len(idx))] # -1 because we start from 0
        d = Den[idx] * c.den_converter
        t = T[idx]

        # Interpolate ----------------------------------------------------------
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2,np.log(t),np.log(d),'linear',0)
        sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
        sigma_rossland_eval = np.exp(sigma_rossland) 

        sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2,np.log(t),np.log(d),'linear',0)
        sigma_plank = [sigma_plank[0][i] for i in range(N_ray)]
        sigma_plank_eval = np.exp(sigma_plank)
        # del sigma_rossland, sigma_plank 
        # gc.collect()

        # Optical Depth ---------------------------------------------------------------
        # Okay, line 232, this is the hard one.
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_cgs # dont know what it do but this is the conversion
        k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
        los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_cgs
        
        # Red -----------------------------------------------------------------------
        # Get 20 unique, nearest neighbors
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        # _, idxnew = tree.query(xyz3, k=20)
        idxnew = eng.knnsearch(tree, xyz3, 'K', 20)
        idxnew = np.unique(idxnew).T
        idxnew = [ int(idxnew[i] -1) for i in range(len(idxnew))]
        dx = 0.5 * Vol[idx]**(1/3) # Cell radius

        # Get the Grads
        # sphere and get the gradient on them. Is it neccecery to re-interpolate?
        # scattered interpolant returns a function
        # griddata DEMANDS that you pass it the values you want to eval at
        f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T

        gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx]+dx, Y[idx], Z[idx]]).T )
        gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx]-dx, Y[idx], Z[idx]]).T )
        gradx = (gradx_p - gradx_m)/ (2*dx)

        gradx = np.nan_to_num(gradx, nan =  0)
        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
        grady = (grady_p - grady_m)/ (2*dx)

        grady = np.nan_to_num(grady, nan =  0)

        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
        # some nans here
        gradz_m = np.nan_to_num(gradz_m, nan =  0)
        gradz = (gradz_p - gradz_m)/ (2*dx)

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
        # del gradx, grady, gradz
        # gc.collect()
        
        R_lamda = grad / ( c.Rsol_cgs * sigma_rossland_eval* Rad_den[idx])
        R_lamda[R_lamda < 1e-10] = 1e-10
        fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have removed the minus
    
        try:
            b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] 
        except IndexError:
            b = 3117 # elad_b = 3117
        Lphoto2 = 4*np.pi*c.c*smoothed_flux[b] * c.Msol_cgs / (c.tsol_cgs**2)
        EEr = Rad_den[idx]
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
        max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_cgs * c.Rsol_cgs / (c.tsol_cgs**2)
        Lphoto = np.min( [Lphoto2, max_length])
        reds[i] = Lphoto
        # del smoothed_flux, R_lamda, fld_factor, EEr, los,
        # gc.collect()
    Lphoto_all[idx_s] = np.mean(reds) # save red

# Lphoto_all[idx_s] = 4*np.pi*np.mean(reds) # save red
    
if save:
    pre_saving = f'/data1/martirep/shocks/shock_capturing/data/{folder}/red'
    if check == '':
        check = 'Low'
    with open(f'{pre_saving}/{check}_eladred.txt', 'a') as file:
        file.write('# t/tfb \n' + ' '.join(map(str, tfb)) + '\n')
        file.write('# FLD \n' + ' '.join(map(str, Lphoto_all)) + '\n')
        file.close()
    print('saved dasy and red')
eng.exit()
