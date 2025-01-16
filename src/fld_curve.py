""" FLD curve accoring to Elad's script. 
Written to be run on alice."""
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks/'

import sys
sys.path.append(abspath)

import gc
import time
import warnings
warnings.filterwarnings('ignore')
import csv

import numpy as np
# import h5py
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import extrapolator_flipper, rich_extrapolator
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up


import Utilities.prelude as prel
from Utilities.operators import make_tree
# from Utilities.parser import parse
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices


#%% Choose parameters -----------------------------------------------------------------
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'LowRes' # '' or 'HiRes'
extr = 'rich' #'rich' or ''

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
save = True
snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
Lphoto_all = np.zeros(len(snaps))

#%% Opacities -----------------------------------------------------------------
# Freq range
f_min = prel.Kb_cgs * 1e3 / prel.h_cgs
f_max = prel.Kb_cgs * 3e13 / prel.h_cgs
f_num = 1_000
frequencies = np.logspace(np.log10(f_min), np.log10(f_max), f_num)

# Opacity Input
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

if extr == '':
    T_cool2, Rho_cool2, rossland2 = extrapolator_flipper(T_cool, Rho_cool, rossland)
if extr == 'rich':
    T_cool2, Rho_cool2, rossland2 = rich_extrapolator(T_cool, Rho_cool, rossland)

# MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
print('we are in: ', pre)
for idx_s, snap in enumerate(snaps):
    print('\n Snapshot: ', snap, '\n')
    box = np.zeros(6)
    # Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/snap_{snap}/CMz_{snap}.npy')
        T = np.load(f'{pre}/snap_{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/snap_{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}/snap_{snap}/Rad_{snap}.npy')
        Vol = np.load(f'{pre}/snap_{snap}/Vol_{snap}.npy')
        box = np.load(f'{pre}/snap_{snap}/box_{snap}.npy')
    else:
        X = np.load(f'{pre}/{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}/{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}/{snap}/CMz_{snap}.npy')
        T = np.load(f'{pre}/{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}/{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}/{snap}/Rad_{snap}.npy')
        Vol = np.load(f'{pre}/{snap}/Vol_{snap}.npy')
        box = np.load(f'{pre}/{snap}/box_{snap}.npy')
    
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad, Vol = make_slices([X, Y, Z, T, Den, Rad, Vol], denmask)
    Rad_den = np.multiply(Rad,Den) # now you have energy density
    del Rad   
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # Cross dot -----------------------------------------------------------------
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
    # Line 17, * is matrix multiplication, ' is .T
    observers_xyz = np.array(observers_xyz).T
    cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
    cross_dot[cross_dot<0] = 0
    cross_dot *= 4/prel.NPIX

    # Tree ----------------------------------------------------------------------
    #from scipy.spatial import KDTree
    xyz = np.array([X, Y, Z]).T
    N_ray = 5_000

    # Flux
    F_photo = np.zeros((prel.NPIX, f_num))
    F_photo_temp = np.zeros((prel.NPIX, f_num))

    # Dynamic Box -----------------------------------------------------------------
    reds = np.zeros(prel.NPIX)
    ## just to check photosphere
    ph_idx = np.zeros(prel.NPIX)
    fluxes = np.zeros(prel.NPIX)
    ##
    for i in range(prel.NPIX):
        # Progress 
        print(f'Snap: {snap}, Obs: {i}', flush=False)
        sys.stdout.flush()

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
        del x, y, z
        tree = KDTree(xyz, leaf_size=50)
        _, idx = tree.query(xyz2, k=1)
        idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
        d = Den[idx] * prel.den_converter
        t = T[idx]

        # Interpolate ----------------------------------------------------------
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T,np.log(t),np.log(d),'linear',0)
        sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
        sigma_rossland_eval = np.exp(sigma_rossland) 
        del sigma_rossland
        gc.collect()

        # Optical Depth ---------------------------------------------------------------
        # Okay, line 232, this is the hard one.
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        # Red -----------------------------------------------------------------------
        # Get 20 unique, nearest neighbors
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew).T
        dx = 0.5 * Vol[idx]**(1/3) # Cell radius #the constant should be 0.62

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
        del gradx_p, gradx_m

        gradx = np.nan_to_num(gradx, nan =  0)
        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
        grady = (grady_p - grady_m)/ (2*dx)
        del grady_p, grady_m

        grady = np.nan_to_num(grady, nan =  0)

        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
        # some nans here
        gradz_m = np.nan_to_num(gradz_m, nan =  0)
        gradz = (gradz_p - gradz_m)/ (2*dx)
        del gradz_p, gradz_m

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
        del gradx, grady, gradz
        gc.collect()

        R_lamda = grad / ( prel.Rsol_cgs * sigma_rossland_eval* Rad_den[idx])
        R_lamda[R_lamda < 1e-10] = 1e-10
        fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have remov
        
        try:
            b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] 
        except IndexError:
            print('No b found, observer ', i)
            b = 3117 # elad_b = 3117
        Lphoto2 = 4*np.pi*prel.c_cgs*smoothed_flux[b] * prel.Msol_cgs / (prel.tsol_cgs**2)
        EEr = Rad_den[idx]
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        max_length = 4*np.pi*prel.c_cgs*EEr[b]*r[b]**2 * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for Erad: energy*r^2/lenght^3 [in SI would be kg m^2/s^2 * m^2 * 1/m^3]
        Lphoto = np.min( [Lphoto2, max_length])
        reds[i] = Lphoto
        ## just to check photosphere
        ph_idx[i] = idx[b]
        fluxes[i] = Lphoto / (4*np.pi*(r[b]*prel.Rsol_cgs)**2)
        ##
        del smoothed_flux, R_lamda, fld_factor, EEr, los,
        gc.collect()
    Lphoto_snap = np.mean(reds)
    Lphoto_all[idx_s] = Lphoto_snap # save red
    # Lphoto_all[idx_s] = 4*np.pi*np.mean(reds) # save red

    # Save red of the single snap
    if save:
        pre_saving = f'{abspath}/data/{folder}'
        # data = [snap, tfb[idx_s], Lphoto_snap]
        # with open(f'{pre_saving}/{check}{extr}_red.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(data)
        # file.close()

        ## just to check photosphere
        time_rph = np.concatenate([[snap,tfb[idx_s]], ph_idx])
        time_fluxes = np.concatenate([[snap,tfb[idx_s]], fluxes])
        with open(f'{pre_saving}/{check}{extr}_phidx_fluxes.txt', 'a') as fileph:
            fileph.write(f'# {folder}_{check}. First data is snap, second time (in t_fb), the rest are the photosphere indices \n')
            fileph.write(' '.join(map(str, time_rph)) + '\n')
            fileph.write(f'# {folder}_{check}. First data is snap, second time (in t_fb), the rest are the fluxes [cgs] for each obs \n')
            fileph.write(' '.join(map(str, time_fluxes)) + '\n')
            fileph.close()
        ##
eng.exit()
