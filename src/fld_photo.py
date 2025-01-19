""" Find photosphere. 
If when == 'fromfld', it will use the data (using the idx of the cells) from fld_curve.py to calculate the photo. 
If when == 'justph', it will compute here the photosphere (just X,Y,Z) as it does in fld_curve.py. 
"""
import sys
sys.path.append('/Users/paolamartire/shocks')
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
import matlab.engine

import numpy as np
# import h5py
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import extrapolator_flipper, rich_extrapolator


import Utilities.prelude as prel
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
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
check = 'LowRes' 
extr = 'rich'
how = 'fromfld' #'justph' or 'fromfld'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
pre_saving = f'{abspath}/data/{folder}'
if how == 'fromfld':
    # Load the data
    # alldata_ph = np.loadtxt(f'{pre_saving}/{check}{extr}_phidx.txt')
    alldata_ph = np.loadtxt(f'{pre_saving}/{check}{extr}_phidx_fluxes.txt')
    snaps_ph, alltimes_ph, allindices_ph = alldata_ph[:, 0], alldata_ph[:, 1], alldata_ph[:, 2:]
    snaps_ph = np.array(snaps_ph)

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

# if extr == '':
#     T_cool2, Rho_cool2, rossland2 = extrapolator_flipper(T_cool, Rho_cool, rossland)
if extr == 'rich':
    T_cool2, Rho_cool2, rossland2 = rich_extrapolator(T_cool, Rho_cool, rossland)

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)

for idx_s, snap in enumerate(snaps):
    print('\n Snapshot: ', snap, '\n')
    if how == 'justph':
        if int(snap)!=80:
            continue
        eng = matlab.engine.start_matlab()
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
        Rad_den = np.multiply(Rad,Den) # now you have enrgy density
        del Rad   
        R = np.sqrt(X**2 + Y**2 + Z**2)
        #%% Cross dot -----------------------------------------------------------------
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

        # Flux?
        F_photo = np.zeros((prel.NPIX, f_num))
        F_photo_temp = np.zeros((prel.NPIX, f_num))

        # Dynamic Box -----------------------------------------------------------------
        reds = np.zeros(prel.NPIX)
        xphot = np.zeros(prel.NPIX) ####
        yphot = np.zeros(prel.NPIX) ####
        zphot = np.zeros(prel.NPIX) 
        rphot = np.zeros(prel.NPIX)
        time_start = 0
        for i in range(prel.NPIX):
            # Progress 
            time_end = time.time()
            print(f'Snap: {snap}, Obs: {i}', flush=False)
            print(f'Time for prev. Obs: {(time_end - time_start)/60} min', flush = False)
            time_start = time.time()
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
            xobs, yobs, zobs = X[idx], Y[idx], Z[idx]

            # Interpolate ----------------------------------------------------------
            sigma_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
            sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
            sigma_rossland_eval = np.exp(sigma_rossland) 

            # sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2,np.log(t),np.log(d),'linear',0)
            # sigma_plank = [sigma_plank[0][i] for i in range(N_ray)]
            # sigma_plank_eval = np.exp(sigma_plank)
            del sigma_rossland#, sigma_plank 
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
            
            xphot[i], yphot[i], zphot[i] = xobs[b], yobs[b], zobs[b]
            rphot[i] = r[b]
            gc.collect()

        # Save red of the single snap
        if save:
            data = [xphot, yphot, zphot]
            np.savetxt(f'{pre_saving}/{check}{how}_photo{snap}.txt', data)

            np.savetxt(f'{pre_saving}/{check}{how}_Rphoto{snap}.txt', rphot)

        eng.exit()

    if how == 'fromfld':
        from Utilities.operators import make_tree
        # Find the line corresponding to the snap and take the indices of the photosphere radii 
        # selected_idx = np.argmin(np.abs(snaps_ph - snap))
        selected_lines = np.concatenate(np.where(snaps_ph == snap))
        selected_idx, selected_fluxes = selected_lines[0], selected_lines[1]
        single_indices_ph = allindices_ph[selected_idx]
        single_indices_ph = single_indices_ph.astype(int)
        # Load the data
        path = f'/home/martirep/data_pi-rossiem/TDE_data/{folder}/snap_{snap}'
        datasnap = make_tree(path, snap, energy = False)
        x, y, z, vol, den, Temp = datasnap.X, datasnap.Y, datasnap.Z, datasnap.Vol, datasnap.Den, datasnap.Temp
        cut = den > 1e-19
        x, y, z, vol, den, Temp = make_slices([x, y, z, vol, den, Temp], cut)
        xph, yph, zph, volph, denph, Tempph = make_slices([x, y, z, vol, den, Temp], single_indices_ph)
        # save the photosphere
        with open(f'{pre_saving}/photo/{check}{how}_photo{snap}.txt', 'a') as f:
            f.write('# Data for the photospere. Lines are: xph, yph, zph, volph, denph, Tempph \n')
            f.write(' '.join(map(str, xph)) + '\n')
            f.write(' '.join(map(str, yph)) + '\n')
            f.write(' '.join(map(str, zph)) + '\n')
            f.write(' '.join(map(str, volph)) + '\n')
            f.write(' '.join(map(str, denph)) + '\n')
            f.write(' '.join(map(str, Tempph)) + '\n')
            f.close()
