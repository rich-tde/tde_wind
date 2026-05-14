""" FLD curve accoring to Elad's script (MATLAB: start from 1 with indices, * is matrix multiplication, ' is .T). """
import sys
from tabnanny import check
sys.path.append('/Users/paolamartire/shocks')
# import resource
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    abspath = '/Users/paolamartire/shocks'
    save = False
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

import gc
import warnings
warnings.filterwarnings('ignore')
import csv

import numpy as np
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from src.Opacity.linextrapolator import opacity_extrap, opacity_linear
from scipy.ndimage import uniform_filter1d

import Utilities.prelude as prel
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.sections import make_slices
import src.orbits as orb
from Utilities.operators import make_tree


def opacity_extrap(x, y, K, which_opacity, xn, yn, scatter = None, slope_length = 7):
    ''' 
    Extra/Interpolation for opacity both in density and temperature.
    Look at: https://gitlab.com/eladtan/RICH/-/blob/master/source/misc/utils.cpp 
    https://gitlab.com/eladtan/RICH/-/blob/master/source/Radiation/MultigroupDiffusionCoefficientCalculator.cpp 
    Diffusion coefficient computed here: https://gitlab.com/eladtan/RICH/-/blob/master/source/Radiation/Diffusion.cpp 
    x: array of ln(T)
    y: array of ln(rho)
    K: array of ln(kappa) [1/cm]
    xn: array of ln(T) for the extrapolated grid
    yn: array of ln(rho) for the extrapolated grid
    scatter: either None or interpoalted scattering table in ln (with the same shape of K).
             if != None, brings to opacity always above scattering. It has to be applied for rosseland.
    slope_length - 1, int: position of the other point used for the slope.
    special_rho_slope, float: slope for some density extrapolations.
    highT_slope, float: slope for high temperature extrapolation.
    extrarowsx/extrarowsy, int: number of rows/columns to extrapolate.
    
    '''
    
    if which_opacity == 'planck':
        special_rho_slope = 2
        highT_slope = -3.5
    else:
        special_rho_slope = 1 
        highT_slope = 0
    Kn = np.zeros(len(xn))
    Kxslopes = np.zeros(len(xn))
    Kyslopes = np.zeros(len(xn))
    change_to_scatter = np.zeros(len(xn), dtype = bool)
    for ix, xsel in enumerate(xn):
        ysel = yn[ix]
        Kxslope = 0
        Kyslope = 0
        if xsel < x[0]: # Too cold
            deltax = x[slope_length - 1] - x[0]
            if ysel < y[0]: # Too rarefied
                Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                if which_opacity == 'planck':
                    Kyslope = special_rho_slope
                else:
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[0, slope_length -1] - K[0, 0]) / deltay
                # Kyslope = (K[0, slope_length -1] - K[0, 0]) / deltay
                Kn[ix] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
            
                if scatter is not None:
                    scatter_this_den = scatter[0][ix]
                    if Kn[ix] < scatter_this_den:
                        Kn[ix] = scatter_this_den
                        Kxslope = scatter[1][ix]
                        Kyslope = scatter[2][ix]
                        change_to_scatter[ix] = True

            elif ysel > y[-1]: # Too dense 
                deltay = y[-1] - y[-slope_length -1] 
                Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                Kyslope = special_rho_slope
                Kn[ix] = K[0, -1] + Kxslope * (xsel - x[0]) +  Kyslope * (ysel - y[-1])
            
            else: # Density is inside the table
                iy_inK = np.argmin(np.abs(y - ysel))
                Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                Kn[ix] = K[0, iy_inK] + Kxslope * (xsel - x[0])
        
        # Too hot
        elif xsel > x[-1]: 
            Kxslope = highT_slope 
            if ysel < y[0]: # Too rarefied
                if which_opacity == 'planck':
                    Kyslope = special_rho_slope
                else:
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[-1, slope_length -1] - K[-1, 0]) / deltay
                Kn[ix] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])
                
                if scatter is not None:
                    scatter_this_den = scatter[0][ix]
                    if Kn[ix] < scatter_this_den:
                        Kn[ix] = scatter_this_den
                        Kxslope = scatter[1][ix]
                        Kyslope = scatter[2][ix]
                        change_to_scatter[ix] = True

            elif ysel > y[-1]: # Too dense
                Kyslope = special_rho_slope 
                Kn[ix] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
            
            else: # Density is inside the table
                iy_inK = np.argmin(np.abs(y - ysel))
                Kn[ix] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
            
        else: # Temperature is inside table
            ix_inK = np.argmin(np.abs(x - xsel))
            if ysel < y[0]: # Too rarefied
                if which_opacity == 'planck':
                    deltay = y[10] - y[0]
                    Kyslope = (K[ix_inK, 10] - K[ix_inK, 0]) / deltay
                else:
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[ix_inK, slope_length -1] - K[ix_inK, 0]) / deltay
                if which_opacity == 'planck':
                    if Kyslope < 0.35 or Kyslope > 2.75:
                        # print('Weird in Planck opacity: Ky slope too high/low. I pass')
                        # raise UniversalError("Planck opacity interpolation failed")
                        pass
                Kn[ix] = K[ix_inK, 0] + Kyslope * (ysel - y[0])

                if scatter is not None:
                    scatter_this_den =  scatter[0][ix] # 1/cm
                    if Kn[ix] < scatter_this_den: 
                        Kn[ix] = scatter_this_den
                        Kxslope = scatter[1][ix]
                        Kyslope = scatter[2][ix]
                        change_to_scatter[ix] = True

            elif ysel > y[-1]:  # Too dense
                Kyslope = special_rho_slope
                Kn[ix] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])

            else:
                iy_inK = np.argmin(np.abs(y - ysel))
                Kn[ix] = K[ix_inK, iy_inK]
        
        Kyslopes[ix] = Kyslope
        Kxslopes[ix] = Kxslope

    return Kn, Kxslopes, Kyslopes, change_to_scatter

def opacity_linear(x, y, K, xn, yn, slope_length = 7, highT_slope = 0):
    ''' 
    Extra/Interpolation for temperature, linear with slope = 1 for density. 
    It's used for scattering and in some runs for rosseland.
    Look at:
    - https://gitlab.com/eladtan/RICH/-/blob/master/source/misc/utils.cpp 
    - CalcDiffusionCoefficient, which gives you the inverse of Rosseland in https://gitlab.com/eladtan/RICH/-/blob/master/source/Radiation/STAgreyOpacity.cpp 
    x: array of ln(T)
    y: array of ln(rho)
    K: array of ln(kappa) [1/cm]
    scatter: either None or interpoalted scattering table in ln(with the same shape of K).
             if != None, brings to opacity always above scattering. It has to be applied for rosseland.
    slope_length, int: position of the other point used for the slope.
    highT_slope, float: slope for high temperature extrapolation.
    extrarowsx/extrarowsy, int: number of rows/columns to extrapolate.
    '''    
    Kn = np.zeros(len(xn))
    Kxslopes = np.zeros(len(xn))
    Kyslopes = np.zeros(len(xn))

    for ix, xsel in enumerate(xn):
        ysel = yn[ix]
        Kxslope = 0
        Kyslope = 0
        if xsel < x[0]: # Too cold
            deltax = x[slope_length - 1] - x[0]
            if ysel < y[0]: # Too rarefied
                Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                Kn[ix] = K[0, 0] + Kxslope * (xsel - x[0]) + (ysel - y[0])
                Kyslope = 1
            elif ysel > y[-1]: # Too dense
                Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                Kn[ix] = K[0, -1] + Kxslope * (xsel - x[0]) + (ysel - y[-1])
                Kyslope = 1
            else: # Density is inside the table
                iy_inK = np.argmin(np.abs(y - ysel))
                Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                Kn[ix] = K[0, iy_inK] + Kxslope * (xsel - x[0])
        
        # Too hot
        elif xsel > x[-1]: 
            if ysel < y[0]: # Too rarefied
                Kxslope = highT_slope #(K[-1, 0] - K[-slope_length - 1, 0]) / deltax
                Kn[ix] = K[-1, 0] + Kxslope * (xsel - x[-1]) + (ysel - y[0])
                Kyslope = 1
            elif ysel > y[-1]: # Too dense
                Kxslope = highT_slope #(K[-1, -1] - K[-slope_length - 1, -1]) / deltax
                Kn[ix] = K[-1, -1] + Kxslope * (xsel - x[-1]) + (ysel - y[-1])
                Kyslope = 1
            else: # Density is inside the table
                iy_inK = np.argmin(np.abs(y - ysel))
                Kxslope = highT_slope #(K[-1, iy_inK] - K[-slope_length - 1, iy_inK]) / deltax
                Kn[ix] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])

        else: 
            ix_inK = np.argmin(np.abs(x - xsel))
            if ysel < y[0]: # Too rarefied, Temperature is inside table
                Kn[ix] = K[ix_inK, 0] + (ysel - y[0])
                Kyslope = 1
                
            elif ysel > y[-1]:  # Too dense, Temperature is inside table
                Kn[ix] = K[ix_inK, -1] + (ysel - y[-1])
                Kyslope = 1

            else:
                iy_inK = np.argmin(np.abs(y - ysel))
                Kn[ix] = K[ix_inK, iy_inK]
        
        Kxslopes[ix] = Kxslope
        Kyslopes[ix] = Kyslope

    return Kn, Kxslopes, Kyslopes

def fld_lightcurve(params, compton, check, N_ray):
    m, Rstar, mstar, beta, n, compton = params
    Mbh = 10**m
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) 
    pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
    pre_saving = f'{abspath}/data/{folder}'

    # observers 
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) # shape: (3, 192)
    observers_xyz = np.array(observers_xyz).T # shape: (192, 3)

    for idx_s, snap in enumerate(snaps):
        if snap not in [151]: # for testing
            continue
        print(f'Snap: {snap}', flush=True)
        if alice:
            loadpath = f'{pre}/snap_{snap}'
        else:
            loadpath = f'{pre}/{snap}'
        data = make_tree(loadpath, snap)
        Lphoto_snap, photosphere, colorsphere, freqs, L_col = single_fld(loadpath, snap, observers_xyz, N_ray)
        data = [snap, tfb[idx_s], Lphoto_snap]
        if save:
            with open(f'{pre_saving}/{check}_red.csv', 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()

            np.savez(f"{pre_saving}/photonew/{check}_photo{snap}.npz", **photosphere)
            # Save spectrum
            np.savez(f"{pre_saving}/spectranew/{check}_Rcol{snap}.npz", **colorsphere)
            np.savetxt(f'{pre_saving}/spectranew/freqs.txt', freqs)
            np.savetxt(f'{pre_saving}/spectranew/{check}_spectra{snap}.txt', L_col)
        del Lphoto_snap, photosphere, colorsphere, L_col
        gc.collect()


def single_fld(loadpath, snap, observers_xyz, N_ray):
    num_obs = len(observers_xyz)
    data = make_tree(loadpath, snap)
    box = np.load(f'{loadpath}/box_{snap}.npy')
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        data.X, data.Y, data.Z, data.Temp, data.Den, data.Rad, data.Vol, data.VX, data.VY, data.VZ, data.Press, data.IE
    denmask = Den > 1e-19
    X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den = \
        make_slices([X, Y, Z, T, Den, Rad_den, Vol, VX, VY, VZ, Press, IE_den], denmask)
    xyz = np.array([X, Y, Z]).T
    R = np.sqrt(X**2 + Y**2 + Z**2)

    photosphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': [], 'ieden': [], 'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'r': [], 'Lum': [], 'Fx': [], 'Fy': [], 'Fz': []}
    colorsphere = {'idx': [], 'x': [], 'y': [], 'z': [], 'vol': [], 'den': [], 'temp': [], 'radden': [], 'vx': [], 'vy': [], 'vz': [], 'P': [], 'ieden': [], 'alpha_rossland': [], 'alpha_scatter': [], 'alpha_abs': [], 'alpha_eff': []}
    freqs = prel.freqs
    L_col = np.zeros((num_obs, len(prel.freqs)))
    for i in range(num_obs):
        if i not in [0, 100]:
            continue
        print(f'Obs: {i}', flush=True)

        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # be sure is normalized, but it should be because it's a healpy vector
        norm_mu = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
        if norm_mu != 1.:
            print('normalizing observers direction. norm:', norm_mu, flush=True)
            mu_x /= norm_mu
            mu_y /= norm_mu
            mu_z /= norm_mu

        if mu_x < 0:
            rmax = box[0] / mu_x
        else:
            rmax = box[3] / mu_x
        if mu_y < 0:
            rmax = min(rmax, box[1] / mu_y)
        else:
            rmax = min(rmax, box[4] / mu_y)
        if mu_z < 0:
            rmax = min(rmax, box[2] / mu_z)
        else:
            rmax = min(rmax, box[5] / mu_z)

        r = np.logspace(-0.25, np.log10(rmax), N_ray)

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        # find the simulation cell corresponding to cells in the wanted ray
        tree = KDTree(xyz, leaf_size=50) 
        _, idx = tree.query(xyz2, k=1)
        idx = idx.ravel()
        # idx = np.unique(idx).astype(np.int64) #####
        # Quantity corresponding to the ray
        d = Den[idx] * prel.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        ray_radDen = Rad_den[idx]
        volume = Vol[idx]
        ray_vx = VX[idx]
        ray_vy = VY[idx]
        ray_vz = VZ[idx]
        ray_press = Press[idx]
        ray_ie_den = IE_den[idx]
        # r = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2) ####
        
        # Interpolate opacity 
        ln_alpha_scatter, A_scatt, B_scatt = opacity_linear(T_cool, Rho_cool, scattering, np.log(t), np.log(d))
        ln_alpha_rossland, A_ross, B_ross, change_to_scatter_ross = opacity_extrap(T_cool, Rho_cool, rossland, 'rossland', np.log(t), np.log(d), scatter = [ln_alpha_scatter, A_scatt, B_scatt])
        ln_alpha_planck, A_planck, B_planck, _ = opacity_extrap(T_cool, Rho_cool, planck, 'planck', np.log(t), np.log(d), scatter = None)
        # ln_alpha_rossland = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d), 'linear', 0)
        # ln_alpha_rossland = np.array(ln_alpha_rossland)[0]
        # ln_alpha_planck = eng.interp2(T_cool2, Rho_cool2, planck2.T, np.log(t), np.log(d), 'linear', 0)
        # ln_alpha_planck = np.array(ln_alpha_planck)[0]
        # ln_alpha_scatter = eng.interp2(T_cool2, Rho_cool2, scatter2.T, np.log(t), np.log(d), 'linear', 0)
        # ln_alpha_scatter = np.array(ln_alpha_scatter)[0]
        underflow_mask = np.logical_and(np.logical_and(ln_alpha_rossland != 0.0, ln_alpha_planck != 0.0), ln_alpha_scatter != 0.0)
        d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ln_alpha_planck, ln_alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx = \
            make_slices([d, t, r, ray_x, ray_y, ray_z, ln_alpha_rossland, ln_alpha_planck, ln_alpha_scatter, ray_radDen, volume, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den, idx], underflow_mask)
        idx = np.array(idx)
        alpha_rossland = np.exp(ln_alpha_rossland) # [1/cm]
        alpha_planck = np.exp(ln_alpha_planck) # [1/cm]
        alpha_scatter = np.exp(ln_alpha_scatter) # [1/cm]
        # del ln_alpha_rossland, ln_alpha_planck, ln_alpha_scatter
        # gc.collect()

        # Optical depth
        r_fuT = np.flipud(r) #.T
        alpha_rossland_fuT = np.flipud(alpha_rossland) 
        # compute the optical depth from the outside in: tau = - int kappa dr. Then reverse the order to have it from the inside to out, so can query.
        los = - np.flipud(sci.cumulative_trapezoid(alpha_rossland_fuT, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r
        
        # alpha_effective = np.sqrt(3 * alpha_planck * (alpha_planck + alpha_scatter))
        alpha_effective = np.sqrt(3 * alpha_planck * alpha_rossland)
        alpha_effective_fuT = np.flipud(alpha_effective)
        los_effective = - np.flipud(sci.cumulative_trapezoid(alpha_effective_fuT, 
                                                         r_fuT, initial = 0)) * prel.Rsol_cgs
        los_effective[los_effective > 30] = 30

        # FLD curve 
        # Get 20 unique nearest neighbors to each cell in the wanted ray and use them to compute the gradient along the ray
        xyz3 = np.array([ray_x, ray_y, ray_z]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew) #.T
        dx = 0.5 * volume**(1/3) # Cell radius 
        f_inter_input = np.array([X[idxnew], Y[idxnew], Z[idxnew]]).T

        gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x+dx, ray_y, ray_z]).T )
        gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x-dx, ray_y, ray_z]).T )
        gradx = (gradx_p - gradx_m)/ (2*dx)
        gradx = np.nan_to_num(gradx, nan =  0)
        del gradx_p, gradx_m

        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y+dx, ray_z]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y-dx, ray_z]).T )
        grady = (grady_p - grady_m)/ (2*dx)
        grady = np.nan_to_num(grady, nan =  0)
        del grady_p, grady_m

        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y, ray_z+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y, ray_z-dx]).T )
        gradz = (gradz_p - gradz_m)/ (2*dx)
        gradz = np.nan_to_num(gradz, nan =  0)
        del gradz_p, gradz_m

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2) # grad = |grad|
        # gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz) # projection of the gradient along the radial direction
        gc.collect()

        # Eq.(28) from Krumholz07.
        R_lamda = grad / ( prel.Rsol_cgs * alpha_rossland* ray_radDen) # this is the conversion for /r from the gradient. It's dimensionless
        R_lamda[R_lamda < 1e-10] = 1e-10
        # Eq.(27) from Krumholz07.
        fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        # Eq.(26) from Krumholz07. 
        Fx = -prel.c_cgs * fld_factor * gradx * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland # CGS
        Fy = -prel.c_cgs * fld_factor * grady * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland
        Fz = -prel.c_cgs * fld_factor * gradz * (prel.en_den_converter/prel.Rsol_cgs)/ alpha_rossland
        Fr = (mu_x * Fx) + (mu_y*Fy) + (mu_z*Fz)
        # smoothed_flux_r2 = -prel.c_cgs * uniform_filter1d(r**2 * fld_factor * gradr / alpha_rossland, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        smoothed_flux_r2 = uniform_filter1d(r**2 * Fr, 7) #r^2 is here (but it's for the flux) otherwise you get annoying errors in the if. 
        del gradx, grady, gradz

        try: 
            photo_idx = np.where( ((smoothed_flux_r2>0) & (los<2/3) ))[0][0] 
        except IndexError: # if you don't find the photosphere, skip the observer
            print(f'No photosphere found for observer {i}', flush=True)
            continue
        # Lphoto2 = 4*np.pi * smoothed_flux_r2[photosphere] * prel.Msol_cgs / (prel.tsol_cgs**2) # you have to convert ray_radDen*r^2/lenght = energy/lenght^2 = mass/time^2
        Lphoto2 = 4*np.pi * smoothed_flux_r2[photo_idx] * prel.Rsol_cgs**2 # you have to convert the r^2 in smoothed_flux_r2
        if Lphoto2 < 0: 
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        # free streaming emission
        max_length = 4*np.pi*(r[photo_idx]**2) * prel.c_cgs * ray_radDen[photo_idx] * prel.Msol_cgs * prel.Rsol_cgs / (prel.tsol_cgs**2) #the conversion is for ray_radDen*r^2 = mass*len/time^2
        Lphoto = np.min( [Lphoto2, max_length]) #that's usually Lphoto2
        photosphere['idx'].append(idx[photo_idx])
        photosphere['x'].append(ray_x[photo_idx])
        photosphere['y'].append(ray_y[photo_idx])
        photosphere['z'].append(ray_z[photo_idx])
        photosphere['vol'].append(volume[photo_idx])
        photosphere['den'].append(d[photo_idx])
        photosphere['temp'].append(t[photo_idx])
        photosphere['radden'].append(ray_radDen[photo_idx])
        photosphere['vx'].append(ray_vx[photo_idx])
        photosphere['vy'].append(ray_vy[photo_idx])
        photosphere['vz'].append(ray_vz[photo_idx])
        photosphere['P'].append(ray_press[photo_idx])
        photosphere['ieden'].append(ray_ie_den[photo_idx])
        photosphere['alpha_rossland'].append(alpha_rossland[photo_idx])
        photosphere['alpha_scatter'].append(alpha_scatter[photo_idx])
        photosphere['alpha_abs'].append(alpha_planck[photo_idx])
        photosphere['r'].append(r[photo_idx])
        photosphere['Fx'].append(Fx[photo_idx])
        photosphere['Fy'].append(Fy[photo_idx])
        photosphere['Fz'].append(Fz[photo_idx])
        photosphere['Lum'].append(Lphoto) # fluxes was from here as L/4pi r[photo)idx]**2

        # Spectra
        color_idx = np.argmin(np.abs(los_effective-5))
        colorsphere['idx'].append(idx[color_idx])
        colorsphere['x'].append(ray_x[color_idx])
        colorsphere['y'].append(ray_y[color_idx])
        colorsphere['z'].append(ray_z[color_idx])
        colorsphere['vol'].append(volume[color_idx])
        colorsphere['den'].append(d[color_idx])
        colorsphere['temp'].append(t[color_idx])
        colorsphere['radden'].append(ray_radDen[color_idx])
        colorsphere['vx'].append(ray_vx[color_idx])
        colorsphere['vy'].append(ray_vy[color_idx])
        colorsphere['vz'].append(ray_vz[color_idx])
        colorsphere['P'].append(ray_press[color_idx])
        colorsphere['ieden'].append(ray_ie_den[color_idx])
        colorsphere['alpha_rossland'].append(alpha_rossland[color_idx])
        colorsphere['alpha_scatter'].append(alpha_scatter[color_idx])
        colorsphere['alpha_abs'].append(alpha_planck[color_idx])
        colorsphere['alpha_eff'].append(alpha_effective[color_idx])

        # Spectra ---
        for k in range(color_idx, len(r)):
            # if k == 0:
            #     continue
            dr = r[k]-r[k-1]
            Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
            wien = np.exp(prel.h_cgs * freqs / (prel.Kb_cgs * t[k])) - 1
            black_body = freqs**3 / wien # There should be a 2 * prel.h_cgs/c^2, but it doesn't matter because we normalize. BB udm: erg/s/cm^2/Hz/ster. 
            L_col[i,:] += alpha_planck[k] * Vcell * np.exp(-los_effective[k]) * black_body # erg/s/Hz.
        
        norm = Lphoto / np.trapezoid(L_col[i,:], freqs)
        L_col[i,:] *= norm

        if plot:
            apo = 330
            trad = (ray_radDen * prel.en_den_converter/prel.alpha_cgs)**(1/4)
            kappa_ross = alpha_rossland/d
            kappa_planck = alpha_planck/d
            # space = 1e-2
            # ln_alpha_rossland_downT = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t-space), np.log(d), 'linear', 0)
            # ln_alpha_rossland_downT = np.array(ln_alpha_rossland_downT)[0]
            # ln_alpha_rossland_upT = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t+space), np.log(d), 'linear', 0)
            # ln_alpha_rossland_upT = np.array(ln_alpha_rossland_upT)[0]
            # ln_alpha_rossland_downd = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d-space), 'linear', 0)
            # ln_alpha_rossland_downd = np.array(ln_alpha_rossland_downd)[0]
            # ln_alpha_rossland_upd = eng.interp2(T_cool2, Rho_cool2, rossland2.T, np.log(t), np.log(d+space), 'linear', 0)
            # ln_alpha_rossland_upd = np.array(ln_alpha_rossland_upd)[0]
            
            # ln_alpha_planck_downT = eng.interp2(T_cool2, Rho_cool2, planck2.T, np.log(t-space), np.log(d), 'linear', 0)
            # ln_alpha_planck_downT = np.array(ln_alpha_planck_downT)[0]
            # ln_alpha_planck_upT = eng.interp2(T_cool2, Rho_cool2, planck2.T, np.log(t+space), np.log(d), 'linear', 0)
            # ln_alpha_planck_upT = np.array(ln_alpha_planck_upT)[0]
            # ln_alpha_planck_downd = eng.interp2(T_cool2, Rho_cool2, planck2.T, np.log(t), np.log(d-space), 'linear', 0)
            # ln_alpha_planck_downd = np.array(ln_alpha_planck_downd)[0]
            # ln_alpha_planck_upd = eng.interp2(T_cool2, Rho_cool2, planck2.T, np.log(t), np.log(d+space), 'linear', 0)
            # ln_alpha_planck_upd = np.array(ln_alpha_planck_upd)[0]

            # A_ross = np.diff(ln_alpha_rossland)/np.diff(np.log(t))
            # B_ross = np.diff(ln_alpha_rossland)/np.diff(np.log(d))
            # A_planck = np.diff(ln_alpha_planck)/np.diff(np.log(t))
            # B_planck = np.diff(ln_alpha_planck)/np.diff(np.log(d))
            
            fig, ((axk, axtau), (axA, axB), (axT, axd)) = plt.subplots(3, 2, figsize = (15, 21))
            axk.plot(r/apo, alpha_rossland, c = 'dodgerblue', label = r'$\alpha_{\rm R}$')
            axk.plot(r/apo, alpha_planck, c = 'r', label = r'$\alpha_{\rm a}$')
            axk.set_ylabel(r'$\alpha$ [1/cm]')
            axk.set_ylim(1e-15, 1e-4)
            axk.legend(fontsize = 18)

            axtau.plot(r/apo, los, c = 'dodgerblue', label = r'$\tau_{\rm R}$')
            axtau.axhline(2/3, c = 'gray', ls = '--')
            axtau.plot(r/apo, los_effective, c = 'r', label = r'$\tau_{\rm eff}$')
            axtau.axhline(5, c = 'gray', ls = '--')
            axtau.set_ylabel(r'$\tau$')
            axtau.set_ylim(1e-4, 1e2)

            axT.plot(r/apo, t, c = 'k', label = r'T')
            axT.plot(r/apo, trad, ls = '--', c = 'gray', label = r'T$_{\rm rad}$')
            axT.set_ylabel(r'T [K]')
            axT.set_ylim(1e3, 2e7)

            axd.plot(r/apo, d, c = 'k')
            axd.set_ylabel(r'Den [g/cm$^3$]')
            axd.set_ylim(1e-18, 1e-12)

            axA.scatter(r/apo, A_ross, c = 'dodgerblue', s = 50, label = r'$\Delta \ln\alpha_{\rm R}/\Delta \ln T$')
            axA.scatter(r/apo, A_planck, c = 'r', s = 25, label = r'$\Delta \ln\alpha_{\rm a}/\Delta \ln T$')
            axA.set_ylabel(r'coeff extrap T')
            # axA.set_ylim(-10, 10)

            axB.scatter(r/apo, B_ross, c = 'dodgerblue', s = 50, label = r'$\Delta \ln\alpha_{\rm R}/\Delta \ln \rho$')
            axB.scatter(r/apo, B_planck, c = 'r', s = 25, label = r'$\Delta \ln\alpha_{\rm a}/\Delta \ln \rho$')
            axB.set_ylabel(r'coeff extrap $\rho$')
            # axB.set_ylim(-10, 10)

            for ax in [axk, axtau, axd, axT, axA, axB]:
                ax.set_xlim(5e-2, 15)
                if ax in [axA, axB]:
                    ax.set_xscale('log')
                else:
                    ax.loglog()
                ax.axvline(r[photo_idx]/apo, ls = '--', c = 'dodgerblue', label = f'obs {i}, ' +  r'$r_{\rm ph}$')
                ax.axvline(r[color_idx]/apo, ls = '--', c = 'r', label = f'obs {i}, ' +  r'$r_{\rm col}$')
                ax.grid()
                ax.legend(fontsize = 18)
                ax.tick_params(axis='both', which='major',length=10, width=1.5)
                ax.tick_params(axis='both', which='minor',length=5, width=1)
                if ax in [axT, axd, axA, axB]:
                    ax.set_xlabel(r'$r/r_{\rm a}$')
                    ax.set_xlabel(r'$r/r_{\rm a}$')
            plt.tight_layout() 
            # plt.savefig(f'{abspath}/Figs/{folder}/Test/{snap}/alphai_{snap}_{i}.png')
            # plt.close()

        del smoothed_flux_r2, R_lamda, fld_factor, ray_radDen, alpha_rossland, alpha_planck, alpha_scatter, los, los_effective, tree, idxnew, f_inter_input, volume, ray_x, ray_y, ray_z, ray_vx, ray_vy, ray_vz, ray_press, ray_ie_den 
        gc.collect()

    Lphoto_snap = np.mean(photosphere['Lum'])
    print('L :', Lphoto_snap, flush=True)

    return Lphoto_snap, photosphere, colorsphere, freqs, L_col

if __name__ == "__main__":
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    N_ray = 5_000
    params = [m, Rstar, mstar, beta, n, compton]
    
    # Load opacity tables
    opac_path = f'{abspath}/src/Opacity'
    T_cool = np.loadtxt(f'{opac_path}/T.txt')
    Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
    rossland = np.loadtxt(f'{opac_path}/ross.txt')
    planck = np.loadtxt(f'{opac_path}/planck.txt')
    scattering = np.loadtxt(f'{opac_path}/scatter.txt') # 1/cm

    #MATLAB, thanks Cindy.
    eng = matlab.engine.start_matlab()

    fld_lightcurve(params, compton, check, N_ray)