""" Find photosphere following FLD curve Elad's script. 
Consider Healpix at the orbital plane.
Check that we are not sensitive from where we start."""
#%%
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
else:
    abspath = '/Users/paolamartire/shocks'
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm

import sys
sys.path.append(abspath)

import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
import healpy as hp
from scipy.ndimage import uniform_filter1d
import Utilities.prelude as prel
import scipy.integrate as spi
import scipy.optimize as spo
from src.Opacity.linextrapolator import nouveau_rich
from Utilities.selectors_for_snap import select_prefix
from Utilities.sections import make_slices
import src.orbits as orb

##
# MAIN
##
save = True

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = '' # '' or 'HiRes'

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
snap = 237
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
e_mb = orb.eccentricity(Rstar, mstar, Mbh, beta)
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
Rt = Rstar * (Mbh/mstar)**(1/3)

# Opacity Input (they are ln)
opac_path = f'{abspath}/src/Opacity'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland)

#%% MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

pre = select_prefix(m, check, mstar, Rstar, beta, n, compton)
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

#%% Observers -----------------------------------------------------------------
observers_xyz = np.array(hp.pix2vec(prel.NSIDE, range(prel.NPIX))) # shape is 3,N
# select only the observers in the orbital plane (will give you a N bool array--> apply to columns)
# mid = np.abs(observers_xyz[2]) == 0 # you can do that beacuse healpix gives you the observers also in the orbital plane (Z==0)
# observers_xyz = observers_xyz[:,mid]
x_heal, y_heal, z_heal = observers_xyz[0], observers_xyz[1], observers_xyz[2]
r_heal = np.sqrt(x_heal**2 + y_heal**2 + z_heal**2)   
observers_xyz = np.transpose(observers_xyz) #shape: Nx3
cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
cross_dot[cross_dot<0] = 0
cross_dot *= 4/len(observers_xyz)

#%% Tree ----------------------------------------------------------------------
xyz = np.array([X, Y, Z]).T
N_ray = 5_000

# Dynamic Box -----------------------------------------------------------------
x_ph = []
y_ph = []
z_ph = []
ph_idx = []
rph = []
r_initial = []
## just to check photosphere
for i in range(0, len(observers_xyz), 20):
    # Progress 
    print(f'Obs: {i}', flush=False)
    sys.stdout.flush()

    mu_x = observers_xyz[i][0] # mu_x = x_heal[i]
    mu_y = observers_xyz[i][1] # mu_y = y_heal[i]
    mu_z = observers_xyz[i][2] # mu_z = z_heal[i]

    # Box is for dynamic ray making
    # box gives -x, -y, -z, +x, +y, +z
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

    # we want rmax = rmax_mine*Robsmax_mine where Robs = sqrt(mu_ x_mine**2 + mu_ y_mine**2 + mu_ z_mine**2)
    # rmax_new = rmax_mine * np.sqrt(mu_x_mine**2 + mu_y_mine**2 + mu_z_mine**2)
    rs_max = [rmax, 0.001*rmax, 0.01*rmax, 0.1*rmax, 0.5*rmax, 2*rmax, 5*rmax]#, 10*rmax]
    label_rs = ['Healp', '0.001', '0.01', '0.1', '0.5', '2', '5']#, '10']
    marker_rs = ['*', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

    x_ph_r = np.zeros(len(rs_max))
    y_ph_r = np.zeros(len(rs_max))
    z_ph_r = np.zeros(len(rs_max))
    ph_idx_r = np.zeros(len(rs_max))
    r_initial_r = np.zeros(len(rs_max))
    plt.figure(figsize=(10, 10))
    for j, rmax_chosen in enumerate(rs_max):
        # print(f'Ray: {j}', flush=False)
        # sys.stdout.flush()
        r = np.logspace( -0.25, np.log10(rmax_chosen), N_ray)
        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        
        r_initial_r[j] = np.max(r)
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
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * prel.Rsol_cgs # this is the conversion for r

        # Red -----------------------------------------------------------------------
        # Get 20 unique, nearest neighbors
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew)
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
        ph_idx_r[j] = idx[b]
        rph_dim = dx[b]
        x_ph_r[j], y_ph_r[j], z_ph_r[j] = X[idx][b], Y[idx][b], Z[idx][b]

        plt.scatter(r_initial_r[j]/apo, r[b]/apo, c = 'dodgerblue', edgecolors = 'k', s = 50, marker=marker_rs[j], label = label_rs[j])
        # add error bars as rph_dim
        plt.errorbar(r_initial_r[j]/apo, r[b]/apo, yerr = rph_dim/apo, fmt = 'none', ecolor = 'k', elinewidth = 2)
        del smoothed_flux, R_lamda, fld_factor, EEr
        gc.collect()

    r_ph_r = np.sqrt(x_ph_r**2 + y_ph_r**2 + z_ph_r**2)
    # save data for photosphere
    r_initial.append(r_initial_r)
    x_ph.append(x_ph_r)
    y_ph.append(y_ph_r)
    z_ph.append(z_ph_r)
    ph_idx.append(ph_idx_r)
    rph.append(r_ph_r)
    plt.xlabel(r'R$_{\rm initial} [R_{\rm a}]$')
    plt.ylabel(r'R$_{\rm ph} [R_{\rm a}]$')
    plt.xlim(1e-2, 9e2)
    plt.ylim(1e-2, 9)
    plt.loglog()
    plt.legend()
    plt.title(f'Observer {i}. ' + r'All R$_{\rm initial}$ are a multiple of R$_{\rm initial Healp}$')
    plt.tight_layout()
    plt.savefig(f'{abspath}/Figs/Test/photosphere/{snap}/{snap}_RinRph_ray{i}.png', bbox_inches='tight')

#%% Save red of the single snap
eng.exit()

