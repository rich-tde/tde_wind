""" 
Find different kind of orbits (and their derivatives) for TDEs. 
Identify the stream using the centre of mass.
Measure the width and the height of the stream.
"""
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import Utilities.operators as op
from scipy.interpolate import griddata
import Utilities.prelude as prel
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import trapezoid

def make_cfr(R, x0=0, y0=0):
    x = np.linspace(-R, R, 100)
    y = np.linspace(-R, R, 100)
    xcfr, ycfr = np.meshgrid(x,y)
    cfr = (xcfr-x0)**2 + (ycfr-y0)**2 - R**2
    return xcfr, ycfr, cfr

def keplerian_energy(Mbh, G, t):
    """ specific orbital energy of a Keplerian orbit
    Parameters
    ----------
    Mbh : float
        Black hole mass in solar masses
    G : float
        Gravitational constant
    t : float
        Time in code units, NOT in fallback time
    Returns
    -------
    energy : float
        Specific orbital energy of a Keplerian orbit in code units
    """
    energy = (np.pi * G * Mbh / (np.sqrt(2) * t))**(2/3)
    return energy

def Mdot_fb(Mbh, G, t, dmdE):
    """ Mass fallback rate according to Keplers law (Eq.26 in Rossi+20)
    Parameters
    ----------
    Mbh : float
        Black hole mass in solar masses
    G : float
        Gravitational constant
    t : float
        Time in code units, NOT in fallback time
    dmdE : float
        dM/dE from numerical simulation data
    Returns
    -------
    Mdot : float
        Mass fallback rate in code units
    """
    Mdot = -1/3 * dmdE * (2 * np.pi * G * Mbh)**(2/3) * t**(-5/3) # -2/3 * dmdE * (np.pi * G * Mbh / np.sqrt(2))**(2/3) * t**(-5/3)
    return Mdot

def keplerian_orbit(theta, a, Rp, ecc=1):
    # Don't care of the sign of theta, since you have the cos
    if ecc == 1:
        p = 2 * Rp
    else:
        p = a * (1 - ecc**2) 
    radius = p / (1 + ecc * np.cos(theta))
    return radius

def bern_coeff(Rsph, vel, den, mass, Press, IE_den, Rad_den, params, G = prel.G):
    orb_en = orbital_energy(Rsph, vel, mass, params, G) 
    orb_en_spec = orb_en/mass
    IE_spec = IE_den / den
    Press_spec = (Rad_den/3 + Press) / den
    B = orb_en_spec + IE_spec + Press_spec
    return B

def pick_wind(X, Y, Z, VX, VY, VZ, Den, Mass, Press, IE_den, Rad_den, params):
    """ select the points that are in the wind, according to the Bernoulli criterion.
    Parameters
    ----------
    X, Y, Z : array
        Coordinates of the points in code units
    VX, VY, VZ : array
        Velocities of the points in code units
    Den, Mass, Press, IE_den, Rad_den : array
        Density, mass, pressure, internal energy density and radiation energy density of the points in code units
    params : array
        Parameters of the simulation [Mbh, Rstar, mstar, beta]
    Returns
    -------
    cond_wind : array
        Boolean array that is True for the points that are in the wind and False for the points that are not in the wind
    bern : array
        Bernoulli coefficient of (ALL) the points in code units
    V_r : array
        Radial velocity of (ALL) the points in code units
    """
    Rsph = np.sqrt(X**2 + Y**2 + Z**2)
    vel = np.sqrt(VX**2 + VY**2 + VZ**2)
    V_r, _, _ = op.to_spherical_components(VX, VY, VZ, X, Y, Z)
    bern = bern_coeff(Rsph, vel, Den, Mass, Press, IE_den, Rad_den, params)
    cond_wind = np.logical_and(V_r >= 0, bern > 0)

    return cond_wind, bern, V_r

def streamlines(x, y, vx, vy, params_x, params_y, color_plot = None):
    xmin, xmax, nx = params_x
    ymin, ymax, ny = params_y
    x_uniform = np.linspace(xmin, xmax, nx)
    y_uniform = np.linspace(ymin, ymax, ny)
    X_grid, Y_grid = np.meshgrid(x_uniform, y_uniform)

    Vx_grid = griddata((x, y), vx,
                    (X_grid, Y_grid), method = 'nearest')

    Vy_grid = griddata((x, y), vy,
                    (X_grid, Y_grid), method = 'nearest')
    
    if color_plot is not None:
        color_grid = griddata((x, y), color_plot,
                        (X_grid, Y_grid), method = 'nearest')
        
        return X_grid, Y_grid, Vx_grid, Vy_grid, color_grid
    
    else:
        return X_grid, Y_grid, Vx_grid, Vy_grid


def R_grav(Mbh, c, G):
    """ Gravitational radius of the black hole."""
    Rg = G * Mbh /c**2
    return Rg

def tidal_radius(Rstar, mstar, Mbh):
    Rt = Rstar * (Mbh/mstar)**(1/3)
    return Rt

def semimajor_axis(Rstar, mstar, Mbh, G):
    """ Semimajor axis of the most bound debris """
    E = energy_mb(Rstar, mstar, Mbh, G)
    a = G * Mbh / (2*E)
    return a

def pericentre(Rstar, mstar, Mbh, beta):
    Rt = tidal_radius(Rstar, mstar, Mbh)
    Rp = Rt/beta
    return Rp

def apocentre(Rstar, mstar, Mbh, beta):
    # comes from Ra=a(1+e), a=Rt^2/2Rstar, e=1-2*Rstar/(beta*Rt)
    Rt = tidal_radius(Rstar, mstar, Mbh)
    apo = Rt**2/Rstar - Rt/beta 
    return apo

def specific_j(r, vel):
    """ (Magnitude of) specific angular momentum """
    j = np.cross(r, vel)
    magnitude_j = np.linalg.norm(j, axis = 1)
    return magnitude_j

def eccentricity_squared(r, vel, specOE, Mbh, G):
    j = specific_j(r, vel)
    ecc2 = 1 + 2 * specOE * j**2 / (G * Mbh)**2
    return ecc2

def e_mb(Rstar, mstar, Mbh, beta):
    # eccentricity of most bound derbis. It comes from Rp = a(1-e), a = Rt^2/2Rstar
    Rt = tidal_radius(Rstar, mstar, Mbh)
    ecc = 1-2*Rstar/(beta*Rt)
    return ecc

def energy_mb(Rstar, mstar, Mbh, G):
    """ Specific orbital energy of most bound debris """
    Rt = tidal_radius(Rstar, mstar, Mbh)
    En = G * Mbh * Rstar / Rt**2
    return En

def Edd(Mbh, kappa, eta, c, G):
    Rg = R_grav(Mbh, c, G)
    Ledd = 4 * np.pi * Rg * c**3 / kappa #1.26e38 * Mbh [erg/s], Mbh is in solar masses
    Medd = Ledd / (eta * c**2)
    return Ledd, Medd

def get_things_about(params, c = prel.csol_cgs, G = prel.G):
    Mbh, Rstar, mstar, beta = params
    Rg = R_grav(Mbh, c, G)
    Rs = 2 * Rg
    Rt = tidal_radius(Rstar, mstar, Mbh)
    R0 = 0.6 * Rt
    Rp = pericentre(Rstar, mstar, Mbh, beta)
    a_mb = semimajor_axis(Rstar, mstar, Mbh, G)
    apo = apocentre(Rstar, mstar, Mbh, beta)
    ecc_mb = e_mb(Rstar, mstar, Mbh, beta)
    E_mb = energy_mb(Rstar, mstar, Mbh, G)
    t_fb_days = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2) #[days]
    # make a dictionary
    things = {
        'Rg': Rg,
        'Rs': Rs,
        'Rp': Rp,
        'Rt': Rt,
        'R0': R0,
        'a_mb': a_mb,
        'apo': apo, 
        'ecc_mb': ecc_mb,
        'E_mb': E_mb,
        't_fb_days': t_fb_days
    }
    return things

def parameters_orbit(Rp, Ra, Mbh, c, G ):
    Rs = 2 * G * Mbh / c**2
    En = G * Mbh * (Rp**2 * (Ra-Rs) - Ra**2 * (Rp-Rs)) / ((Ra**2-Rp**2) * (Rp-Rs) * (Ra-Rs))
    L = np.sqrt(2 * Ra**2 * (En + G*Mbh/(Ra-Rs)))
    return Rs, En, L

def solvr(x, theta, Rp, Ra, Mbh, c, G ):
    Rs, _, L = parameters_orbit(Rp, Ra, Mbh, c, G)
    u,y = x
    res =  np.array([y, (-u + G * Mbh / ((1 - Rs*u) * L)**2)])
    return res

def Witta_orbit(theta_data, Rp, Ra, Mbh, c, G):
    theta_data += np.pi
    u,y = odeint(solvr, [0, 0], theta_data, args = (Rp, Ra, Mbh, c, G)).T 
    r = 1/u
    return r

def precession_angle(Rstar, mstar, Mbh, beta, c, G):
    a = semimajor_axis(Rstar, mstar, Mbh, beta)
    e = e_mb(Rstar, mstar, Mbh, beta)
    theta = 6 * np.pi * G * Mbh / (c**2 * a * (1-e**2))
    return theta

def R_selfinter(Rstar, mstar, Mbh, beta, c, G):
    theta = precession_angle(Rstar, mstar, Mbh, beta, c, G)
    e_mb = e_mb(Rstar, mstar, Mbh, beta)
    Rt = tidal_radius(Rstar, mstar, Mbh)
    R = (1+e_mb) * Rt / (beta * (1-e_mb * np.cos(theta/2)))
    return R

def smoothed_pw_potential(r, params, G):
    """
    Compute the smoothed Paczyński-Wiita potential at a single radius r.

    Parameters:
        r : float
            Radius at which to compute the potential.
        params : list
            List of parameters [Mbh, Rstar, mstar, beta] 
        G : float
            Gravitational constant.
    Returns: 
        Phi : float
            Gravitational potential at radius r.
    """
    things = get_things_about(params)
    Mbh = params[0]
    Rs = things['Rs']   
    R0 = things['R0']
    pot = np.empty_like(r, dtype=float)

    mask = r >= R0
    pot[mask] = -G * Mbh / (r[mask] - Rs)
    pot[~mask] = -G * Mbh / (R0 - Rs) * (1 + 0.5 * R0 / (R0 - Rs) - 0.5 * r[~mask]**2 / (R0 * (R0 - Rs)))

    return pot

def orbital_energy(r, vel, mass, params, G):
    # no angular momentum??
    potential = smoothed_pw_potential(r, params, G)
    energy = mass * (0.5 * vel**2 + potential)
    return energy

def deriv_an_orbit(theta, a, Rp, ecc, choose):
    # we need the - in front of theta to be consistent with the function to_cylindric, where we change the orientation of the angle
    theta = -theta
    if choose == 'Keplerian':
        if ecc == 1:
            p = 2 * Rp
        else:
            p = a * (1 - ecc**2)
        dr_dtheta = p * ecc * np.sin(theta)/ (1 + ecc * np.cos(theta))**2
    # elif choose == 'Witta':
    return dr_dtheta

def deriv_maxima(theta_arr, x_orbit, y_orbit):
    # Find the idx where the orbit starts to decrease too much (i.e. the stream is not there anymore)
    idx = len(y_orbit)
    for i in range(len(y_orbit)):
        if np.abs(y_orbit[i]-y_orbit[i-1]) >= 50:
            idx = i
            break
    # Find the numerical derivative of r with respect to theta. 
    # Shift theta to start from 0 and compute the arclenght
    theta_shift = theta_arr-theta_arr[0] 
    r_orbit = np.sqrt(x_orbit**2 + y_orbit**2)
    dr_dtheta = np.gradient(r_orbit, theta_shift)

    return dr_dtheta, idx

def find_arclenght(theta_arr, orbit, choose, params, c = prel.csol_cgs, G = prel.G):
    """ 
    Compute the arclenght of the orbit
    Parameters
    ----------
    theta_arr : array
        The angles of the orbit
    orbit : 1D or 2D array
        The orbit. 
        If choose is Keplerian/Witta: radius(thetas). 
        If choose is stream: x and y of the orbit.
    choose : str
        The kind of orbit. It can be 'Keplerian', 'Witta', 'stream'
    params : array
        The parameters of the orbit
    Returns
    -------
    s : array
        The arclenght of the orbit
    """
    if choose == 'stream':
        drdtheta, idx = deriv_maxima(theta_arr, orbit[0], orbit[1])
        r_orbit = np.sqrt(orbit[0]**2 + orbit[1]**2)
    elif choose == 'Keplerian' or choose == 'Witta':
        things = get_things_about(params, c, G)
        a, Rp, ecc = things['a_mb'], things['Rp'], things['ecc_mb'] # params[0], params[1], params[2] #
        drdtheta = deriv_an_orbit(theta_arr, a, Rp, ecc, choose)
        r_orbit = orbit
        idx = len(r_orbit) # to keep the whole orbit

    ds = np.sqrt(r_orbit**2 + drdtheta**2)
    theta_arr = theta_arr-theta_arr[0] # to start from 0
    s = np.zeros(len(theta_arr))
    for i in range(len(theta_arr)):
        s[i] = trapezoid(ds[:i+1], theta_arr[:i+1])

    return s, idx


if __name__ == '__main__':
    from Utilities.operators import make_tree, Ryan_sampler, from_cylindric
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from Utilities.sections import make_slices
    
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    check = 'NewAMR'
    compton = 'Compton'

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    snap = '238'
    path = f'{abspath}/TDE/{folder}{check}/{snap}'

    params = [Mbh, Rstar, mstar, beta]
    things = get_things_about(params)
    Rt = things['Rt']
    Rp = things['Rp']
    apo = things['apo']
    a_mb = things['a_mb']
    ecc_mb = things['ecc_mb']
    t_fb_days = things['t_fb_days']
    t_fb_days_cgs = t_fb_days * 24 * 3600 

    theta_lim =  np.pi
    step = np.round((2*theta_lim)/200, 3)
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_arr = Ryan_sampler(theta_init)

    # data = make_tree(path, snap, energy = False)
    # dim_cell = data.Vol**(1/3)
    # cut = data.Den > 1e-19
    # X, Y, Z, dim_cell, Den, Mass = \
    #     make_slices([data.X, data.Y, data.Z, dim_cell, data.Den, data.Mass], cut)

    make_stream = False
    make_width = False
    test_s = True
    test_orbit = False

    if test_s:
        # arclenght of a circle with keplerian
        r_kep = keplerian_orbit(theta_arr, a_mb, Rp, ecc_mb)
        x_kep, y_kep = from_cylindric(theta_arr, r_kep)
        s_kep, _ = find_arclenght(theta_arr, r_kep, choose = 'Keplerian', params = params)
        s, _ = find_arclenght(theta_arr, [x_kep, y_kep], choose = 'stream', params = params)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
        plt.suptitle('Check with Kepler', fontsize = 18)
        img = ax1.scatter(x_kep, y_kep, c = theta_arr, s = 5, cmap = 'rainbow', vmin = theta_arr[0], vmax = theta_arr[-1])
        plt.colorbar(img, label = r'$\theta$') 
        ax1.set_xlim(-300, 50)
        ax1.set_ylim(-100, 100)
        ax1.set_xlabel(r'X [$R_\odot$]')
        ax1.set_ylabel(r'Y [$R_\odot$]')
        # ax2.plot(theta_arr[:idx], s, c = 'r', label = 'maxima')
        ax2.plot(theta_arr, s_kep, c = 'k')
        ax2.plot(theta_arr, s, c = 'r', ls = '--')
        ax2.set_xlabel(r'$\theta$')
        ax2.set_ylabel(r'$s [R_\odot]$')
        ax2.grid()
        plt.tight_layout()      

        idx_chosen = 70
        theta_arr, x_stream, y_stream, z_stream, thresh_cm = \
            np.load(f'{abspath}/data/{folder}/WH/stream/stream_{check}{snap}.npy', allow_pickle=True)
        s, idx = find_arclenght(theta_arr, [x_stream, y_stream], choose = 'stream', params = params)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
        img = ax1.scatter(x_stream[:idx], y_stream[:idx], c = theta_arr[:idx], s = 5, cmap = 'rainbow', vmin = theta_arr[0], vmax = theta_arr[-1])
        plt.colorbar(img, label = r'$\theta$') 
        ax1.scatter(x_stream[idx_chosen], y_stream[idx_chosen], c = 'k', s = 50, marker = 'x')
        ax1.set_xlim(-300, 50)
        ax1.set_ylim(-100, 100)
        ax1.set_xlabel(r'X [$R_\odot$]')
        ax1.set_ylabel(r'Y [$R_\odot$]')
        ax2.plot(theta_arr[:idx], s[:idx], c = 'r')
        ax2.scatter(theta_arr[idx_chosen], s[idx_chosen], c = 'k', s = 50, marker = 'x')
        ax2.set_xlabel(r'$\theta$')
        ax2.set_ylabel(r'$s [R_\odot]$')
        ax2.grid()
        plt.tight_layout()      
    
    if test_orbit:
        r_Witta = Witta_orbit(theta_arr, Rp, Ra, Mbh, c, G)
        x_Witta, y_Witta = from_cylindric(theta_arr, r_Witta)
        r_kepler = keplerian_orbit(theta_arr, a, Rp, ecc = 1)
        x_kepler, y_kepler = from_cylindric(theta_arr, r_kepler)
        plt.plot(x_Witta, y_Witta, c = 'r', label = 'Witta')
        plt.plot(x_kepler, y_kepler, '--', c = 'b', label = 'Keplerian')
        plt.xlim(-300,20)
        plt.ylim(-100, 100)
        plt.legend()
        plt.grid()
        plt.show()

    if make_stream:
        x_stream, y_stream, z_stream, thresh_cm, x_cmTR, y_cmTR, z_cmTR, x_stream_rad, y_stream_rad, z_stream_rad = find_transverse_com(X, Y, Z, dim_cell, Den, Mass, theta_arr, params, test = True)
        np.save(f'/Users/paolamartire/shocks/data/{folder}/streamRad_{check}{snap}.npy', [theta_arr, x_stream_rad, y_stream_rad, z_stream_rad])
        np.save(f'/Users/paolamartire/shocks/data/{folder}/streamcmTR_{check}{snap}.npy', [theta_arr, x_cmTR, y_cmTR, z_cmTR])
        np.save(f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy', [theta_arr, x_stream, y_stream, z_stream, thresh_cm])

        #%%
        plt.figure(figsize = (10,6))
        plt.plot(x_stream_rad, y_stream_rad, c = 'b', label = 'Max density stream')
        plt.plot(x_cmTR, y_cmTR, c = 'r', label = 'COM TZ plane of max density points')
        plt.plot(x_stream, y_stream, c = 'k', label = 'COM TZ plane of previous COM points')
        plt.xlim(-300,20)
        plt.ylim(-60,60)
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.grid()
        plt.legend()
        plt.savefig(f'/Users/paolamartire/shocks/Figs/WH/TZStream{snap}.png')
        plt.show() 

        plt.figure(figsize = (6,6))
        plt.plot(x_stream_rad, y_stream_rad, c = 'b', label = 'Max density stream')
        plt.plot(x_cmTR, y_cmTR, c = 'r', label = 'COM TZ plane of max density points')
        plt.plot(x_stream, y_stream, c = 'k', label = 'COM TZ plane of previous COM points')
        plt.xlim(-60,20)
        plt.ylim(-50,50)
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.grid()
        plt.legend()
        plt.show() 
    
    if make_width:
        file = f'/Users/paolamartire/shocks/data/{folder}/stream_{check}{snap}.npy' 
        stream, indeces_boundary, x_T_width, w_params, h_params, theta_arr  = follow_the_stream(X, Y, Z, dim_cell, Mass, path = file, params = params)
        cm_x, cm_y, cm_z = stream[0], stream[1], stream[2]
        low_x, low_y = X[indeces_boundary[:,0]] , Y[indeces_boundary[:,0]]
        up_x, up_y = X[indeces_boundary[:,1]] , Y[indeces_boundary[:,1]]
        # midplane = Z < dim_cell
        # X_midplane, Y_midplane, Den_midplane, Mass_midplane = make_slices([X, Y, Den, Mass], midplane)
        plt.figure(figsize = (12,8))
        # img = plt.scatter(X_midplane, Y_midplane, c = Den_midplane, s = 1, cmap = 'viridis', alpha = 0.2, vmin = 2e-8, vmax = 6e-8)
        # cbar = plt.colorbar(img)
        # cbar.set_label(r'Density', fontsize = 16)
        # plt.plot(cm_x, cm_y,  c = 'k')
        plt.plot(low_x[:-10], low_y[:-10], '--', c = 'k')
        plt.plot(up_x[:-10], up_y[:-10], '-.', c = 'k', label = 'Upper tube')
        plt.xlabel(r'X [$R_\odot$]', fontsize = 18)
        plt.ylabel(r'Y [$R_\odot$]', fontsize = 18)
        plt.xlim(-300,30)
        plt.ylim(-75,75)
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{check}/stream{snap}.png', dpi=300)
        plt.show()
# %%
