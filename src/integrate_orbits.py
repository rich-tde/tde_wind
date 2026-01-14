abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
from scipy.integrate import cumulative_trapezoid
import Utilities.prelude as prel 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import src.orbits as orb
from Utilities.operators import from_cylindric

def solv_PW(x, theta, E, Rp, Mbh, c, G):
    Rs = 2 * G * Mbh / c**2
    # Rs, _, L = parameters_orbit(Rp, Ra, Mbh, c, G)
    L2 = 2 * Rp**2 * (E + G*Mbh/(Rp-Rs))

    u, y = x
    du_dtheta = y
    dy_dtheta = -u + G * Mbh / ((1 - Rs*u)**2 * L2)
    res =  np.array([du_dtheta, dy_dtheta])
    return res

def int_PW_orbit(theta_data, E, Rp, Mbh, c, G):
    # initial value has to be 0, for the initial condition, so you have [0,0]
    # initial conditions at pericenter
    u0 = 1/Rp
    y0 = 0  # dr/dtheta = 0 at pericenter

    u,y = odeint(solv_PW, [u0, y0], theta_data, args = (E, Rp, Mbh, c, G)).T 
    r = 1/u
    return r

def solv_kep(x, theta, G, Mbh, L2):
    u, y = x
    du_dtheta = y
    dy_dtheta = -u + G * Mbh / L2
    return np.array([du_dtheta, dy_dtheta])

def int_keplerian_orbit(theta_arr, Rp, E, Mbh, G):
    # theta_shift = -theta_arr 
    L2 = 2 * Rp**2 * ( E + G * Mbh/Rp)
    # initial conditions at pericenter
    u0 = 1/Rp
    y0 = 0  # dr/dtheta = 0 at pericenter

    u, y = odeint(solv_kep, [u0, y0], theta_arr, args=(G, Mbh, L2)).T
    r = 1 / u

    # time as a function of theta
    dt_dtheta = r**2 / np.sqrt(L2)
    t = cumulative_trapezoid(dt_dtheta, theta_arr, initial=0.0)

    return r, t

def keplerian_orbit(theta, a, Rp, ecc=1):
    # Don't care of the sign of theta, since you have the cos
    if ecc == 1:
        print('Parabolic')
        p = 2 * Rp
    else:
        p = a * (1 - ecc**2) 
    radius = p / (1 + ecc * np.cos(theta))
    return radius

def parameters_orbit_kep(Rp, a, Mbh, G, ecc):
    if ecc == 1:
        En = 0
    elif ecc < 0: # unbound
        En = G * Mbh / (2*a)
    elif ecc > 0: # bound
        En = - G * Mbh / (2*a)
    # L = np.sqrt(2 * G* Mbh * Ra *(1 + Ra/(Ra+Rp)))
    L = np.sqrt(2 * Rp**2 * ( En + G * Mbh/Rp))
    return En, L

# def solv_kep(x, theta, Rp, Ra, Mbh, G ):
#     _, L = parameters_orbit_kep(Rp, Ra, Mbh, G)
#     u,y = x
#     res =  np.array([y, (-u + G * Mbh / L**2)])
#     return res

# def int_keplerian_orbit(theta_data, Rp, Ra, Mbh, G):
#     # initial value has to be at theta=0 (i.e. pericenter), for the initial condition or you'd have infty, which is not accepted. 
#     # The initial condition are [r(0), v(0)]
#     theta_shift = theta_data + np.pi
#     u,y = odeint(solv_kep, [1/Rp, 0], theta_shift, args = (Rp, Ra, Mbh, G)).T 
#     r = 1/u
#     return r


if __name__ == "__main__":
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5 
    compton = 'Compton'
    check = 'HiResNewAMR' # '' or 'LowRes' or 'HiRes' 

    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    params = [Mbh, Rstar, mstar, beta]
    things = orb.get_things_about(params)
    Rs = things['Rs']
    Rt = things['Rt']
    Rp = things['Rp']
    R0 = things['R0']
    apo = things['apo']
    ecc_mb = things['ecc_mb']
    a_mb = things['a_mb']
    t_fb_days = things['t_fb_days']
    t_fb_cgs = t_fb_days * 24 * 3600

    # En = 0
    En, _ = parameters_orbit_kep(Rp, a_mb, Mbh, prel.G, ecc_mb)
    En = En
    a_kep = np.abs(prel.G * Mbh / (2 * En))
    ecc_kep = 1 - Rp / a_kep

    theta_step = 0.01
    theta_pre = np.arange(0, -np.pi-theta_step, -theta_step) # first element = 0, last = -pi --> you have to flip when concatenating
    theta_post = np.arange(0, np.pi+theta_step, theta_step) # it has to be from 0 because odeint requires initial conditions at theta_arr[0]
    theta_arr = np.concatenate((theta_pre[::-1], theta_post))
    # analytical keplerian orbit for mb
    r_kep_an_mb = keplerian_orbit(theta_arr, a_mb, Rp, ecc_mb)
    x_kep_an_mb, y_kep_an_mb = from_cylindric(theta_arr, r_kep_an_mb)
    # analytical keplerian orbit 
    r_kep_an = keplerian_orbit(theta_arr, a_kep, Rp, ecc_kep)
    x_kep_an, y_kep_an = from_cylindric(theta_arr, r_kep_an)
    # numerical keplerian orbit
    r_kep_pre, t_pre = int_keplerian_orbit(theta_pre, Rp, En, Mbh, prel.G) # first element at 0, last at -pi --> you have to flip when concatenating
    r_kep_post, t_post = int_keplerian_orbit(theta_post, Rp, En, Mbh, prel.G)
    r_kep = np.concatenate((r_kep_pre[::-1], r_kep_post))
    x_kep, y_kep = from_cylindric(theta_arr, r_kep)
    t_kep = np.concatenate((t_pre[::-1], t_post))
    fig, (ax1, ax2) =plt.subplots(2,1, figsize=(10,7))
    ax1.plot(t_kep*prel.tsol_cgs/t_fb_cgs, r_kep/Rt) 
    ax1.set_ylabel(r'r [$r_{\rm t}$]')
    ax2.plot(t_kep*prel.tsol_cgs/t_fb_cgs, theta_arr) 
    ax2.set_xlabel(r'Time [$t_{\rm fb}$]')
    ax2.set_ylabel(r'$\theta$ [rad]')

    r_witt_pre = int_PW_orbit(theta_pre, En, Rp, Mbh, prel.csol_cgs, prel.G)
    r_witt_post = int_PW_orbit(theta_post, En, Rp, Mbh, prel.csol_cgs, prel.G)
    r_witta = np.concatenate((r_witt_pre[::-1], r_witt_post))   
    x_witta, y_witta = from_cylindric(theta_arr, r_witta)

    plt.figure(figsize=(10,5))
    plt.plot(x_kep_an_mb/apo, y_kep_an_mb/apo, label = 'Analytical Kep Orbit mb', c = 'grey', alpha = 0.5)
    plt.plot(x_kep_an/apo, y_kep_an/apo, label = 'Analytical Kep Orbit')
    plt.plot(x_kep/apo, y_kep/apo, label = 'Numerical Kep Orbit', ls ='--')
    plt.plot(x_witta/apo, y_witta/apo, label = 'Numerical PW Orbit', ls =':')
    plt.xlabel(r'x [$r_{\rm a}$]')
    plt.ylabel(r'y [$r_{\rm a}$]')
    # plt.title(f'Orbit Comparison, ecc = {ecc_kep:.2f}, a = {a_kep:.0f} $R_\odot$')
    plt.xlim([-1.1, 0.2])
    plt.ylim([-0.4, 0.4])
    # plt.grid(True)
    plt.legend(fontsize = 18)
    plt.show()