abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)
import numpy as np
import Utilities.prelude as prel
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import src.orbits as orb
from Utilities.operators import from_cylindric

def parameters_orbit(Rp, Ra_pos, Mbh, c, G ):
    Ra = -Ra_pos  # Because we define apo as negative
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
    # initial value has to be 0, for the initial condition, so you have [0,0]
    # theta_data += np.pi
    u,y = odeint(solvr, [0, 0], theta_data, args = (Rp, Ra, Mbh, c, G)).T 
    r = 1/u
    return r

# def parameters_orbit_kep(Rp, Ra_pos, Mbh, G ):
#     Ra = -Ra_pos  # Because we define apo as negative
#     En = G * Mbh / (Rp + Ra)
#     L = np.sqrt(2 * G* Mbh * Ra *(1 + Ra/(Ra+Rp)))
#     return En, L

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

def solv_kep(x, theta, G, Mbh, L2):
    u, y = x
    du_dtheta = y
    dy_dtheta = -u + G * Mbh / L2
    return np.array([du_dtheta, dy_dtheta])

def int_keplerian_orbit(theta_arr, Rp, E, Mbh, G):
    theta_shift = theta_arr - theta_arr[0] # Shift to have pericenter at theta=0

    L2 = 2 * Rp**2 * ( E + G * Mbh/Rp)

    # initial conditions at pericenter
    u0 = 1/Rp
    y0 = 0  # dr/dtheta = 0 at pericenter

    u, y = odeint(solv_kep, [u0, y0], theta_arr, args=(G, Mbh, L2)).T
    return 1/u

def keplerian_orbit(theta, a, Rp, ecc=1):
    # Don't care of the sign of theta, since you have the cos
    if ecc == 1:
        p = 2 * Rp
    else:
        p = a * (1 - ecc**2) 
    radius = p / (1 + ecc * np.cos(theta))
    return radius

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

    data_arr = np.arange(-np.pi, np.pi, 0.1)
    # r_kep = int_keplerian_orbit(data_arr, Rp, apo, Mbh, prel.G)
    r_kep = int_keplerian_orbit(data_arr, Rp, 0, Mbh, prel.G)
    x_kep, y_kep = from_cylindric(data_arr, r_kep)
    # r_witta = Witta_orbit(data_arr, Rp, apo, Mbh, prel.csol_cgs, prel.G)
    # x_witta, y_witta = from_cylindric(data_arr, r_witta)
    r_kep_an = keplerian_orbit(data_arr, a_mb, Rp, ecc = 1)
    x_kep_an, y_kep_an = from_cylindric(data_arr, r_kep_an)

    plt.figure(figsize=(10,8))
    plt.plot(x_kep_an/Rt, y_kep_an/Rt, label = 'Analytical Keplerian Orbit', ls=':')
    plt.plot(x_kep/Rt, y_kep/Rt, label = 'Keplerian Orbit')
    # plt.plot(x_witta/Rt, y_witta/Rt, label = 'Witta Orbit', ls ='--')
    plt.xlabel(r'x [$r_{\rm t}$]')
    plt.ylabel(r'y [$r_{\rm t}$]')
    # plt.title('Orbit Comparison')
    plt.xlim([-apo/Rt, 1.5])
    plt.ylim([-20, 20])
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.show()