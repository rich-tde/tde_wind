import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from Utilities.operators import from_cylindric, sort_list

# def compute_semi_major_axis(Rp, e):
#     """Compute semi-major axis from pericenter Rp and eccentricity e."""
#     if not (0 <= e < 1):
#         raise ValueError("Eccentricity must be in [0, 1).")
#     return Rp / (1 - e)

def r_of_f(f, a, e):
    """Radius at true anomaly f (focus at origin)."""
    return a * (1 - e**2) / (1 + e * np.cos(f))

def dr_df(f, a, e):
    """Derivative dr/df."""
    return a * (1 - e**2) * (e * np.sin(f)) / (1 + e * np.cos(f))**2

def ds_df(f, a, e):
    """Differential arclength ds/df for orbit parameterized by true anomaly f."""
    r = r_of_f(f, a, e)
    dr = dr_df(f, a, e)
    return np.sqrt(dr**2 + r**2)

def total_arclength(a, e):
    """Total orbit arclength for f in [0, 2π]."""
    val, _ = integrate.quad(lambda ff: ds_df(ff, a, e), 0, 2*np.pi, limit=400)
    return val

def arclength_to_f(S_target, a, e, S_total):
    """Find true anomaly f such that arc length from 0 to f equals S_target."""
    def residual(f):
        val, _ = integrate.quad(lambda ff: ds_df(ff, a, e), 0, f, limit=200)
        return val - S_target
    return optimize.brentq(residual, 0, 2*np.pi)

def generate_uniform_keplerian_points(e, G, N=200, cart=False, method="interp"):
    """
    Generate N equally spaced points (in arclength) along a Keplerian ellipse.

    Parameters
    ----------
    Rp : float
        Pericenter distance.
    e : float
        Eccentricity (0 <= e < 1).
    N : int
        Number of points to generate.
    cart : bool
        If True, return x and y coordinates. If False, return f and r arrays.
    method : str
        'interp' (fast) or 'root' (slow but accurate).

    Returns
    -------
    If cart=True:
        x_values, y_values : arrays
    If cart=False:
        f_values, r_values : arrays
    """
    a = orb.semimajor_axis(Rstar, mstar, Mbh, G) #compute_semi_major_axis(Rp, e)
    S_total = total_arclength(a, e)

    S_values = np.linspace(0, S_total, N, endpoint=False)

    if method == "root":
        f_values = np.array([arclength_to_f(S, a, e, S_total) for S in S_values])
    else:
        # fast interpolation method
        M = 20000
        f_grid = np.linspace(0, 2*np.pi, M)
        ds_vals = ds_df(f_grid, a, e)
        dS = np.cumsum(0.5 * (ds_vals[:-1] + ds_vals[1:]) * (f_grid[1]-f_grid[0]))
        S_grid = np.concatenate(([0.0], dS))
        f_values = np.interp(S_values, S_grid, f_grid)

    # convert to [-π, π)
    f_values = (f_values + np.pi) % (2*np.pi) - np.pi

    r_values = r_of_f(f_values, a, e)
    r_values, f_values = sort_list([r_values, f_values], f_values)
    x_values, y_values = from_cylindric(f_values, r_values)

    if cart:
        return x_values, y_values
    else:
        return f_values, r_values

if __name__ == "__main__":
    import prelude as prel
    import src.orbits as orb
    from operators import from_cylindric, to_cylindric, Ryan_sampler
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    params = [Mbh, Rstar, mstar, beta]
    things = orb.get_things_about(params)
    e = things['ecc_mb']
    a = things['a_mb']
    Rp = things['Rp']
    Ra = things['apo']

    # my sample
    theta_lim =  np.pi
    step = np.round((2*np.pi)/200,3)
    print(step)
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_ryan = Ryan_sampler(theta_init)
    r_ryan = orb.keplerian_orbit(theta_ryan, a, Rp, e)
    x_ryan, y_ryan = from_cylindric(theta_ryan, r_ryan)

    N = 200    # number of points
    # Cartesian coordinates
    x, y = generate_uniform_keplerian_points(e, prel.G, N, cart=True)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_ryan, y_ryan, s = 1, label='Ryan Points', c = 'b')
    # plt.scatter(x, y, s = 2, label='Ellipse Points', c = 'r')
    # plt.xlim(-20,20)
    # plt.ylim(-20,20)
    plt.legend()
    plt.show()
    
    # Polar (true anomaly, radius)
    theta_ellip, r = generate_uniform_keplerian_points(e, prel.G, N, cart=False)

    # look at the angles
    plt.figure(figsize=(10, 5))
    img = plt.scatter(np.arange(N), theta_ellip, s = 2, label='Ellipse Points', c = x, cmap = 'rainbow', vmin = -330, vmax = 20)
    plt.colorbar(img, label = 'X') 
    plt.scatter(np.arange(len(theta_ryan)), theta_ryan, s = 2, c = 'k', label='Ryan Points')
    plt.ylabel(r'$\theta$')
    plt.xlabel('Index')
    plt.axhline(np.pi/2, c = 'k', ls = '--')
    plt.legend()
    plt.show()
