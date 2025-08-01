import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
from Utilities.operators import to_cylindric

def compute_semi_minor_axis(a, e):
    """Compute semi-minor axis from semi-major axis and eccentricity."""
    if not (0 <= e < 1):
        raise ValueError("Eccentricity must be in [0, 1).")
    return a * np.sqrt(1 - e**2)


def ds_dt(t, a, b):
    """Differential arclength function for an ellipse parameterized by t."""
    return np.sqrt((a * np.sin(t))**2 + (b * np.cos(t))**2)


def total_arclength(a, b):
    """Compute the total arclength of the ellipse."""
    result, _ = scipy.integrate.quad(lambda t: ds_dt(t, a, b), 0, 2 * np.pi)
    return result


def arclength_to_t(S_target, a, b, s_max):
    """Find parameter t such that arclength from 0 to t equals S_target."""
    def f(t):
        result, _ = scipy.integrate.quad(lambda t_: ds_dt(t_, a, b), 0, t)
        return result - S_target

    return scipy.optimize.root_scalar(f, bracket=[0, 2 * np.pi], method='brentq').root


def generate_uniform_ellipse_points(a, e, N=100, cart = False):
    """
    Generate N equally spaced points (in arclength) along an ellipse
    defined by semi-major axis a and eccentricity e.

    Returns:
        x_values, y_values: Numpy arrays of x and y coordinates
    """
    b = compute_semi_minor_axis(a, e)
    s_max = total_arclength(a, b)
    S_values = np.linspace(0, s_max, N)

    t_values = np.array([arclength_to_t(S, a, b, s_max) for S in S_values])
    x_values = a * np.cos(t_values)
    y_values = b * np.sin(t_values)

    if cart:
        return x_values, y_values
    else:
        theta_values, _ = to_cylindric(x_values, y_values)
        theta_values = np.sort(theta_values)
        return theta_values


def plot_ellipse(x_values, y_values, a=None, e=None, title=None):
    """
    Plot ellipse points with aspect ratio 1.
    """
    x_ryan, y_ryan = from_cylindric(theta_ryan, r_ryan)

    plt.figure(figsize=(10, 10))
    plt.plot(x_values, y_values, '.', markersize = 1, label='Ellipse Points')
    plt.plot(x_ryan, y_ryan, 'r.', markersize = 1, label='Ryan Points')

    plt.gca().set_aspect('equal', adjustable='box')
    if title:
        plt.title(title)
    elif a is not None and e is not None:
        plt.title(f"Ellipse with a={a}, e={e}")
    plt.grid(True)
    plt.show()


def verify_spacing(x_values, y_values):
    """
    Print min, max, mean spacing between consecutive points.
    """
    distances = np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2)
    print(f"Min distance: {distances.min():.5f}")
    print(f"Max distance: {distances.max():.5f}")
    print(f"Mean distance: {distances.mean():.5f}")
    return distances

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
    Rt = Rstar * (Mbh/mstar)**(1/3)
    Rp =  Rt / beta

    a = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
    e = orb.e_mb(Rstar, mstar, Mbh, beta) 
    N = 200    # number of points
    x, y = generate_uniform_ellipse_points(a, e, N)
    plot_ellipse(x, y, a, e)
    verify_spacing(x, y)

    # my sample
    theta_lim =  np.pi
    step = 0.03
    theta_init = np.arange(-theta_lim, theta_lim, step)
    theta_ryan = Ryan_sampler(theta_init)
    r_ryan = orb.keplerian_orbit(theta_ryan, a, Rp, e)

    # look at the angles
    theta_ellip, _ = to_cylindric(x,y)
    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(theta_ellip), '.', markersize=1, label='Ellipse Points')
    plt.plot(np.sort(theta_ryan), 'r.', markersize=1, label='Ryan Points')
    plt.legend()
    plt.show()
