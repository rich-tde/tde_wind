import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import Utilities.prelude as prel


def system(y, t, m, G = prel.G):
    # dH is dH/dt, so you'll have both the variables H and dH since you have H''
    bigDelta, dDelta, bigH, dH, smallLambda, omega, v, bigLambda = y
    r = 1
    rho = bigLambda / (np.pi * bigDelta * bigH)
    delta = 1
    K = 1
    ddDelta_dt = -G * m / r**3 * bigDelta * (1-3*np.sin(delta)**2)+ bigDelta * omega**2 - 2 * omega * (bigDelta * omega + v) \
                + K * rho**(2/3) / bigDelta \
                - 4*np.pi**2 * G * rho * bigDelta * bigH / (bigDelta + bigH)
    ddH_dt = -G * m / r**3 * bigH \
            + K * rho**(2/3) / bigH \
            - 4*np.pi**2 * G * rho * bigDelta * bigH / (bigDelta + bigH)
    dsmallLambda_dt = omega**2 - smallLambda**2 - G*m/r**3 * (1-3*np.cos(delta)**2)
    domega_dt = -2*omega*smallLambda + 3*G*m/r**3 * np.cos(delta) * np.sin(delta)
    dv_dt = 3 * G * m/r**3 * bigDelta * np.cos(delta) * np.sin(delta) + dDelta * omega - smallLambda * (bigDelta * omega + v)
    dbigLambda_dt = -smallLambda * bigLambda
    return [ddDelta_dt, ddH_dt, dsmallLambda_dt, domega_dt, dv_dt, dbigLambda_dt]

Rstar = 0.47
mu = 1
width_i = Rstar * np.sqrt(1-mu**2)
bigLambda_i = 1
y0 = [width_i, 0, width_i, 0, 0, 0, 0, bigLambda_i]
t = np.linspace(0, 10, 100)  # time points from t=0 to t=10

solution = odeint(system, y0, t)
# Extract the results
bigDelta, dDelta, bigH, dH, smallLambda, omega, v, bigLambda = solution.T