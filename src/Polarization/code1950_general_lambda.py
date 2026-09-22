'''Numerical implementation of A. D. Code (1950), Sections III-V, for the gray (n,1) approximation to compute local polarization. 
Apply to find net polarization.'''
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import Utilities.prelude as prel


class Code1950Atmosphere:
    """
    Numerical implementation of A. D. Code (1950), Sections III-V,
    for the gray (n,1) approximation.

    Parameters
    ----------
    lam : float
        Code's lambda = kappa_abs / (sigma + kappa_abs).
        lam = 0 -> pure electron scattering
        lam = 1 -> pure absorption
    n : int
        Order of Code's approximation. The angular quadrature has 2*n
        Gauss-Legendre directions (zeros of P_{2n}).
    F : float
        Net flux normalization. Polarization fractions are independent of F.

    Notes
    -----
    This follows Code's eqs. (19)-(20), the deep linear solution eq. (39),
    the zero-incident-radiation boundary condition eq. (44) through (45, 46), and the formal
    solution for the emergent radiation eq. (69).

    The matrix/eigenmode formulation avoids explicitly solving Code's
    characteristic polynomial.
    """

    def __init__(self, lam=0.5, n=3, F=1.0):
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must lie between 0 and 1.")
        if n < 1:
            raise ValueError("n must be >= 1.")

        self.lam = float(lam)
        self.n = int(n)
        self.F = float(F)

        self.mu_quad, self.a_quad = leggauss(2*self.n) # Gauss-Legendre quadrature to compute the and mu integrals in Code's eqs. (11)-(12) as he did in eqs. (19)-(20).
        self.N = len(self.mu_quad)

        self.A = self._build_transfer_matrix() 
        self._solve_surface_modes() # solve radiative transfer using matrix A

    def _build_transfer_matrix(self):
        """
        Build y' = A y from Code eqs. (19)-(20).
        y = [I_l(mu_1)...I_l(mu_2n), I_r(mu_1)...I_r(mu_2n)]
        i.e. we have to solve for I_l and I_r at each of the 2n quadrature directions, so we have 4n eqs 
        but we will discard the 2n for diverging modes
        """
        mu = self.mu_quad
        a = self.a_quad
        lam = self.lam
        N = self.N

        A = np.zeros((2*N, 2*N), dtype=float)

        for i, mui in enumerate(mu):

            # Eq. (19): source term feeding I_l(mu_i)
            coeff_ll = (
                (3.0/8.0)*(1.0-lam) *
                (
                    2.0*a*(1.0-mu**2)
                    + mui**2 * a*(3.0*mu**2 - 2.0)
                )
                + (lam/4.0)*a
            )

            coeff_lr = (
                (3.0/8.0)*(1.0-lam) *
                (mui**2 * a)
                + (lam/4.0)*a
            )

            # Eq. (20): source term feeding I_r(mu_i)
            coeff_rl = (
                (3.0/8.0)*(1.0-lam) *
                (a*mu**2)
                + (lam/4.0)*a
            )

            coeff_rr = (
                (3.0/8.0)*(1.0-lam) *
                a
                + (lam/4.0)*a
            )

            # mu_i dI/dtau = I - S
            A[i, :N] = -coeff_ll / mui
            A[i, N:] = -coeff_lr / mui
            A[i, i] += 1.0 / mui

            A[N+i, :N] = -coeff_rl / mui
            A[N+i, N:] = -coeff_rr / mui
            A[N+i, N+i] += 1.0 / mui

        return A

    def _solve_surface_modes(self):
        """
        Construct the general semi-infinite general solution, as linear term + exponential modes
        The linear term is the deep-atmosphere solution carrying the constant net flux, given by: 
            I_l = I_r = b (tau + mu + Q),  with b = 3F/8 

        The exponential modes only modify it near the surface and are defined to respect the boundary conditions.
        """
        vals, vecs = eig(self.A) # returns ordinary eigenvectors (i.e. of the form y ~ eigenvec * exp(eigenvalue * tau)) for Av = kv, not the full Jordan chain.

        # We want to exclude: 
        # - the mode for k = 0 (which we compute separately from eq.39)
        # - the divergent mode (which is not physically relevant). this corresponds to the eigenvalue with Re(eigenvalue) < 0.
        tol = 1e-6
        keep = np.where(np.real(vals) < -tol)[0]

        # scipy.linalg.eig(A) returns complex-valued arrays even if, as in this case, the eigvals are real (i.e. it gives like 5+0i or with a small imaginary part due to numerical noise)
        eigvecs = np.real(vecs[:, keep])
        if np.max(np.abs(np.imag(vals))) > 1e-5:
            raise ValueError("Unexpected complex eigenvalues: " + str(vals))
        if np.max(np.abs(np.imag(vecs))) > 1e-5:
            raise ValueError("Unexpected complex eigenvectors: " + str(vecs))
        eigvals = np.real(vals[keep]) 

        # order = np.argsort(eigvals)
        # eigvals = eigvals[order]
        # eigvecs = eigvecs[:, order]

        # There should be 2n-1 decaying modes because the 2n+1 roots of Code's characteristic polynomial are:
        #   2n-1 decaying modes (Re(eigenvalue) < 0)
        #   2n-1 growing modes (Re(eigenvalue) > 0)
        #   2 near-zero roots (Re(eigenvalue) ~ 0)
        expected = 2*self.n - 1
        if len(eigvals) != expected:
            raise RuntimeError(
                f"Expected {expected} decaying modes, found {len(eigvals)}. "
                "Try changing the numerical tolerance."
            )

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        b = 3.0*self.F/8.0
        self.b = b

        # To find the 2n costants (Lb, Ma and Q), we apply to compute the solution of the rad transf. with the boundary conditions at the surface (tau = 0):
        # Incoming rays at the surface have I = 0 (i.e. no incident radiation).
        mu_dup = np.concatenate((self.mu_quad, self.mu_quad))
        ones = np.ones(2*self.N)

        inc = np.where(self.mu_quad < 0.0)[0]
        # Il entries are followed by Ir entries
        incoming = np.concatenate((inc, self.N + inc))

        # At tau=0: 0 = b(mu + Q) + sum_m c_m v_m -->  bQ + sum_m c_m v_m = -b mu 
        # so on the left side you have the unknowns (Q, c_1, c_2, ..., c_{2n-1}) and on the right side you have -b mu.
        M = np.column_stack((
            b*ones[incoming],
            eigvecs[incoming, :]
        ))
        rhs = -b*mu_dup[incoming] 

        x = np.linalg.solve(M, rhs)

        self.Q = x[0]
        self.mode_amplitudes = x[1:]

        self.y_surface = (
            b*mu_dup
            + b*self.Q*ones
            + eigvecs @ self.mode_amplitudes
        )

    def discrete_surface_intensities(self):
        """
        Intensities at Code's Gauss-Legendre quadrature directions.

        Returns
        -------
        mu, Il, Ir
        """
        N = self.N
        return (
            self.mu_quad.copy(),
            self.y_surface[:N].copy(),
            self.y_surface[N:].copy()
        )

    def _y_at_tau(self, tau):
        """I_l and I_r at the quadrature directions as functions of tau."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))

        mu_dup = np.concatenate((self.mu_quad, self.mu_quad))

        linear = self.b * (
            tau[:, None]
            + mu_dup[None, :]
            + self.Q
        )

        modes = (
            np.exp(np.outer(tau, self.eigvals))
            * self.mode_amplitudes[None, :]
        ) @ self.eigvecs.T

        return linear + modes

    def _source_function(self, tau, mu_out):
        """
        Compute S_l(tau,mu_out), S_r(tau,mu_out)
        from the quadrature moments of the self-consistent solution.
        """
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        mu_out = np.atleast_1d(np.asarray(mu_out, dtype=float))

        Y = self._y_at_tau(tau)

        N = self.N
        mu = self.mu_quad
        a = self.a_quad
        lam = self.lam

        Il = Y[:, :N]
        Ir = Y[:, N:]

        # Code's moments: J = 1/2 integral I dmu,
        #                 K = 1/2 integral mu^2 I dmu
        # but we use Gauss-Legendre quadrature to compute the integrals (Eq.55,56,57).
        Jl = 0.5*np.sum(Il*a[None, :], axis=1)
        Jr = 0.5*np.sum(Ir*a[None, :], axis=1)
        Kl = 0.5*np.sum(Il*(a*mu**2)[None, :], axis=1)

        # Eq.65-66 (integrated gray source functions from eqs. (11)-(12), using B = J_l + J_r from eq. (16)).
        Sl = (
            (3.0/4.0)*(1.0-lam) *
            (
                2.0*(Jl-Kl)[:, None]
                + mu_out[None, :]**2 *
                  (3.0*Kl - 2.0*Jl + Jr)[:, None]
            )
            + (lam/2.0)*(Jl+Jr)[:, None]
        )

        Sr_1d = (
            (3.0/4.0)*(1.0-lam)*(Jr+Kl)
            + (lam/2.0)*(Jl+Jr)
        )

        Sr = np.repeat(Sr_1d[:, None], len(mu_out), axis=1)

        return Sl, Sr

    def emergent(self, mu, x_max=35.0, nx=6000):
        """
        Emergent I_l(0,mu), I_r(0,mu) for arbitrary 0 <= mu <= 1.

        Uses Code's formal solution (eq. 69):
            I(0,mu) = integral_0^infty S(tau,mu) exp(-tau/mu) d tau / mu

        We integrate using x=tau/mu, so:
            I(0,mu) = integral_0^infty S(mu*x,mu) exp(-x) dx

        Parameters
        ----------
        mu : float or array
            Outgoing cosine relative to local plane-parallel normal.
        """
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))

        if np.any(mu_arr < 0.0) or np.any(mu_arr > 1.0):
            raise ValueError("For emergent radiation, mu must lie in [0,1].")

        Il_out = np.empty_like(mu_arr)
        Ir_out = np.empty_like(mu_arr)

        x = np.linspace(0.0, x_max, nx)
        expx = np.exp(-x)

        for j, muj in enumerate(mu_arr):

            if muj == 0.0:
                # Limb limit: formal solution tends to source function at tau=0.
                Sl, Sr = self._source_function(
                    np.array([0.0]), np.array([0.0])
                )
                Il_out[j] = Sl[0, 0]
                Ir_out[j] = Sr[0, 0]
                continue

            tau = muj*x
            Sl, Sr = self._source_function(tau, np.array([muj]))

            Il_out[j] = trapezoid(Sl[:, 0]*expx, x)
            Ir_out[j] = trapezoid(Sr[:, 0]*expx, x)

        I = Il_out + Ir_out
        Q = Ir_out - Il_out
        P = Q / I

        if np.ndim(mu) == 0:
            return Il_out[0], Ir_out[0], I[0], Q[0], P[0]

        return Il_out, Ir_out, I, Q, P

def compute_polarization_code(
        Fx, Fy, Fz,
        n_obs,
        atmosphere,
        weight=None,
        all_data=False
    ):
    """
    Compute net polarization using the local plane-parallel
    solution of Code (1950).

    Parameters
    ----------
    Fx, Fy, Fz :
        Components of the local flux vector.

        The direction of the flux is used as the local
        plane-parallel normal.

        The magnitude |F| is used to normalize the local
        Code intensity.

    n_obs :
        Observer direction.

    atmosphere :
        Code1950Atmosphere instance, e.g.

            atmosphere = Code1950Atmosphere(
                lam=0.0,
                n=3,
                F=1.0
            )

    weight :
        Optional additional weight for each cell.
        For example surface area, volume weight, etc.

        If the points are photospheric surface elements,
        this could contain dA, while the projected-area
        factor mu is applied below.

    all_data :
        If True, also return local/cell Stokes quantities.

    Returns
    -------
    P, I, Q, U
        Net polarization fraction and Stokes parameters.
    """

    # Observer directions and planes normal
    n = np.asarray(n_obs, dtype=float)
    n /= np.linalg.norm(n)
    F_vec = np.column_stack((Fx, Fy, Fz))
    F_mag = np.linalg.norm(F_vec, axis=1)
    good_flux = F_mag > 1e-20
    F_hat = np.zeros_like(F_vec)
    F_hat[good_flux] = (
        F_vec[good_flux] /
        F_mag[good_flux, None])


    mu = F_hat @ n
    # outward-going radiation only
    visible = (mu > 0.0) & good_flux
    mu_vis = mu[visible]
    Fhat_vis = F_hat[visible]
    Fmag_vis = F_mag[visible]

    if len(mu_vis) == 0:
        if all_data:
            return 0., 0., 0., 0., {}
        return 0., 0., 0., 0.

    # CODE LOCAL SOLUTION
    # atmosphere is constructed with F=1, then scaled by the actual local flux magnitude.
    Il_unit, Ir_unit, I_unit, Q_unit, P_local = \
        atmosphere.emergent(mu_vis) 
    Il_local = Fmag_vis * Il_unit
    Ir_local = Fmag_vis * Ir_unit
    I_local = Fmag_vis * I_unit
    Q_code = Fmag_vis * Q_unit

    # Fixed observer sky basis
    tmp = np.array([0.0, 0.0, 1.0])
    e2 = tmp - np.dot(tmp, n) * n
    if np.linalg.norm(e2) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0])
        e2 = tmp - np.dot(tmp, n) * n
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n)
    e1 /= np.linalg.norm(e1)


    # From Code
    # Il = E vector IN the meridian plane
    # Ir = E vector PERPENDICULAR to meridian plane
    # where the meridian plane contains F_hat and n.
    # We defined:
    # Q_code = Ir - Il
    # Therefore +Q_code points perpendicular to the local meridian plane (so in the direction of n x F_hat)

    e_pol = np.cross(n, Fhat_vis)
    e_pol_mag = np.linalg.norm(e_pol, axis=1)

    # At mu=1 polarization should be zero, so the
    # polarization direction is undefined but irrelevant.
    good_pol = e_pol_mag > 1e-12

    e_pol[good_pol] /= e_pol_mag[good_pol, None]

    # arbitrary harmless direction for exactly mu=1
    e_pol[~good_pol] = e1

    # Rotate into the common observer basis
    cos_phi = e_pol @ e1
    sin_phi = e_pol @ e2
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2.0 * cos_phi * sin_phi
    Q_cell = Q_code * cos2phi
    U_cell = Q_code * sin2phi

    # --------------------------------------------------
    # Weights
    # --------------------------------------------------

    if weight is None:
        w = np.ones_like(I_local)

    elif np.isscalar(weight):
        w = np.full_like(I_local, weight, dtype=float)

    else:
        weight = np.asarray(weight)
        w = weight[visible]

    # --------------------------------------------------
    # If these are surface elements:
    #
    # observed contribution = I(mu) * mu * dA
    #
    # Here mu is therefore the projected-area factor.
    #
    # If your "weight" already contains projected area,
    # remove this extra mu.
    # --------------------------------------------------

    w_obs = w * mu_vis

    # --------------------------------------------------
    # Net Stokes
    # --------------------------------------------------

    I = np.sum(w_obs * I_local)

    Q = np.sum(w_obs * Q_cell)
    U = np.sum(w_obs * U_cell)

    P = np.sqrt(Q**2 + U**2) / (I + 1e-30)

    if all_data:

        data = {
            "visible": visible,
            "mu": mu_vis,

            "Il_local": Il_local,
            "Ir_local": Ir_local,

            "I_local": I_local,
            "Q_code": Q_code,
            "U_code": np.zeros_like(Q_code),
            "P_local": P_local,

            "cos2phi": cos2phi,
            "sin2phi": sin2phi,

            "Q_cell": Q_cell,
            "U_cell": U_cell,

            "weight_obs": w_obs
        }

        return P, I, Q, U, data

    return P, I, Q, U

if __name__ == "__main__":

    # Example 1: reproduce the qualitative lambda=0.5 Table-3 curve
    atmosphere = Code1950Atmosphere(lam=0.5, n=3, F=1.0)

    mu = np.arange(0.0, 1.01, 0.1)
    Il, Ir, I, Q, P = atmosphere.emergent(mu)

    print("lambda = 0.5, n = 3")
    print(" mu       Il/F       Ir/F       P")
    for m, il, ir, p in zip(mu, Il, Ir, P):
        print(f"{m:3.1f}   {il:9.6f}  {ir:9.6f}  {p:9.6f}")

    # Example 2: pure scattering
    pure_scattering = Code1950Atmosphere(lam=0.0, n=3, F=1.0)
    _, _, _, _, P0 = pure_scattering.emergent(mu)

    print("\npure scattering (lambda = 0)")
    print(" mu       P")
    for m, p in zip(mu, P0):
        print(f"{m:3.1f}   {p:9.6f}")

    plt.plot(mu, P0, label=r"$\lambda=0$")
    plt.plot(mu, P, label=r"$\lambda=0.5$")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$P_{\rm loc}$")
    plt.legend()


    atm = Code1950Atmosphere(
    lam=0.0,   # pure scattering
    n=3,
    F=1.0)

    Fx_obs = np.zeros(10)
    Fy_obs = np.zeros(10)
    Fz_obs = 2.0 * np.ones(10)
    n_obs = np.array([1, 0.0, 1e-4])
    n_obs /= np.linalg.norm(n_obs)
    P, I, Q, U = compute_polarization_code(
        Fx_obs,
        Fy_obs,
        Fz_obs,
        n_obs,
        atmosphere=atm)

    print(f"\nNet polarization for uniform vertical flux for obs {n_obs}:")
    print(f"P = {P:.6f}, I = {I:.6f}, Q = {Q:.6f}, U = {U:.6f}")
    # Example for a photospheric element:
    #
    # n_ph  = local photosphere normal
    # n_obs = observer direction
    #
    # mu_obs = dot(n_ph, n_obs)
    #
    # if mu_obs > 0:
    #     Il, Ir, I, Qlocal, Plocal = atmosphere.emergent(mu_obs)
