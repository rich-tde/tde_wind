'''Numerical implementation of A. D. Code (1950), Sections III-V, for the gray (n,1) approximation to compute local polarization. 
Apply to find net polarization.'''
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig
from scipy.integrate import trapezoid

lam_sc = 0.0
lam_abs = 1.0
lam_code = 0.5
lam_ShSu = 1/6

def single_scattering(mu, F=1.0):
    """
    Single Thomson scattering of an initially unpolarized
    beam travelling along the local normal.

    Parameters
    ----------
    mu :
        cos(scattering angle) = n_local . n_obs

    F :
        Arbitrary normalization of the incident radiation.

    Returns
    -------
    Il, Ir, I, Q, P

    Convention:
        Q = Ir - Il
    """

    mu = np.asarray(mu, dtype=float)

    # Thomson scattering:
    # perpendicular component ~ 1
    # parallel component      ~ mu^2
    
    # normalization is arbitrary here.
    Ir = F * np.ones_like(mu)
    Il = F * mu**2

    I = Ir + Il
    Q = Ir - Il

    P = Q / I

    return Il, Ir, I, Q, P

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
        vals, vecs = eig(self.A) # returns ordinary eigenvectors for Av = kv (i.e. of the form y ~ eigenvec * exp(eigenvalue * tau)), which you exepct to be 2n independent, not the full Jordan chain.

        # We want to exclude: 
        # - the mode for k = 0 (which we compute separately from eq.39). 
        # The zero eigenvalue has a an ordinary solution v0 s.t. Av0 = 0, which is found with eig(A), but also a generalized eigenvector w satisfying Aw=v0. 
        # The pair v0,w form a Jordan chain and so y = w + \tau v0 is a solution
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
        atmospheres,
        x=None, y=None, z=None,
        weight=None,
        area_weight=False,
        atmospheres_unit_flux=True,
        all_data=False
    ):
    """
    Compute net polarization using the local plane-parallel
    solution of Code (1950).

    Parameters
    ----------
    Fx, Fy, Fz : arrays
        Local radiative-flux components.

        The local flux direction is used as an approximation
        to the local plane-parallel normal:

            n_local ~ F / |F|

    n_obs : array-like, shape (3,)
        Observer direction.

    atmospheres :
        Either:
        1. a single Code1950Atmosphere instance, used for
           every patch;
        2. a list/array of Code1950Atmosphere objects,
           one atmosphere per patch.
        The second case allows lambda to vary across the
        photosphere.

    x, y, z : arrays, optional
        Photospheric positions.
        Required only if area_weight=True.

    weight : scalar or array, optional
        Additional user-supplied weight.

        For example, if you already know the surface area dA
        of each patch, pass it here.

    area_weight : bool
        If True, estimate dA from the HEALPix photospheric
        positions x,y,z. The final observed weight is then mu * dA

        If False, the supplied `weight` is treated as dA
        (or simply 1 if weight=None), and the projected-area
        factor mu is still applied.

    atmospheres_unit_flux : bool
        If True, atmospheres were constructed with F=1,
        and their I,Q are scaled by the local |F|.

        If False, each atmosphere already contains its
        physical flux normalization and no further scaling
        by |F| is done.

    all_data : bool
        If True, return patch-level quantities as well.

    Returns
    -------
    P, I, Q, U
        Net polarization fraction and Stokes parameters.

    optionally:
        data
    """

    n = np.asarray(n_obs, dtype=float)
    n /= np.linalg.norm(n)

    # ---------------------------------------
    # Local flux vectors
    # ---------------------------------------

    Fvec = np.column_stack((Fx, Fy, Fz))
    Fmag = np.linalg.norm(Fvec, axis=1)

    good = Fmag > 1e-30

    Fhat = np.zeros_like(Fvec)
    Fhat[good] = (
        Fvec[good]
        / Fmag[good, None]
    )

    # Approximation:
    # local plane-parallel normal = flux direction
    nhat = Fhat

    # Viewing cosine
    mu = nhat @ n
    visible = good & (mu > 0.0)
    idx_vis = np.where(visible)[0]


    if len(idx_vis) == 0:
        if all_data:
            return 0., 0., 0., 0., {}
        return 0., 0., 0., 0.

    nhat_vis = nhat[idx_vis]
    mu_vis = mu[idx_vis]
    Fmag_vis = Fmag[idx_vis]

    # Is atmospheres one object or one per patch?
    single_atmosphere = not isinstance(
        atmospheres,
        (list, tuple, np.ndarray)
    )

    # Code local solution
    Il_local = np.zeros(len(idx_vis))
    Ir_local = np.zeros(len(idx_vis))
    I_local = np.zeros(len(idx_vis))
    Q_local = np.zeros(len(idx_vis))
    P_local = np.zeros(len(idx_vis))

    if single_atmosphere:
        # vectorized call: same lambda for all patches
        Il, Ir, I, Q, P = atmospheres.emergent(mu_vis)
        if atmospheres_unit_flux:
            Il_local = Fmag_vis * Il
            Ir_local = Fmag_vis * Ir
            I_local  = Fmag_vis * I
            Q_local  = Fmag_vis * Q
        else:
            Il_local = Il
            Ir_local = Ir
            I_local  = I
            Q_local  = Q

        P_local = P

    else:
        # each patch has its own atmosphere / lambda
        if len(atmospheres) != len(Fx):
            raise ValueError(
                "If atmospheres is a list, it must have "
                "one entry per photospheric patch.")

        for j, k in enumerate(idx_vis):
            atm_k = atmospheres[k]
            Il, Ir, I, Q, P = \
                atm_k.emergent(mu[k])

            if atmospheres_unit_flux:
                Il *= Fmag[k]
                Ir *= Fmag[k]
                I  *= Fmag[k]
                Q  *= Fmag[k]

            Il_local[j] = Il
            Ir_local[j] = Ir
            I_local[j]  = I
            Q_local[j]  = Q
            P_local[j]  = P

    # --------------------------------------------------
    # Fixed observer sky basis
    # --------------------------------------------------

    tmp = np.array([0.0, 0.0, 1.0])

    e2 = tmp - np.dot(tmp, n) * n

    if np.linalg.norm(e2) < 1e-10:
        tmp = np.array([1.0, 0.0, 0.0])
        e2 = tmp - np.dot(tmp, n) * n

    e2 /= np.linalg.norm(e2)

    e1 = np.cross(e2, n)
    e1 /= np.linalg.norm(e1)

    # ---------------------------------------
    # Positive local-Q direction
    #
    # Q_local = Ir - Il, therefore positive Q
    # is perpendicular to the meridian plane
    # ---------------------------------------

    e_pol = np.cross(n, nhat_vis)

    e_pol_mag = np.linalg.norm(e_pol, axis=1)

    good_pol = e_pol_mag > 1e-12

    e_pol[good_pol] /= e_pol_mag[good_pol, None]

    # at mu=1 polarization vanishes anyway
    e_pol[~good_pol] = e1

    # ---------------------------------------
    # Rotate local Q to common sky Q,U
    # ---------------------------------------

    cos_phi = e_pol @ e1
    sin_phi = e_pol @ e2

    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2.0 * cos_phi * sin_phi

    Q_sky =  Q_local * cos2phi
    U_sky = Q_local * sin2phi

    # Base patch weight
    if weight is None:
        w = np.ones(len(idx_vis))

    elif np.isscalar(weight):
        w = np.full( len(idx_vis),
                    weight,
                    dtype=float)

    else:
        weight = np.asarray(
            weight,
            dtype=float)

        w = weight[idx_vis]

    # --------------------------------------------------
    # Surface area
    if area_weight:
        if x is None or y is None or z is None:
            raise ValueError(
                "x,y,z are required when "
                "area_weight=True.")

        r_vec = np.column_stack((x, y, z))[idx_vis]
        r = np.linalg.norm(r_vec,axis=1)
        r_hat = (r_vec/ np.maximum(r[:, None], 1e-30))

        # dA = r^2 dOmega / |n_local . r_hat|
        dOmega = 4.0 * np.pi / len(x)
        cos_nr = np.abs(np.sum(nhat_vis * r_hat, axis=1))

        dA = r**2* dOmega/ np.maximum(cos_nr, 1e-8)
        w *= dA

    # --------------------------------------------------
    # Projected surface area
    #
    # observed flux contribution:
    #
    #      I(mu) * mu * dA
    #
    # So mu must be included regardless of whether
    # dA came from `weight` or area_weight=True.
    # --------------------------------------------------

    w_obs = (w * mu_vis)

    # Net Stokes
    I_tot = np.sum(w_obs* I_local)
    Q_tot = np.sum(w_obs* Q_sky)
    U_tot = np.sum(w_obs* U_sky)
    P_tot = (np.sqrt(Q_tot**2+ U_tot**2)/ (I_tot + 1e-30))

    # Optional local data
    if all_data:

        data = {
            "visible": visible,
            "indices": idx_vis,

            "mu": mu_vis,

            "F_mag": Fmag_vis,
            "F_hat": nhat_vis,

            "Il_local": Il_local,
            "Ir_local": Ir_local,
            "I_local": I_local,

            "Q_local": Q_local,
            "U_local": np.zeros_like(Q_local),
            "P_local": P_local,

            "cos2phi": cos2phi,
            "sin2phi": sin2phi,

            "Q_sky": Q_sky,
            "U_sky": U_sky,

            "weight_obs": w_obs
        }

        return (P_tot,
                I_tot,
                Q_tot,
                U_tot,
                data)

    return (P_tot,
            I_tot,
            Q_tot,
            U_tot)


if __name__ == "__main__":
    import healpy as hp
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    import Utilities.prelude as prel
    from src.Polarization.geometries import ellipsoid_surface, ellipsoid_unit_normal
    #%% Example 1: reproduce the qualitative lambda=0.5 Table-3 curve
    atmosphere = Code1950Atmosphere(lam=lam_code, n=3, F=1.0)

    mu = np.arange(0.0, 1.01, 0.1)
    Il, Ir, I, Q, P = atmosphere.emergent(mu)

    print(f"lambda = {lam_code}, n = 3")
    print(" mu       Il/F       Ir/F       P")
    for m, il, ir, p in zip(mu, Il, Ir, P):
        print(f"{m:3.1f}   {il:9.6f}  {ir:9.6f}  {p:9.6f}")

    #%% Example 2: pure scattering
    pure_scattering = Code1950Atmosphere(lam=lam_sc, n=3, F=1.0)
    _, _, _, _, P0 = pure_scattering.emergent(mu)

    print(f"\npure scattering (lambda = {lam_sc})")
    print(" mu       P")
    for m, p in zip(mu, P0):
        print(f"{m:3.1f}   {p:9.6f}")

    plt.figure(figsize=(8,8))
    plt.plot(mu, P0, label=r"$\lambda=$" + f"{lam_sc}")
    plt.plot(mu, P, label=r"$\lambda=$" + f"{lam_code}")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$P_{\rm loc}$")
    plt.legend(fontsize=16)

    #%% Example 3: net polarization for uniform vertical flux (disk)
    print("\nNet polarization for uniform vertical flux (disk):")
    flux_mag = 2.0
    Fx_obs = np.zeros(10)
    Fy_obs = np.zeros(10)
    Fz_obs = flux_mag * np.ones(10)

    pure_scattering = Code1950Atmosphere(lam=lam_sc, n=3, F=flux_mag)
    n_obs_chosen = np.array([[0, 0, 1.], [1, 0.0, 1e-4], [1, 0, 1]])
    mu_chosen_obs = n_obs_chosen[:, 2] / np.linalg.norm(n_obs_chosen, axis=1)
    P_disk = np.zeros(len(n_obs_chosen))
    P_singlescatt = np.zeros(len(n_obs_chosen))
    for n_idx, n_obs in enumerate(n_obs_chosen):
        # mu_d = mu_chosen_obs[n_idx]
        print(f"\nNet polarization for uniform vertical flux for obs {n_obs}:")
        # Il, Ir, I, Q, P = pure_scattering.emergent(mu)
        P, I, Q, U = compute_polarization_code(
            Fx_obs,
            Fy_obs,
            Fz_obs,
            n_obs,
            atmospheres=pure_scattering,
            weight=None,
            area_weight=False,
            atmospheres_unit_flux=False)

        # _, _, _, _, P_singlescatt[n_idx] = single_scattering(mu_chosen_obs[n_idx], F=flux_mag)
        
        P_disk[n_idx] = P
    
        print(f"P = {P:.6f}, I = {I:.6f}, Q = {Q:.6f}, U = {U:.6f}")

    #%% Example 4: sphere
    print("\nNet polarization for spherical photosphere:")
    nside = 16
    Npix = hp.nside2npix(nside)
    observers_xyz = hp.pix2vec(nside, np.arange(Npix)) # shape: (3, 192)
    x_obs, y_obs, z_obs = observers_xyz
    Fr_obs = np.ones_like(x_obs) * 4
    Fx_obs = Fr_obs * x_obs
    Fy_obs = Fr_obs * y_obs
    Fz_obs = Fr_obs * z_obs

    P_sphere = np.zeros(len(n_obs_chosen))
    for n_idx, n_obs in enumerate(n_obs_chosen):
        # mu_d = n_obs[2] / np.linalg.norm(n_obs)
        print(f"\nNet polarization for uniform vertical flux for obs {n_obs}:")
        # Il, Ir, I, Q, P = pure_scattering.emergent(mu_d)
        P, I, Q, U = compute_polarization_code(
                    Fx_obs, Fy_obs, Fz_obs,
                    n_obs,
                    atmospheres=pure_scattering,
                    weight=None,
                    area_weight=False,
                    atmospheres_unit_flux=False)
        P_sphere[n_idx] = P
    
        print(f"P = {P:.6f}, I = {I:.6f}, Q = {Q:.6f}, U = {U:.6f}")

    #%% Plotting
    plt.figure(figsize=(8,8))
    # plt.scatter(mu_chosen_obs, P_singlescatt, s = 70, label = "single scattering", c = 'k')
    plt.plot(mu, P0, label=r"$P_{\rm loc}$", c = 'gray', ls = '--')
    plt.scatter(mu_chosen_obs, P_disk, s = 50, label = "disk")
    plt.scatter(mu_chosen_obs, P_sphere, s = 50, label = "sphere")
    plt.legend(fontsize=16)
    plt.xlabel(r"$\mu=\cos\theta$")
    plt.ylabel(r"$P_{\rm net}$")
    plt.legend(fontsize=16)

    #%% Example 5: net polarization for ellipsoid from ShapiroSutherland (1982)
    print("TEST ellipsoid ShapiroSutherland (1982)")
    a = b = 1.0 
    c_all = np.linspace(0, a, 10)
    P_ell_sc_Obl = np.zeros(len(c_all))
    P_ell_Obl = np.zeros(len(c_all))
    xi_Obl = np.zeros(len(c_all))
    n_obs_perp = np.array([1, 0.0, 1e-4]) # perpendicular to symmetry axis

    pure_scattering = Code1950Atmosphere(lam=lam_sc, n=3, F=1)
    scattering_shsu = Code1950Atmosphere(lam=lam_ShSu, n=3, F=1)
    for c_idx, c in enumerate(c_all):
        x_obs, y_obs, z_obs, dA_obs = ellipsoid_surface(1e2, a, b, c)
        xi_Obl[c_idx] = 1 - min(a,c) / max(a,c) 
        F_vec = ellipsoid_unit_normal(x_obs, y_obs, z_obs, a, b, c)
        Fx_obs, Fy_obs, Fz_obs = F_vec[:,0], F_vec[:,1], F_vec[:,2]
        if not np.allclose(np.sum(x_obs), 0, atol=1e-7):
            print(f"Warning: x-coordinates not symmetric for c={c}. sum(x) = {np.sum(x_obs)}")
        if not np.allclose(np.sum(y_obs), 0, atol=1e-7):
            print(f"Warning: y-coordinates not symmetric for c={c}. sum(y) = {np.sum(y_obs)}")
        if not np.allclose(np.sum(z_obs), 0, atol=1e-7):
            print(f"Warning: z-coordinates not symmetric for c={c}. sum(z) = {np.sum(z_obs)}")

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection = '3d')
        # ax.scatter(x_obs, y_obs, z_obs, s = 20, c = 'sandybrown')
        # ax.quiver(x_obs[::5], y_obs[::5], z_obs[::5], Fx_obs[::5], Fy_obs[::5], Fz_obs[::5], length=0.1, color='k')
        # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        # ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
        # plt.tight_layout()
        
        P, I, Q, U = compute_polarization_code(
                    Fx_obs, Fy_obs, Fz_obs,
                    n_obs_perp,
                    atmospheres=pure_scattering,
                    weight=dA_obs,
                    area_weight=False,
                    atmospheres_unit_flux=True)
        P_ell_sc_Obl[c_idx] = P

        P, I, Q, U = compute_polarization_code(
                            Fx_obs, Fy_obs, Fz_obs,
                            n_obs_perp,
                            atmospheres=scattering_shsu,
                            weight=dA_obs,
                            area_weight=False,
                            atmospheres_unit_flux=True)
        P_ell_Obl[c_idx] = P

    #%%
    c_all = np.linspace(a, 10*a, 10)
    P_ell_sc_Pro = np.zeros(len(c_all))
    P_ell_Pros = np.zeros(len(c_all))
    xi_Pro = np.zeros(len(c_all))
    n_obs_perp = np.array([1, 0.0, 1e-4]) # perpendicular to symmetry axis

    pure_scattering = Code1950Atmosphere(lam=lam_sc, n=3, F=1)
    scattering_shsu = Code1950Atmosphere(lam=lam_ShSu, n=3, F=1)
    for c_idx, c in enumerate(c_all):
        x_obs, y_obs, z_obs, dA_obs = ellipsoid_surface(1e2, a, b, c)
        xi_Pro[c_idx] = 1 - min(a,c) / max(a,c) 
        F_vec = ellipsoid_unit_normal(x_obs, y_obs, z_obs, a, b, c)
        Fx_obs, Fy_obs, Fz_obs = F_vec[:,0], F_vec[:,1], F_vec[:,2]
        if not np.allclose(np.sum(x_obs), 0, atol=1e-7):
            print(f"Warning: x-coordinates not symmetric for c={c}. sum(x) = {np.sum(x_obs)}")
        if not np.allclose(np.sum(y_obs), 0, atol=1e-7):
            print(f"Warning: y-coordinates not symmetric for c={c}. sum(y) = {np.sum(y_obs)}")
        if not np.allclose(np.sum(z_obs), 0, atol=1e-7):
            print(f"Warning: z-coordinates not symmetric for c={c}. sum(z) = {np.sum(z_obs)}")

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection = '3d')
        # ax.scatter(x_obs, y_obs, z_obs, s = 20, c = 'sandybrown')
        # ax.quiver(x_obs[::5], y_obs[::5], z_obs[::5], Fx_obs[::5], Fy_obs[::5], Fz_obs[::5], length=0.1, color='k')
        # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        # ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_zlim(-1.45, 1.45)
        # plt.tight_layout()
        
        P, I, Q, U = compute_polarization_code(
                    Fx_obs, Fy_obs, Fz_obs,
                    n_obs_perp,
                    atmospheres=pure_scattering,
                    weight=dA_obs,
                    area_weight=False,
                    atmospheres_unit_flux=True)
        P_ell_sc_Pro[c_idx] = P

        P, I, Q, U = compute_polarization_code(
                            Fx_obs, Fy_obs, Fz_obs,
                            n_obs_perp,
                            atmospheres=scattering_shsu,
                            weight=dA_obs,
                            area_weight=False,
                            atmospheres_unit_flux=True)
        P_ell_Pros[c_idx] = P
    #%%
    fig, (axO, axP) = plt.subplots(1,2, figsize=(16,8)) 
    axO.plot(xi_Obl, P_ell_sc_Obl*100, label = r"$\lambda = $" + f"{lam_sc}")
    axO.plot(xi_Obl, P_ell_Obl*100, label = r"$\lambda = $" + f"{lam_ShSu:.2f}")
    axP.plot(xi_Pro, P_ell_sc_Pro*100, label = r"$\lambda = $" + f"{lam_sc}")
    axP.plot(xi_Pro, P_ell_Pros*100, label = r"$\lambda = $" + f"{lam_ShSu:.2f}")
    axP.set_xlabel(r"$\xi = 1 - \min(a,c)/\max(a,c)$")
    axO.set_xlabel(r"$\xi = 1 - \min(a,c)/\max(a,c)$")
    axO.set_ylabel(r"$P_{\rm net}$ [%]")
    axP.set_ylabel(r"$P_{\rm net}$ [%]")
    axO.set_title('Oblate ellipsoid, observer perpendicular to symmetry axis', fontsize=16)
    axP.set_title('Prolate ellipsoid, observer perpendicular to symmetry axis', fontsize=16)
    plt.legend(fontsize=16)

    #%% Our simulation
    m = 4
    Mbh = 10**m
    beta = 1
    mstar = .5
    Rstar = .47
    n = 1.5
    compton = 'Compton'
    check = 'HiResNewAMR' 
    snap = 151
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
    observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX)) 
    x_heal, y_heal, z_heal = observers_xyz[0], observers_xyz[1], observers_xyz[2]
    observers_xyz = np.transpose(observers_xyz)
    mu_healp = z_heal / np.linalg.norm(observers_xyz, axis=1)
    phi_heal = np.arctan2(y_heal, x_heal)
    theta_heal = np.arccos(z_heal)
    longitude_heal_moll = phi_heal 
    latitude_heal_moll = np.pi/2 - theta_heal

    photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
    x, y, z, den, Fx, Fy, Fz, alpha_rossland, alpha_scatter, alpha_abs = \
        photo['x'], photo['y'], photo['z'], photo['den'], photo['Fx'], photo['Fy'], photo['Fz'], photo['alpha_rossland'], photo['alpha_scatter'], photo['alpha_abs']
    lam_sim = 1 - alpha_scatter / alpha_rossland
    # lam_median = np.median(lam_sim)

    # compute Code atmospheres (by unit flux) for each of the observers' lambda 
    atmospheres = [Code1950Atmosphere(
            lam=lam_sim[k], n=3, F=1.0)
            for k in range(len(lam_sim))]

    P_sim = np.zeros(len(observers_xyz))
    I_sim = np.zeros(len(observers_xyz))
    Q_sim = np.zeros(len(observers_xyz))
    U_sim = np.zeros(len(observers_xyz))

    for n_idx, n_obs in enumerate(observers_xyz):
        P, I, Q, U = compute_polarization_code(
                    Fx, Fy, Fz,
                    n_obs,
                    atmospheres,
                    x=x, y=y, z=z,
                    area_weight=False,
                    atmospheres_unit_flux=False
                )

        P_sim[n_idx] = P
        I_sim[n_idx] = I
        Q_sim[n_idx] = Q
        U_sim[n_idx] = U

    longitude_heal_moll = phi_heal 
    latitude_heal_moll = np.pi/2 - theta_heal
    lon_1d_heal = longitude_heal_moll
    lat_1d_heal = latitude_heal_moll
    # Define a regular grid in (lon, lat) for visualization
    nlon = 360
    nlat = 180
    lon_heal_grid = np.linspace(lon_1d_heal.min(), lon_1d_heal.max(), nlon)
    lat_heal_grid = np.linspace(lat_1d_heal.min(), lat_1d_heal.max(), nlat)
    lon_heal_mesh, lat_heal_mesh = np.meshgrid(lon_heal_grid, lat_heal_grid)
    data_grid_P = griddata(
                points=(lon_1d_heal, lat_1d_heal), 
                values=P_sim,
                xi=(lon_heal_mesh, lat_heal_mesh),
                method='linear')

    #%%
    fig = plt.figure(figsize=(10, 8))
    axP = fig.add_subplot(111, projection='mollweide')
    img = axP.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_P, cmap='viridis', vmin = 0, vmax = 0.009)  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'P')
    cbar.ax.tick_params(which='major',length = 6)

# %%
