"""Code (1950) with an additional prescribed-atmosphere, nonconstant-flux solver.
Original constant-flux class and net-polarization function retained from your file.
Use solve_simulation_patch(...) with arrays directly; no extraction or ray skipping.
"""
import sys
sys.path.append('/Users/paolamartire/shocks')
# import resource
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    save = True
else:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    abspath = '/Users/paolamartire/shocks'
    save = False
    mpl.rcParams['savefig.transparent'] = False 
    mpl.rcParams['figure.facecolor'] = 'white'    # Figure when displayed
    mpl.rcParams['axes.facecolor'] = 'white'  

import numpy as np
from numpy.polynomial.legendre import leggauss
from Utilities import prelude as prel
from src.Polarization.code1950 import Code1950Atmosphere


class Code1950NonConstantFluxAtmosphere:
    """Code's angular scattering equations with prescribed depth-dependent emission.

    Inputs are arrays at the SAME optical-depth samples:
      tau: total absorption+scattering optical depth, increasing inward from 0;
      temperature: gas temperature [K];
      lam: absorption fraction alpha_abs/(alpha_abs+alpha_scatter).
    Alternatively give alpha_abs and alpha_scatter instead of lam. Their units
    must agree; when tau is supplied, only their ratio enters the transfer ODE.
    radiation_energy_density [erg/cm^3] is optional comparison data, not a second
    imposed radiation field. The LTE thermal source is B=sigma_SB*T_gas^4/pi.

    bottom_intensity is REQUIRED: total unpolarized intensity entering upward
    at the deepest sample, either a scalar or a function of positive mu.
    Specify your boundary choice explicitly. An isotropic LTE boundary would
    be sigma_SB*T[-1]^4/pi; an isotropic estimate from radiation energy density
    would be c*u_rad[-1]/(4*pi). Neither is automatically assumed here, and
    a scalar intensity cannot prescribe a general anisotropic boundary field.
    top_intensity is the isotropic unpolarized downward incident intensity.

    Mathematical change: the old term lam*J/2 in each polarization source is
    replaced by lam(tau)*B[T(tau)]/2. The conservative scattering kernel is
    unchanged. Solve mu*y'=y-(1-lam)*C_sc*y-lam*B/2 with two boundary conditions.
    No constant flux or radiative equilibrium is imposed. No rays are skipped.

    Numerical method: solve ALL layer intensities simultaneously using a sparse
    linear system and exact exponential integration of a piecewise linear source
    (short characteristics). Tiny delta_tau does not require derivative residual
    refinement. Sources and coefficients are interpolated between supplied samples.
    Increase subdivisions and decrease max_delta_tau to check depth convergence;
    increase n to check angular convergence. This replaces the old eigenmode solve,
    which requires constant coefficients and the radiative-equilibrium closure.

    Retains stationary, gray, local plane-parallel, axisymmetric LTE emission and
    coherent Thomson scattering assumptions. Does not include velocities, lateral
    transport or Compton redistribution. Intensities are bolometric in cgs.
    """

    def __init__(self, tau, temperature, *, bottom_intensity, lam=None,
                 alpha_abs=None, alpha_scatter=None, radiation_energy_density=None,
                 top_intensity=0., n=3, subdivisions=4, max_delta_tau=0.05,
                 max_depth_nodes=200000):
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve

        self.tau_input = np.array(tau, dtype=float, copy=True)
        t = self.tau_input
        if (t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t))
                or t[0] != 0 or np.any(np.diff(t) <= 0)):
            raise ValueError('tau must be finite, strictly increasing inward, and start at 0.')
        def profile(value, name):
            v = np.asarray(value, dtype=float)
            if v.ndim == 0:
                v = np.full(t.shape, float(v))
            if v.shape != t.shape or not np.all(np.isfinite(v)):
                raise ValueError(f'{name} must be finite and scalar or match tau.')
            return v.copy()
        for value, name in ((n, 'n'), (subdivisions, 'subdivisions'),
                            (max_depth_nodes, 'max_depth_nodes')):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if not np.isfinite(max_delta_tau) or max_delta_tau <= 0:
            raise ValueError('max_delta_tau must be positive and finite.')
        self.temperature = profile(temperature, 'temperature')
        if np.any(self.temperature < 0):
            raise ValueError('temperature cannot be negative.')
        self.B_input = 5.670374419e-5*self.temperature**4/np.pi
        if not np.all(np.isfinite(self.B_input)):
            raise ValueError('Thermal source overflow: check temperature units.')
        if lam is None:
            if alpha_abs is None or alpha_scatter is None:
                raise ValueError('Supply lam OR both alpha_abs and alpha_scatter.')
            aa, asc = profile(alpha_abs, 'alpha_abs'), profile(alpha_scatter, 'alpha_scatter')
            if np.any(aa < 0) or np.any(asc < 0) or np.any(aa+asc <= 0):
                raise ValueError('Opacities must be nonnegative with positive total extinction.')
            self.lam_input = aa/(aa+asc)
        else:
            if alpha_abs is not None or alpha_scatter is not None:
                raise ValueError('Supply lam OR opacities, not both.')
            self.lam_input = profile(lam, 'lam')
        if np.any((self.lam_input < 0) | (self.lam_input > 1)):
            raise ValueError('lam is the absorption fraction and must lie in [0,1].')
        self.radiation_energy_density = None
        if radiation_energy_density is not None:
            self.radiation_energy_density = profile(radiation_energy_density, 'radiation_energy_density')
            if np.any(self.radiation_energy_density < 0):
                raise ValueError('Radiation energy density cannot be negative.')
        self.n = int(n)
        self.mu_quad, self.a_quad = leggauss(2*self.n)
        self.N = len(self.mu_quad)
        self._mu_dup = np.tile(self.mu_quad, 2)
        self._bottom = bottom_intensity
        self.top_intensity = float(top_intensity)
        if not np.isfinite(self.top_intensity) or self.top_intensity < 0:
            raise ValueError('top_intensity must be finite and nonnegative.')
        outgoing = self.mu_quad > 0
        bottom = self._bottom_value(self.mu_quad[outgoing])
        self._scale = max(np.max(self.B_input), np.max(bottom), self.top_intensity, 1e-300)

        # Reuse the original angular kernel without invoking its eigenmode solve.
        self.lam = 0.
        D = 2*self.N
        A_sc = Code1950Atmosphere._build_transfer_matrix(self)
        self.C_sc = np.eye(D)-self._mu_dup[:, None]*A_sc
        # Fine spacing near the surface; progressively larger spacing at depth.
        grid = [t[0]]

        for left, right in zip(t[:-1], t[1:]):
            # Preserve subdivisions within every original profile interval.
            anchors = np.linspace(left, right, int(subdivisions) + 1)

            for a, b in zip(anchors[:-1], anchors[1:]):
                current = a

                while current < b:
                    step = max_delta_tau * max(1.0, current)
                    next_tau = min(b, current + step)

                    if next_tau <= current:
                        raise ValueError("Depth step is below floating-point precision.")

                    grid.append(next_tau)
                    current = next_tau

                    if len(grid) > max_depth_nodes:
                        raise ValueError("Requested grid exceeds max_depth_nodes.")

        self.tau = np.asarray(grid)
        eps = np.interp(self.tau, t, self.lam_input)
        B = np.interp(self.tau, t, self.B_input)/self._scale
        M = len(self.tau)
        self.lam_grid = eps
        self.B_grid = B*self._scale
        index = np.arange(D)
        rows, cols, data = [], [], []
        rhs = np.zeros(M*D)
        eye = np.eye(D)
        for j, dt in enumerate(np.diff(self.tau)):
            d = dt/np.abs(self._mu_dup)
            e = np.exp(-d)
            w = -np.expm1(-d)
            g = np.empty_like(d)
            small = d < 1e-3
            v = d[small]
            g[small] = v/2-v**2/3+v**3/8-v**4/30+v**5/144
            g[~small] = w[~small]/d[~small]-e[~small]
            current = np.where(self._mu_dup < 0, j+1, j)
            previous = np.where(self._mu_dup < 0, j, j+1)
            equation = current*D+index
            local = eye-((w-g)*(1-eps[current]))[:, None]*self.C_sc
            upstream = -e[:, None]*eye-(g*(1-eps[previous]))[:, None]*self.C_sc
            for node, block in ((current, local), (previous, upstream)):
                rows.append(np.repeat(equation, D))
                cols.append((node[:, None]*D+index[None, :]).ravel())
                data.append(block.ravel())
            rhs[equation] = ((w-g)*eps[current]*B[current]/2
                             + g*eps[previous]*B[previous]/2)
        incoming_idx = np.flatnonzero(self._mu_dup < 0)
        outgoing_idx = np.flatnonzero(self._mu_dup > 0)
        bc_rows = np.r_[incoming_idx, (M-1)*D+outgoing_idx]
        rows.append(bc_rows); cols.append(bc_rows); data.append(np.ones(D))
        rhs[incoming_idx] = self.top_intensity/(2*self._scale)
        rhs[(M-1)*D+outgoing_idx] = np.tile(bottom, 2)/(2*self._scale)
        matrix = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(M*D, M*D)).tocsc()
        y = spsolve(matrix, rhs)
        if not np.all(np.isfinite(y)):
            raise RuntimeError('Layer transfer linear solve produced nonfinite intensities.')
        self.linear_residual = float(np.max(np.abs(matrix @ y-rhs)))
        if self.linear_residual > 1e-8:
            raise RuntimeError(f'Layer linear-system residual too large: {self.linear_residual:g}.')
        self.y_grid = y.reshape(M, D)*self._scale
        self.y_surface = self.y_grid[0]

    def _bottom_value(self, mu):
        value = self._bottom(mu) if callable(self._bottom) else self._bottom
        value = np.broadcast_to(np.asarray(value, dtype=float), np.shape(mu)).copy()
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError('bottom_intensity must return finite nonnegative total intensity.')
        return value

    def _y_at_tau(self, tau):
        t = np.atleast_1d(np.asarray(tau, dtype=float))
        if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any((t < 0) | (t > self.tau[-1])):
            raise ValueError('Requested depth lies outside the supplied atmosphere.')
        return np.column_stack([np.interp(t, self.tau, row) for row in self.y_grid.T])

    def discrete_surface_intensities(self):
        return self.mu_quad.copy(), self.y_surface[:self.N].copy(), self.y_surface[self.N:].copy()

    def _source_function(self, tau, mu_out):
        t = np.atleast_1d(np.asarray(tau, dtype=float))
        m = np.atleast_1d(np.asarray(mu_out, dtype=float))
        y = self._y_at_tau(t)
        Il, Ir = y[:, :self.N], y[:, self.N:]
        Jl, Jr = .5*(Il @ self.a_quad), .5*(Ir @ self.a_quad)
        Kl = .5*(Il @ (self.a_quad*self.mu_quad**2))
        eps = np.interp(t, self.tau_input, self.lam_input)
        B = np.interp(t, self.tau_input, self.B_input)
        Sl = (.75*(1-eps)[:, None]*(2*(Jl-Kl)[:, None]
              + m[None, :]**2*(3*Kl-2*Jl+Jr)[:, None])+(eps*B/2)[:, None])
        Sr = .75*(1-eps)*(Jr+Kl)+eps*B/2
        return Sl, np.broadcast_to(Sr[:, None], Sl.shape)

    def emergent(self, mu):
        """Same return signature as your original solver: Il,Ir,I,Q, signed Q/I."""
        m = np.atleast_1d(np.asarray(mu, dtype=float))
        if m.ndim != 1 or not np.all(np.isfinite(m)) or np.any((m < 0) | (m > 1)):
            raise ValueError('mu must be scalar or 1D and in [0,1].')
        Sl, Sr = self._source_function(self.tau, m)
        outputs = []
        for source in (Sl, Sr):
            out = np.empty_like(m)
            for k, cosine in enumerate(m):
                if cosine == 0:
                    out[k] = source[0, k]
                    continue
                d = np.diff(self.tau)/cosine
                w = -np.expm1(-d); g = np.empty_like(d)
                small = d < 1e-3; v = d[small]
                g[small] = v/2-v**2/3+v**3/8-v**4/30+v**5/144
                g[~small] = w[~small]/d[~small]-np.exp(-d[~small])
                out[k] = (np.sum(np.exp(-self.tau[:-1]/cosine)
                                *(source[:-1, k]*w+np.diff(source[:, k])*g))
                          + float(self._bottom_value(cosine))/2*np.exp(-self.tau[-1]/cosine))
            outputs.append(out)
        Il, Ir = outputs
        I, Q = Il+Ir, Ir-Il
        P = np.divide(Q, I, out=np.zeros_like(I), where=I != 0)
        if np.ndim(mu) == 0:
            return tuple(float(v[0]) for v in (Il, Ir, I, Q, P))
        return Il, Ir, I, Q, P

    def moments(self, tau=None):
        """Flux is an OUTPUT: F=2*pi*integral mu*(Il+Ir) dmu.

        dF_dtau_exchange is the transfer energy-exchange expression
        4*pi*lam*(J-B), not a numerical derivative of the interpolated flux.
        Its integrated agreement converges as depth resolution is increased.
        """
        t = self.tau_input if tau is None else np.atleast_1d(np.asarray(tau, dtype=float))
        y = self._y_at_tau(t); I = y[:, :self.N]+y[:, self.N:]
        J = .5*(I @ self.a_quad)
        F = 2*np.pi*(I @ (self.a_quad*self.mu_quad))
        eps = np.interp(t, self.tau_input, self.lam_input)
        B = np.interp(t, self.tau_input, self.B_input)
        result = dict(tau=t.copy(), J=J, F=F, B=B, lam=eps,
                      u_rad=4*np.pi*J/2.99792458e10,
                      dF_dtau_exchange=4*np.pi*eps*(J-B))
        if self.radiation_energy_density is not None:
            result['u_rad_sim'] = np.interp(t, self.tau_input, self.radiation_energy_density)
        return result

    def flux(self, tau=None):
        value = self.moments(tau)['F']
        return float(value[0]) if tau is not None and np.ndim(tau) == 0 else value


def solve_simulation_patch(tau, temperature, alpha_abs, alpha_scatter, *,
                           bottom_intensity, radiation_energy_density=None, n=3,
                           subdivisions=4, max_delta_tau=.05):
    """Direct entry point: pass your simulation arrays, without ray-file handling."""
    return Code1950NonConstantFluxAtmosphere(
        tau, temperature, alpha_abs=alpha_abs, alpha_scatter=alpha_scatter,
        bottom_intensity=bottom_intensity, radiation_energy_density=radiation_energy_density,
        n=n, subdivisions=subdivisions, max_delta_tau=max_delta_tau)


def compute_polarization_code(
        Fx, Fy, Fz,
        n_obs,
        atmospheres,
        dA = None,
        atmospheres_are_per_unit_flux = True,
        all_data = False
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

    dA : scalar or array, optional
        Surface area of the patch.

    atmospheres_are_per_unit_flux : bool
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

    atmosphere_items = atmospheres if isinstance(atmospheres, (list, tuple, np.ndarray)) else [atmospheres]
    if atmospheres_are_per_unit_flux and any(isinstance(a, Code1950NonConstantFluxAtmosphere) for a in atmosphere_items):
        raise ValueError('Set atmospheres_are_per_unit_flux=False for physical profile intensities.')
    n_hat_obs = np.array(n_obs, dtype=float, copy=True)
    n_hat_obs /= np.linalg.norm(n_hat_obs)

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
    nhat_loc = Fhat

    # Viewing cosine
    eta = nhat_loc @ n_hat_obs
    visible = good & (eta > 0.0)
    idx_vis = np.where(visible)[0]

    if len(idx_vis) == 0:
        if all_data:
            return 0., 0., 0., 0., {}
        return 0., 0., 0., 0.

    nhat_loc_vis = nhat_loc[idx_vis]
    eta_vis = eta[idx_vis]
    Fmag_vis = Fmag[idx_vis]

    # Is atmospheres one object or one per patch?
    single_atmosphere = not isinstance(
        atmospheres,
        (list, tuple, np.ndarray)) #i.e. True if atmospheres is not a list/tuple/NumPy array

    # Code local solution
    Il_local = np.zeros(len(idx_vis))
    Ir_local = np.zeros(len(idx_vis))
    I_local = np.zeros(len(idx_vis))
    Q_local = np.zeros(len(idx_vis))
    P_local = np.zeros(len(idx_vis))

    if single_atmosphere:
        # vectorized call: same lambda for all patches
        Il, Ir, I, Q, P = atmospheres.emergent(eta_vis)
        if atmospheres_are_per_unit_flux:
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
                "Atmospheres must have one entry per photospheric patch.")

        for j, k in enumerate(idx_vis): # because we eta comes from the direction of the flux, so what is visible
            atm_k = atmospheres[k]
            Il, Ir, I, Q, P = \
                atm_k.emergent(eta[k])

            if atmospheres_are_per_unit_flux:
                Il *= Fmag[k]
                Ir *= Fmag[k]
                I  *= Fmag[k]
                Q  *= Fmag[k]

            Il_local[j] = Il
            Ir_local[j] = Ir
            I_local[j]  = I
            Q_local[j]  = Q
            P_local[j]  = P

    # Fixed observer sky basis
    tmp = np.array([0.0, 0.0, 1.0])
    e2 = tmp - np.dot(tmp, n_hat_obs) * n_hat_obs
    if np.linalg.norm(e2) < 1e-10:
        tmp = np.array([1.0, 0.0, 0.0])
        e2 = tmp - np.dot(tmp, n_hat_obs) * n_hat_obs
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n_hat_obs)
    e1 /= np.linalg.norm(e1)

    e_pol = np.cross(n_hat_obs, nhat_loc_vis)
    e_pol_mag = np.linalg.norm(e_pol, axis=1)
    good_pol = e_pol_mag > 1e-12 # i.e. the polarization direction is undefined geometrically, which correspond to eta = 1, for which polarization vanish
    e_pol[good_pol] /= e_pol_mag[good_pol, None]
    e_pol[~good_pol] = e1 # polarization vanishes, so we can choose any direction in the plane of the sky

    # Rotate local Q to common sky Q,U
    cos_phi = e_pol @ e1
    sin_phi = e_pol @ e2
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2.0 * cos_phi * sin_phi
    Q_sky =  Q_local * cos2phi
    U_sky = Q_local * sin2phi

    # Patch dA
    if dA is not None:
        if np.isscalar(dA):
            w = np.full(len(idx_vis), dA, dtype=float)
        else:
            dA = np.asarray(dA, dtype=float)
            w = dA[idx_vis]

    else:
        # Treat every patch as having equal area if dA = None and compute_spherical_dA = False
        w = np.ones(len(idx_vis))

    # Projected area factor (Code's eta)
    w_obs = (w * eta_vis)

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

            "eta": eta_vis,

            "F_mag": Fmag_vis,
            "F_hat": nhat_loc_vis,

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


def compute_simulation_nonconstant_flux(
        tau_profiles, temperature_profiles, alpha_abs_profiles, alpha_scatter_profiles,
        Fx, Fy, Fz, observers_xyz, *, bottom_intensities,
        radiation_energy_density_profiles=None, dA=None, n=3,
        subdivisions=4, max_delta_tau=.05):
    """Apply directly to your simulation arrays, one profile per photo patch.

    tau_profiles[k], temperature_profiles[k], etc. describe patch k; they may
    have different lengths between patches. Within each patch they must match.
    Fx,Fy,Fz supply the same approximate patch normals as in your original net
    calculation. observers_xyz is (N_observers,3), independent of patch count.
    bottom_intensities is a list of explicit scalar/callable boundary intensities
    (one per patch), or one scalar/callable to apply to all patches.
    No patch is omitted and no flux renormalization is performed.
    Returns (results, atmospheres). I,Q,U retain the original area-weighted
    Stokes sum convention; they are not fluxes at Earth without distance**2.
    """
    count = len(Fx)
    groups = (tau_profiles, temperature_profiles, alpha_abs_profiles, alpha_scatter_profiles)
    if count == 0 or len(Fy) != count or len(Fz) != count or any(len(g) != count for g in groups):
        raise ValueError('Supply one matching depth profile per photospheric patch.')
    if radiation_energy_density_profiles is not None and len(radiation_energy_density_profiles) != count:
        raise ValueError('Supply one radiation energy-density profile per patch.')
    if np.isscalar(bottom_intensities) or callable(bottom_intensities):
        boundaries = [bottom_intensities]*count
    else:
        boundaries = list(bottom_intensities)
        if len(boundaries) != count:
            raise ValueError('bottom_intensities must have one entry per patch.')
    observers = np.asarray(observers_xyz, dtype=float)
    if observers.ndim != 2 or observers.shape[1] != 3 or len(observers) == 0 or not np.all(np.isfinite(observers)) or np.any(np.linalg.norm(observers, axis=1) == 0):
        raise ValueError('observers_xyz must contain finite nonzero vectors with shape (N,3).')
    atmospheres = []
    for k in range(count):
        try:
            atmospheres.append(solve_simulation_patch(
                tau_profiles[k], temperature_profiles[k], alpha_abs_profiles[k],
                alpha_scatter_profiles[k], bottom_intensity=boundaries[k], n=n,
                radiation_energy_density=(None if radiation_energy_density_profiles is None
                                          else radiation_energy_density_profiles[k]),
                subdivisions=subdivisions, max_delta_tau=max_delta_tau))
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f'Photospheric patch {k}: {exc}') from exc
    stokes = np.array([compute_polarization_code(
        Fx, Fy, Fz, direction, atmospheres, dA=dA,
        atmospheres_are_per_unit_flux=False) for direction in observers])
    results = dict(P=stokes[:,0], I=stokes[:,1], Q=stokes[:,2], U=stokes[:,3],
                   F_surface=np.array([atm.flux(0) for atm in atmospheres]))
    return results, atmospheres


if __name__ == "__main__":
    import healpy as hp
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
    x_ph, y_ph, z_ph, Fx_ph, Fy_ph, Fz_ph = photo['x'], photo['y'], photo['z'], photo['Fx'], photo['Fy'], photo['Fz']

    r_vec_ph = np.column_stack((x_ph, y_ph, z_ph))
    r = np.linalg.norm(r_vec_ph,axis=1)
    r_hat = (r_vec_ph/ np.maximum(r[:, None], 1e-30))
    F_vec_ph = np.column_stack((Fx_ph, Fy_ph, Fz_ph))
    F_mag_ph = np.linalg.norm(F_vec_ph, axis=1)
    F_hat_ph = F_vec_ph / np.maximum(F_mag_ph[:, None], 1e-30)
    dOmega = 4.0 * np.pi / len(x_ph)
    cos_nr = np.abs(np.sum(F_hat_ph * r_hat, axis=1))
    dA = r**2* dOmega/ np.maximum(cos_nr, 1e-8)
    # dA = None

    all_radii = np.load(f'{abspath}/data/{folder}/flux_along_patch{snap}.npz', allow_pickle=True)    
    keys = list(all_radii.keys())

    tau_profiles = []
    T_profiles = []
    u_rad_profiles = []
    alpha_abs_profiles = []
    alpha_scatter_profiles = []

    for key in keys:  
        ray = all_radii[key].item()
        key = int(key)
        tau = np.asarray(ray['tau'])
        temp = np.asarray(ray['temp'])
        radden = np.asarray(ray['radden'])
        radden *= prel.en_den_converter
        alpha_abs = np.asarray(ray['alpha_abs'])
        alpha_scatter = np.asarray(ray['alpha_scatter'])

        # Solver requires tau increasing inward.
        tau[tau<0] = 1e20
        # if np.all(np.diff(tau) <= 0):
        tau = tau[::-1]
        temp = temp[::-1]
        radden = radden[::-1]
        alpha_abs = alpha_abs[::-1]
        alpha_scatter = alpha_scatter[::-1]

        tau_profiles.append(tau)
        T_profiles.append(temp)
        u_rad_profiles.append(radden)
        alpha_abs_profiles.append(alpha_abs)
        alpha_scatter_profiles.append(alpha_scatter)

    I_bottom = [ prel.c_cgs * profile[-1] / (4*np.pi)
    for profile in u_rad_profiles]

    results, atmospheres = compute_simulation_nonconstant_flux(
        tau_profiles=tau_profiles,
        temperature_profiles=T_profiles,
        alpha_abs_profiles=alpha_abs_profiles,
        alpha_scatter_profiles=alpha_scatter_profiles,
        radiation_energy_density_profiles=u_rad_profiles,
        bottom_intensities=I_bottom,
        Fx=Fx_ph,
        Fy=Fy_ph,
        Fz=Fz_ph,
        observers_xyz=observers_xyz,
        dA=dA,              # one surface area per patch; None means equal areas
        n=3,
        subdivisions=4,
        max_delta_tau=0.05,
    )

    #%%
    P_sim = results['P']
    I_sim = results['I']
    Q_sim = results['Q']
    U_sim = results['U']

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
    
    fig = plt.figure(figsize=(10, 8))
    axP = fig.add_subplot(111, projection='mollweide')
    img = axP.pcolormesh(lon_heal_mesh, lat_heal_mesh, data_grid_P, cmap='viridis', vmin = 0, vmax = np.max(P_sim))  #color by intensity
    cbar = plt.colorbar(img, orientation='horizontal', pad = 0.1, label =r'P')
    cbar.ax.tick_params(which='major',length = 6)


    print(np.median(data_grid_P))
# %%
