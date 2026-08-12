"""
NLS on the lumpy torus  —  Python companion to
https://smellslike.ml/posts/nls-on-the-lumpy-torus/

Solves the focusing/defocusing nonlinear Schrodinger equation on a surface of
revolution ("lumpy torus") with intrinsic metric

        ds^2 = dx^2 + A(x)^2 dtheta^2 ,      A(x) = sqrt((1 + cos^2 x)/2)

    i u_t = -Delta_g u + sigma |u|^p u ,   (x,theta) doubly periodic

Improvements over the reference nls.edp:
  * mass-conserving Laplace-Beltrami assembly (Hermitian stiffness, lumped mass)
  * Crank-Nicolson (implicit midpoint), 2nd order in time
  * Picard/fixed-point nonlinear solve WITH a residual convergence check
    (the .edp did a fixed 3 iterations with no check) + prefactorized linear op
  * mass AND energy conservation diagnostics (the .edp tracked only max|u|)
  * resolvable, clearly-labelled parameters
  * exact double periodicity (no dead adaptmesh, no off-by-one)

Laplace-Beltrami for this metric:  Delta_g u = u_xx + (A'/A) u_x + (1/A^2) u_thth
Weak (mass-conserving) form uses  sqrt|g|=A, g^xx=1, g^thth=1/A^2:
    stiffness  a(u,v) = int [ A u_x v_x + (1/A) u_th v_th ] dx dtheta
    mass       m(u,v) = int  A u vbar          dx dtheta   (lumped -> diagonal)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


# ----------------------------------------------------------------------------- metric
def profile_A(x):
    return np.sqrt((1.0 + np.cos(x) ** 2) / 2.0)


# ----------------------------------------------------------------------- FE operators
def build_operators(Nx, Nth, Lx=np.pi, x0=-np.pi / 2.0):
    """Return grid + (K stiffness sparse, Mdiag lumped-mass diagonal vector).

    Node ordering is flattened (i in x, j in theta) -> i*Nth + j.
    """
    dx = Lx / Nx
    dth = 2.0 * np.pi / Nth
    x = x0 + dx * np.arange(Nx)          # periodic over [x0, x0+Lx)
    th = dth * np.arange(Nth)            # periodic over [0, 2pi)
    A = profile_A(x)                     # (Nx,)

    # --- 1D x-stiffness  int A u_x v_x dx  (periodic, face-averaged A) ---
    Aface = 0.5 * (A + np.roll(A, -1))   # A_{i+1/2}
    Am = np.roll(Aface, 1)               # A_{i-1/2}
    main = (Am + Aface) / dx
    off = -Aface / dx                    # edge (i, i+1)
    rows = np.concatenate([np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx])
    cols = np.concatenate([np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)])
    vals = np.concatenate([main, off, off])
    Sx = sp.coo_matrix((vals, (rows, cols)), shape=(Nx, Nx)).tocsr()

    # --- 1D theta-stiffness  int u_th v_th dth  (periodic) ---
    mt = np.full(Nth, 2.0 / dth)
    ot = np.full(Nth, -1.0 / dth)
    rows = np.concatenate([np.arange(Nth), np.arange(Nth), (np.arange(Nth) + 1) % Nth])
    cols = np.concatenate([np.arange(Nth), (np.arange(Nth) + 1) % Nth, np.arange(Nth)])
    vals = np.concatenate([mt, ot, ot])
    Sth = sp.coo_matrix((vals, (rows, cols)), shape=(Nth, Nth)).tocsr()

    Ith = sp.identity(Nth, format="csr")
    # K = dth*(Sx (x) Ith)  +  dx*(diag(1/A) (x) Sth)
    K = dth * sp.kron(Sx, Ith) + dx * sp.kron(sp.diags(1.0 / A), Sth)
    K = K.tocsr()

    # lumped mass diagonal  A_i * dx * dth  (broadcast over theta)
    Mdiag = np.repeat(A, Nth) * dx * dth
    return dict(x=x, th=th, A=A, K=K, Mdiag=Mdiag, dx=dx, dth=dth, Nx=Nx, Nth=Nth)


# --------------------------------------------------------------------- diagnostics
def mass(U, Mdiag):
    return float(np.sum(Mdiag * np.abs(U) ** 2))


def energy(U, K, Mdiag, sigma, p):
    grad = 0.5 * np.real(np.vdot(U, K @ U))                    # 1/2 int |grad u|^2_g
    pot = (sigma / (p + 2.0)) * np.sum(Mdiag * np.abs(U) ** (p + 2))
    return float(grad + pot)


# ------------------------------------------------------------------------ IC
def initial_condition(grid, amp=1.0, xc=0.0, thc=np.pi, wx=0.4, wth=0.5, k=4):
    """Localized 2-D wavepacket with integer theta-momentum k (periodic-safe)."""
    x, th = grid["x"], grid["th"]
    X, TH = np.meshgrid(x, th, indexing="ij")
    dth_c = np.angle(np.exp(1j * (TH - thc)))                  # signed periodic distance
    env = np.exp(-((X - xc) ** 2) / (2 * wx ** 2) - dth_c ** 2 / (2 * wth ** 2))
    U0 = amp * env * np.exp(1j * k * TH)
    return U0.ravel()


def beam_along_geodesic(grid, xc=0.0, amp=1.0, wx=0.25, k=6, Lx=np.pi):
    """Gaussian beam concentrated transverse (x) to the parallel geodesic x=xc,
    uniform along it (theta), carrying integer angular momentum k.

    xc=0     -> elliptic (stable) equator, A maximum, curvature K>0
    xc=pi/2  -> hyperbolic (unstable) neck, A minimum, curvature K<0
    Because the flow is axisymmetric, u stays in the e^{ik theta} sector: the
    ring only breathes/spreads transversely, driven by the centrifugal
    potential k^2/A^2 (a well at x=0, a barrier at the neck).
    """
    x, th = grid["x"], grid["th"]
    X, TH = np.meshgrid(x, th, indexing="ij")
    dxc = (X - xc + Lx / 2.0) % Lx - Lx / 2.0                  # periodic distance in x
    U0 = amp * np.exp(-dxc ** 2 / (2 * wx ** 2)) * np.exp(1j * k * TH)
    return U0.ravel()


def beam_along_meridian(grid, thc=0.0, amp=1.0, wth=0.4, q=4):
    """Gaussian beam concentrated transverse (theta) to the meridian theta=thc,
    extended along the meridian (x), carrying poloidal momentum q.

    Every meridian is a geodesic, but it crosses the K>0 belly (x=0) and the
    K<0 necks (x=+-pi/2), so the transverse (theta) width should FOCUS over the
    belly and DEFOCUS over the necks -- a Hill/Floquet geodesic, not a stable
    one. q must be an even integer (x has period pi) so e^{iqx} stays periodic.
    """
    x, th = grid["x"], grid["th"]
    X, TH = np.meshgrid(x, th, indexing="ij")
    dthc = np.angle(np.exp(1j * (TH - thc)))                   # signed periodic theta-distance
    U0 = amp * np.exp(-dthc ** 2 / (2 * wth ** 2)) * np.exp(1j * q * X)
    return U0.ravel()


# ------------------------------------------------------------------- time stepping
def make_stepper(grid, dt, sigma, p, picard_tol=1e-11, picard_max=60):
    K, Mdiag = grid["K"], grid["Mdiag"]
    N = Mdiag.size
    iM = sp.diags(1j * Mdiag)
    L = (iM - 0.5 * dt * K).tocsc()          # linear CN operator (constant in time)
    lu = splu(L)                             # prefactorize once
    rhs_const_op = (iM + 0.5 * dt * K)       # applied to U^n each step

    def nl(V):                               # lumped nonlinear vector sigma*tau*N(V)
        return sigma * dt * (Mdiag * (np.abs(V) ** p * V))

    stats = {"picard_iters": []}

    def step(Un):
        base = rhs_const_op @ Un
        W = Un.copy()
        it = 0
        for it in range(1, picard_max + 1):
            Uhat = 0.5 * (W + Un)
            Wn = lu.solve(base + nl(Uhat))
            denom = np.linalg.norm(Wn) + 1e-300
            if np.linalg.norm(Wn - W) / denom < picard_tol:
                W = Wn
                break
            W = Wn
        stats["picard_iters"].append(it)
        return W

    return step, stats


def run(grid, U0, dt, T, sigma, p, n_snapshots=120, verbose=True):
    step, stats = make_stepper(grid, dt, sigma, p)
    K, Mdiag = grid["K"], grid["Mdiag"]
    nsteps = int(round(T / dt))
    snap_every = max(1, nsteps // n_snapshots)

    U = U0.copy()
    hist = {"t": [], "mass": [], "energy": [], "maxabs2": []}
    snaps = {"t": [], "U": []}

    def record(t):
        hist["t"].append(t)
        hist["mass"].append(mass(U, Mdiag))
        hist["energy"].append(energy(U, K, Mdiag, sigma, p))
        hist["maxabs2"].append(float(np.max(np.abs(U) ** 2)))

    record(0.0)
    snaps["t"].append(0.0); snaps["U"].append(U.copy())
    for n in range(1, nsteps + 1):
        U = step(U)
        t = n * dt
        record(t)
        if n % snap_every == 0 or n == nsteps:
            snaps["t"].append(t); snaps["U"].append(U.copy())
        if verbose and (n % max(1, nsteps // 10) == 0 or n == nsteps):
            m = hist["mass"][-1]; e = hist["energy"][-1]
            dm = (m - hist["mass"][0]) / hist["mass"][0]
            de = (e - hist["energy"][0]) / (abs(hist["energy"][0]) + 1e-300)
            print(f"  step {n:5d}/{nsteps}  t={t:6.3f}  "
                  f"dMass/M0={dm:+.2e}  dEnergy/E0={de:+.2e}  "
                  f"picard~{np.mean(stats['picard_iters'][-snap_every:]):.1f}")
    for k in hist:
        hist[k] = np.array(hist[k])
    return U, hist, snaps, stats


# ------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    Nx, Nth = (48, 96) if quick else (64, 128)
    grid = build_operators(Nx, Nth)
    # symmetry / PSD sanity on K and M
    K = grid["K"]
    asym = abs((K - K.T)).max()
    print(f"grid {Nx}x{Nth}  N={grid['Mdiag'].size}  "
          f"K symmetric (max|K-K^T|={asym:.1e})  Mdiag>0: {np.all(grid['Mdiag']>0)}")
    p = 2
    for sigma, tag in [(+1.0, "defocusing"), (-1.0, "focusing")]:
        print(f"\n[{tag}]  sigma={sigma:+.0f}, p={p}")
        U0 = initial_condition(grid, amp=1.0, k=4)
        T = 0.2 if quick else 1.0
        U, hist, snaps, stats = run(grid, U0, dt=2e-3, T=T, sigma=sigma, p=p,
                                    n_snapshots=40, verbose=True)
        dm = (hist["mass"] - hist["mass"][0]) / hist["mass"][0]
        de = (hist["energy"] - hist["energy"][0]) / (abs(hist["energy"][0]) + 1e-300)
        print(f"  => max |dMass/M0| = {np.max(np.abs(dm)):.2e}   "
              f"max |dEnergy/E0| = {np.max(np.abs(de)):.2e}   "
              f"max picard iters = {max(stats['picard_iters'])}")
