"""Discrete operators for the NLS on the lumpy torus.

  * build_surface(Nx, Nth, eps): the 2-D Laplace-Beltrami operator as a Hermitian
    stiffness K and a positive lumped-mass diagonal M (mass-conserving finite-element
    assembly on the metric ds^2 = dx^2 + A(x)^2 dtheta^2).
  * ring_grid(Nth): the 1-D belly ring (spectral / split-step Fourier) — wavenumbers.
  * wg_chain(...): the whispering-gallery tight-binding reduction (for the topological
    / Floquet / Thouless lattice experiments).

Extracted and unified from nls_lumpy_torus.build_operators.
"""
import numpy as np
import scipy.sparse as sp
from .geometry import profile_A


def build_surface(Nx=64, Nth=128, Lx=np.pi, x0=-np.pi / 2.0, eps=1.0):
    """Grid + (K stiffness sparse, Mdiag lumped-mass vector). Node order i*Nth+j."""
    dx = Lx / Nx
    dth = 2.0 * np.pi / Nth
    x = x0 + dx * np.arange(Nx)
    th = dth * np.arange(Nth)
    A = profile_A(x, eps)

    # 1-D x-stiffness  int A u_x v_x dx  (periodic, face-averaged A)
    Aface = 0.5 * (A + np.roll(A, -1))
    Am = np.roll(Aface, 1)
    rows = np.concatenate([np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx])
    cols = np.concatenate([np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)])
    vals = np.concatenate([(Am + Aface) / dx, -Aface / dx, -Aface / dx])
    Sx = sp.coo_matrix((vals, (rows, cols)), shape=(Nx, Nx)).tocsr()

    # 1-D theta-stiffness  int u_th v_th dth  (periodic)
    mt = np.full(Nth, 2.0 / dth); ot = np.full(Nth, -1.0 / dth)
    rows = np.concatenate([np.arange(Nth), np.arange(Nth), (np.arange(Nth) + 1) % Nth])
    cols = np.concatenate([np.arange(Nth), (np.arange(Nth) + 1) % Nth, np.arange(Nth)])
    vals = np.concatenate([mt, ot, ot])
    Sth = sp.coo_matrix((vals, (rows, cols)), shape=(Nth, Nth)).tocsr()

    Ith = sp.identity(Nth, format="csr")
    K = (dth * sp.kron(Sx, Ith) + dx * sp.kron(sp.diags(1.0 / A), Sth)).tocsr()
    Mdiag = np.repeat(A, Nth) * dx * dth
    return dict(x=x, th=th, A=A, K=K, Mdiag=Mdiag, dx=dx, dth=dth, Nx=Nx, Nth=Nth, eps=eps)


def ring_grid(Nth=512, radius=1.0):
    """1-D belly ring of circumference 2*pi*radius: angle grid + physical wavenumbers."""
    th = np.linspace(0, 2 * np.pi * radius, Nth, endpoint=False)
    q = 2 * np.pi * np.fft.fftfreq(Nth, d=(2 * np.pi * radius) / Nth)
    return dict(th=th, q=q, Nth=Nth, radius=radius)


def wg_chain(kind="harper", Ncell=20, alpha=1.0 / 3.0, t_hop=1.0, lam=1.0):
    """Whispering-gallery tight-binding reduction. kind='harper' -> Hofstadter strip
    H(k_theta) (chiral-twist topological bands); returns a callable H(k)."""
    N = Ncell
    n = np.arange(N)

    def H(k):
        Hm = np.diag(2 * lam * np.cos(2 * np.pi * alpha * n + k)).astype(complex)
        Hm += np.diag(-t_hop * np.ones(N - 1), 1) + np.diag(-t_hop * np.ones(N - 1), -1)
        return Hm
    return H, N
