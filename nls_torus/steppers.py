"""Time steppers, with a single pluggable nonlinearity so every dynamical experiment
shares one well-tested engine (the scripts each grew their own copy — and their own
guard bugs).

  * CNStepper — 2-D Crank-Nicolson (implicit midpoint) on the surface. The linear
    operator (iM - dt/2 K) is coupling-independent, so it is prefactorized ONCE; the
    nonlinearity  N(u,t) = sigma3(t)|u|^2 u + sigma5|u|^4 u  is applied per step via a
    Picard fixed-point. sigma3 may be a constant or a callable sigma3(t) (breathing
    geometry / nonlinearity management); sigma5>0 adds a defocusing quintic.
  * RingStepper — 1-D belly ring split-step Fourier (Strang), coupling g(t) optional.

Collapse is flagged ONLY by peak-density blow-up / non-finite — never by Picard
non-convergence (a stiff-but-stable cubic-quintic step can hit max iters without
blowing up; conflating the two was a real bug in the ad-hoc scripts).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class CNStepper:
    def __init__(self, surface, dt, p=2, sigma5=0.0, tol=1e-10, pmax=60):
        self.M = surface["Mdiag"]
        self.dt = dt; self.p = p; self.sigma5 = sigma5
        self.tol = tol; self.pmax = pmax
        iM = sp.diags(1j * self.M)
        self._lu = splu((iM - 0.5 * dt * surface["K"]).tocsc())
        self._rhs = (iM + 0.5 * dt * surface["K"]).tocsr()

    def step(self, U, sigma3, n_iter_out=None):
        base = self._rhs @ U
        M, dt, p, s5 = self.M, self.dt, self.p, self.sigma5
        W = U.copy(); it = 0
        for it in range(1, self.pmax + 1):
            Uh = 0.5 * (W + U)
            a2 = np.abs(Uh) ** p
            Wn = self._lu.solve(base + dt * (M * ((sigma3 * a2 + s5 * a2 ** 2) * Uh)))
            if np.linalg.norm(Wn - W) / (np.linalg.norm(Wn) + 1e-300) < self.tol:
                W = Wn; break
            W = Wn
        if n_iter_out is not None:
            n_iter_out.append(it)
        return W


class RingStepper:
    """1-D focusing/defocusing NLS on the ring: i u_t + u_thth + g|u|^2 u = 0."""
    def __init__(self, ring, dt):
        self.q = ring["q"]; self.dt = dt
        self.Lh = np.exp(-1j * self.q ** 2 * dt / 2)

    def step(self, u, g):
        u = np.fft.ifft(self.Lh * np.fft.fft(u))
        u = u * np.exp(1j * g * np.abs(u) ** 2 * self.dt)
        return np.fft.ifft(self.Lh * np.fft.fft(u))
