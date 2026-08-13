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
from scipy.sparse.linalg import splu, spsolve


class CNStepper:
    """2-D Crank-Nicolson step. The implicit nonlinear equation
        L W = base + dt M N((W+U)/2),   L = iM - dt/2 K,  N(u)=(s3|u|^2 + s5|u|^4)u
    is solved by either Picard (fixed-point, reuses the prefactorized L) or Newton
    (real-split Jacobian — |u|^2 u is not holomorphic, so the correction couples du and
    conj(du); quadratic convergence, robust where Picard stalls near strong
    concentration). Newton assumes the cubic power p=2. Both solve the SAME equation,
    so a converged step is method-independent."""

    def __init__(self, surface, dt, p=2, sigma5=0.0, tol=1e-10, pmax=60):
        self.M = surface["Mdiag"]
        self.dt = dt; self.p = p; self.sigma5 = sigma5
        self.tol = tol; self.pmax = pmax
        K = surface["K"]
        iM = sp.diags(1j * self.M)
        self._L = (iM - 0.5 * dt * K).tocsc()
        self._lu = splu(self._L)
        self._rhs = (iM + 0.5 * dt * K).tocsr()
        # real 2N x 2N block of L (for Newton): L = (-dt/2 K) + i diag(M)
        self._Lr = (-0.5 * dt * K).tocsr()
        self._Li = sp.diags(self.M).tocsr()
        self._J0 = sp.bmat([[self._Lr, -self._Li], [self._Li, self._Lr]]).tocsc()
        self._N = self.M.size

    def _nl(self, u, s3):
        a2 = np.abs(u) ** self.p
        return (s3 * a2 + self.sigma5 * a2 ** 2) * u

    def step(self, U, sigma3, method="picard", n_iter_out=None):
        return (self._newton(U, sigma3, n_iter_out) if method == "newton"
                else self._picard(U, sigma3, n_iter_out))

    def _picard(self, U, sigma3, n_iter_out):
        base = self._rhs @ U
        W = U.copy(); it = 0
        for it in range(1, self.pmax + 1):
            Wn = self._lu.solve(base + self.dt * (self.M * self._nl(0.5 * (W + U), sigma3)))
            if np.linalg.norm(Wn - W) / (np.linalg.norm(Wn) + 1e-300) < self.tol:
                W = Wn; break
            W = Wn
        if n_iter_out is not None:
            n_iter_out.append(it)
        return W

    def _newton(self, U, s3, n_iter_out):
        base = self._rhs @ U
        M, dt, s5, N = self.M, self.dt, self.sigma5, self._N
        W = U.copy(); it = 0
        for it in range(1, self.pmax + 1):
            Uh = 0.5 * (W + U); a2 = np.abs(Uh) ** 2
            F = self._L @ W - base - dt * (M * self._nl(Uh, s3))
            # dN/du (real) and dN/dubar (complex) at Uh, times the chain factor dUh/dW=1/2
            a = 2 * s3 * a2 + 3 * s5 * a2 ** 2
            B = Uh ** 2 * (s3 + 2 * s5 * a2)
            c = -0.5 * dt * M
            Da, Dbr, Dbi = c * a, c * np.real(B), c * np.imag(B)
            Jn = sp.bmat([[sp.diags(Da + Dbr), sp.diags(Dbi)],
                          [sp.diags(Dbi), sp.diags(Da - Dbr)]])
            J = (self._J0 + Jn).tocsc()
            d = spsolve(J, np.concatenate([-np.real(F), -np.imag(F)]))
            dW = d[:N] + 1j * d[N:]
            W = W + dW
            if np.linalg.norm(dW) / (np.linalg.norm(W) + 1e-300) < self.tol:
                break
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
