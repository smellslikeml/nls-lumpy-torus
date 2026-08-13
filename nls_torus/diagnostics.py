"""Observables and — crucially — verification signals. Every experiment returns a
`verification` block so an agent (or a human) can tell a trustworthy result from a
plausible-looking artifact. These hooks are what caught the real bugs in this project
(the Kibble-Zurek noise-argmax, the Hawking sonic-point kink, delay-vs-arrest)."""
import numpy as np


def mass(U, Mdiag):
    return float(np.sum(Mdiag * np.abs(U) ** 2))


def energy(U, K, Mdiag, sigma3, p=2, sigma5=0.0):
    grad = 0.5 * np.real(np.vdot(U, K @ U))
    a2 = np.abs(U) ** p
    pot = np.sum(Mdiag * (sigma3 / (p + 2) * a2 * np.abs(U) ** 2
                          + sigma5 / (2 * p + 2) * a2 ** 2 * np.abs(U) ** 2))
    return float(grad + pot)


def peak(U):
    return float(np.max(np.abs(U) ** 2))


def ring_dominant_wavenumber(u, ring, band=None):
    """Dominant density wavenumber (defect-count / KZ). Restrict to `band` (a max |q|)
    so a high-q noise spike can't masquerade as the signal."""
    q = ring["q"]
    drho = np.abs(u) ** 2 - np.mean(np.abs(u) ** 2)
    P = np.abs(np.fft.fft(drho)) ** 2
    mask = q > 0 if band is None else (q > 0) & (q <= band)
    qp = q[mask]
    return float(qp[np.argmax(P[mask])])


def level_spacings(E, lo=0.15, hi=0.75, deg=14):
    """Unfolded nearest-neighbour spacings (quantum-chaos level statistics)."""
    E = np.sort(E); n = len(E)
    i0, i1 = int(lo * n), int(hi * n)
    Eb = E[i0:i1]
    c = np.polyfit(Eb, np.arange(i0, i1), deg)
    s = np.diff(np.polyval(c, Eb))
    return s[s > 0]


# ----------------------------------------------------------------- verification
def conservation_drift(hist_mass, hist_energy):
    """Relative mass and energy drift over a run — the licence to trust it."""
    m = np.asarray(hist_mass); e = np.asarray(hist_energy)
    dm = float(np.max(np.abs((m - m[0]) / (m[0] + 1e-300))))
    de = float(np.max(np.abs((e - e[0]) / (abs(e[0]) + 1e-300))))
    return {"mass_drift": dm, "energy_drift": de,
            "mass_conserved": dm < 1e-6, "energy_conserved": de < 1e-2}


def wronskian_residual(v, w, omega_i):
    """|v v'* - v* v'| should equal 1 for a correctly-normalised Bogoliubov mode."""
    W = np.abs(v * np.conj(w) - np.conj(v) * w)
    return {"wronskian_max_dev": float(np.max(np.abs(W - 1.0)))}
