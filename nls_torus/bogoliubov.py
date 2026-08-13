"""Linear Bogoliubov mode integration for particle production on an expanding /
driven ring. Each comoving mode is a parametric oscillator

    v_k'' + omega_k(t)^2 v_k = 0 ,

started in its adiabatic (WKB) vacuum. Returns produced-pair number |beta_k|^2, the
observed density structure factor S_k(t_obs), and the Wronskian residual (the
built-in verification: |v v'* - v* v'| must stay 1)."""
import numpy as np


def integrate_modes(ks, omega_of_t, tgrid, c=1.0, a_i=1.0, a_f=1.0):
    """omega_of_t(t) -> array of omega_k at time t (shape like ks). Returns dict with
    |beta_k|^2, S_k(t_obs)=2 omega_f |v|^2, and the Wronskian residual."""
    ks = np.asarray(ks, float)
    omega_i = omega_of_t(tgrid[0]); omega_f = omega_of_t(tgrid[-1])
    v = (1.0 / np.sqrt(2.0 * omega_i)).astype(complex)
    w = (-1j * np.sqrt(omega_i / 2.0)).astype(complex)
    dt = tgrid[1] - tgrid[0]

    def deriv(vv, ww, t):
        return ww, -omega_of_t(t) ** 2 * vv

    for i in range(len(tgrid) - 1):
        t = tgrid[i]
        k1v, k1w = deriv(v, w, t)
        k2v, k2w = deriv(v + dt / 2 * k1v, w + dt / 2 * k1w, t + dt / 2)
        k3v, k3w = deriv(v + dt / 2 * k2v, w + dt / 2 * k2w, t + dt / 2)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w, t + dt)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)

    N_k = np.maximum(np.abs(w) ** 2 / (2 * omega_f) + omega_f / 2 * np.abs(v) ** 2 - 0.5, 0.0)
    S_k = 2.0 * omega_f * np.abs(v) ** 2
    Wdev = float(np.max(np.abs(np.abs(v * np.conj(w) - np.conj(v) * w) - 1.0)))
    return {"beta2": N_k, "S_k": S_k, "omega_i": omega_i, "omega_f": omega_f,
            "wronskian_max_dev": Wdev}
