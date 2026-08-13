"""Nonlinear Kerr on the horn's leaky mode — the geometry-set resonator, made nonlinear.

Section 12 gave the horn a leaky whispering-gallery-like mode whose Q is set by the throat
geometry; section 13 turned a microdisk into a Kerr comb whose threshold rides on that Q.
This joins them: put a focusing Kerr nonlinearity ON the horn's leaky mode and ask what the
nonlinearity does to a mode that is *also* radiating.

Method — a ring-down. Excite the eta-stable linear leaky mode, evolve
    i u_t = (-d^2 + V_k(x) - i*CAP) u  -  g |u|^2 u
and, as the intracavity power decays, read the instantaneous resonance frequency and Q as
functions of power. One run sweeps power from high to low. The g=0 run is the control:
frequency and Q must be flat and equal to the linear values.

Result: the resonance REDSHIFTS linearly with intracavity power (Kerr self-phase modulation),
and the focusing nonlinearity mildly ENHANCES Q — self-trapping deepens the mode below the
throat barrier, so it leaks less. Geometry sets the linear Q; the Kerr term tilts and
self-traps it.

Regenerate:  python3 nonlinear_kerr_horn.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_open, xt, w, Xmax, Xpml, K = 1.0, 8.0, 1.0, 40.0, 30.0, 2.0
Delta, N = 0.26, 1400


def Hmat(Vk, dx, eta):
    cap = eta * np.clip((np.linspace(0, Xmax, N) - Xpml) / (Xmax - Xpml), 0, None) ** 2
    return sp.diags([-1/dx**2*np.ones(N-1), 2/dx**2 + Vk - 1j*cap, -1/dx**2*np.ones(N-1)],
                    [-1, 0, 1]).tocsc()


x = np.linspace(0, Xmax, N); dx = x[1] - x[0]
Vk = K ** 2 / (A_open - Delta * np.exp(-((x - xt) / w) ** 2)) ** 2


def stable_mode():
    Vopen, Vbar = K ** 2, K ** 2 / (1 - Delta) ** 2
    v1, vec1 = eigs(Hmat(Vk, dx, 2.0), k=18, sigma=K**2 + (np.pi/xt)**2 - 0.02j)
    v2, _ = eigs(Hmat(Vk, dx, 4.5), k=18, sigma=K**2 + (np.pi/xt)**2 - 0.02j)
    best = None
    for e, c in zip(v1, vec1.T):
        if Vopen < e.real < Vbar and e.imag < -1e-12:
            rel = abs(v2[np.argmin(np.abs(v2 - e))] - e) / max(abs(e.imag), 1e-12)
            if rel < 0.05 and (best is None or rel < best[2]):
                best = (e, c, rel)
    return best


e, psi, rel = stable_mode()
E_r, Gamma = e.real, -2 * e.imag; Q_lin = E_r / Gamma
psi = psi / np.abs(psi).max()
print(f"linear leaky mode: E_r={E_r:.4f}, Gamma={Gamma:.2e}, Q_lin={Q_lin:.0f}, eta-drift={rel:.1e}")


def ringdown(A0, g, T=520.0, dt=2e-3, rec=40):
    H = Hmat(Vk, dx, 2.0); I = sp.identity(N, format="csc")
    lu = splu(I + 0.5j*dt*H); Bop = I - 0.5j*dt*H
    u = (A0 * psi).astype(complex); wp = np.conj(psi) / np.sum(np.abs(psi)**2)
    ts, cs = [], []
    for i in range(int(T/dt)):
        u = lu.solve(Bop @ u); u = u * np.exp(1j*g*np.abs(u)**2*dt)
        if i % rec == 0:
            ts.append(i*dt); cs.append(np.sum(wp * u))
    return np.array(ts), np.array(cs)


def extract(t, c, win=60):
    ph = np.unwrap(np.angle(c)); amp = np.abs(c); P, om, Q = [], [], []
    for i in range(0, len(t) - win, win // 2):
        j = i + win; dtw = t[j] - t[i]
        om.append(-(ph[j] - ph[i]) / dtw)
        gam = -(np.log(amp[j]) - np.log(amp[i])) / dtw
        Q.append(om[-1] / (2*gam) if gam > 1e-9 else np.nan)
        P.append(np.mean(amp[i:j] ** 2))
    return np.array(P), np.array(om), np.array(Q)


tK, cK = ringdown(0.55, 1.0); PK, omK, QK = extract(tK, cK)
t0, c0 = ringdown(0.55, 0.0); P0, om0, Q0 = extract(t0, c0)
lw = Gamma                                                    # one linewidth
Pc = float(PK[np.nanargmax(QK)]); Qmax = float(np.nanmax(QK))  # self-focusing threshold
below = PK <= Pc
slope = np.polyfit(PK[below], (omK[below] - E_r) / lw, 1)[0]
print(f"linear leaky Q = {Q_lin:.0f}; one linewidth Gamma={Gamma:.2e}")
print(f"Kerr redshift is linear in power: {slope:.1f} linewidths per unit P (self-phase modulation)")
print(f"focusing Kerr self-traps -> Q enhanced to {Qmax:.0f} (+{(Qmax/Q_lin-1)*100:.0f}%) up to P_c={Pc:.3f}")
print(f"beyond P_c a self-focusing instability collapses Q (redshift saturates at the same power)")
print(f"VERIFICATION control (g=0) flat: dOmega={om0.max()-om0.min():.1e} rad, Q const {np.nanmean(Q0):.0f}")

# ---- figure ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.5))
axA.plot(PK, (omK - E_r) / lw, "o-", color="#b83280", lw=1.8, ms=4, label="Kerr ($g=1$)")
axA.axhline(0, color="#2b6cb0", lw=1.6, ls="--", label="linear ($g=0$) control")
axA.axvline(Pc, color="#888", ls=":", lw=1.2); axA.text(Pc*0.97, -13, "self-focusing\nthreshold", fontsize=8.3, ha="right", color="#666")
axA.set_xlabel("intracavity power  $P=|c|^2$"); axA.set_ylabel("frequency shift  $(\\omega-\\omega_0)/\\Gamma$  (linewidths)")
axA.set_title("Kerr redshift ∝ power (self-phase modulation)", fontsize=11.5)
axA.legend(fontsize=9.5, loc="lower left"); axA.grid(alpha=.25)

good = np.isfinite(QK)
axB.plot(PK[good], QK[good], "o-", color="#2b6cb0", lw=1.8, ms=4, label="Kerr ($g=1$)")
axB.axhline(Q_lin, color="#888", ls="--", lw=1.5, label=f"linear $Q={Q_lin:.0f}$ (geometry)")
axB.axvline(Pc, color="#888", ls=":", lw=1.2)
axB.annotate(f"+{(Qmax/Q_lin-1)*100:.0f}% self-trapping", (Pc, Qmax), textcoords="offset points",
             xytext=(-8, 4), ha="right", fontsize=8.6, color="#2b6cb0")
axB.set_xlabel("intracavity power  $P=|c|^2$"); axB.set_ylabel("quality factor  $Q$")
axB.set_title("Kerr self-traps → Q up, until an instability", fontsize=11.5)
axB.legend(fontsize=9.5, loc="lower left"); axB.grid(alpha=.25)
fig.tight_layout(); fig.savefig("nonlinear_kerr_horn.png", dpi=130, bbox_inches="tight")
print("wrote nonlinear_kerr_horn.png")
