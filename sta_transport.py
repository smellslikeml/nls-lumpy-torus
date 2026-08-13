"""H3 — shortcut-to-adiabaticity (STA) transport of a soliton by a moving geometric trap.

A bright soliton sits in an attractive well (the belly of a movable lump). Drag the well
from x=0 to x=D in time T and the soliton must come along WITHOUT being shaken: a naive
constant-velocity drag jerks it (start/stop kicks) so it arrives oscillating; a smooth
(poly5) trajectory helps; the inverse-engineered STA trajectory

    X_trap(t) = q(t) + q''(t)/omega^2,     q(t) = D * poly5(t/T)

(Torrontegui et al., transport STA for a moving harmonic trap; omega^2 = trap curvature)
delivers the soliton *at rest* even for fast, non-adiabatic T. Metric: residual
centre-of-mass oscillation + width breathing during a hold after arrival. Verification:
split-step conserves mass to machine precision.

Regenerate:  python3 sta_transport.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- grid & physics ------------------------------------------------------------------
N, Lbox = 2048, 80.0
x = np.linspace(-Lbox / 2, Lbox / 2, N, endpoint=False)
dx = Lbox / N
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
D, Ctrap = 15.0, 0.25                 # transport distance; harmonic trap V = C(x-X)^2
omega2 = 4 * Ctrap                     # Ehrenfest COM frequency for i u_t=-u_xx+V u  (=4C)
g = 2.0                               # focusing: i u_t = -u_xx - g|u|^2 u + V u
# A belly well V_k=k^2/A^2 is harmonic near its bottom; we use a clean harmonic trap to
# isolate the transport physics (the effective omega can be sourced from belly curvature).


def poly5(s):                          # 0->1 with zero 1st,2nd deriv at both ends
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def poly5_dd(s):                       # second derivative w.r.t. s
    return 60 * s - 180 * s ** 2 + 120 * s ** 3


def trap_center(t, T, kind):
    s = np.clip(t / T, 0.0, 1.0)
    if t >= T:
        return D
    if kind == "linear":
        return D * (t / T)
    q = D * poly5(s)
    if kind == "poly5":
        return q
    if kind == "sta":                  # q + q''/omega^2  (inverse engineering)
        qdd = D * poly5_dd(s) / T ** 2
        return q + qdd / omega2
    raise ValueError(kind)


def run(kind, T, T_hold=26.0, dt=2e-3):
    u = 1.0 / np.cosh(x)               # bright soliton (eta=1) at the trap centre x=0
    Lh = np.exp(-1j * k ** 2 * dt / 2)
    nsteps = int((T + T_hold) / dt)
    com, wid, ts, mass = [], [], [], []
    m0 = np.sum(np.abs(u) ** 2) * dx
    for i in range(nsteps):
        t = (i + 0.5) * dt
        Xc = trap_center(t, T, kind)
        V = Ctrap * (x - Xc) ** 2
        u = np.fft.ifft(Lh * np.fft.fft(u))
        u = u * np.exp(-1j * (V - g * np.abs(u) ** 2) * dt)
        u = np.fft.ifft(Lh * np.fft.fft(u))
        rho = np.abs(u) ** 2; m = rho.sum() * dx
        c = (x * rho).sum() * dx / m
        com.append(c); wid.append(np.sqrt(((x - c) ** 2 * rho).sum() * dx / m))
        ts.append((i + 1) * dt); mass.append(m)
    com, wid, ts, mass = map(np.array, (com, wid, ts, mass))
    hold = ts >= T
    return dict(t=ts, com=com, wid=wid,
                resid_osc=float(com[hold].max() - com[hold].min()),
                breath=float(wid[hold].max() - wid[hold].min()),
                arrive_err=float(abs(com[hold][0] - D)),
                mass_drift=float(np.max(np.abs(mass - m0)) / m0))


T = 4.0                                # fast: shorter than a trap period 2*pi/omega
w = float(2 * np.pi / np.sqrt(omega2))
print(f"trap period 2pi/omega = {w:.2f} ; transport time T = {T} (T/period = {T/w:.2f}, non-adiabatic)")
runs = {k_: run(k_, T) for k_ in ("linear", "poly5", "sta")}
print(f"{'protocol':8s} {'resid_osc':>10s} {'breathing':>10s} {'arrive_err':>11s} {'mass_drift':>11s}")
for kind, r in runs.items():
    print(f"{kind:8s} {r['resid_osc']:10.3f} {r['breath']:10.3f} {r['arrive_err']:11.3f} {r['mass_drift']:11.1e}")

# ---- figure ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.4))
cols = {"linear": "#c0803a", "poly5": "#8a4fb0", "sta": "#2b6cb0"}
labs = {"linear": "naive linear drag", "poly5": "smooth (poly5)", "sta": "STA (inverse-engineered)"}
for kind, r in runs.items():
    axL.plot(r["t"], r["com"], color=cols[kind], lw=1.9, label=labs[kind])
axL.axhline(D, color="#aaa", ls=":", lw=1.2); axL.axvline(T, color="#ccc", ls="--", lw=1)
axL.text(T, 1.5, "  transport done", color="#888", fontsize=9)
axL.set_xlabel("time"); axL.set_ylabel("soliton centre of mass")
axL.set_title("transport, then hold — who arrives at rest?", fontsize=11.5)
axL.legend(fontsize=9.5, loc="lower right")
for kind, r in runs.items():
    hold = r["t"] >= T
    axR.plot(r["t"][hold] - T, r["com"][hold] - D, color=cols[kind], lw=1.9, label=labs[kind])
axR.axhline(0, color="#aaa", ls=":", lw=1)
axR.set_xlabel("time after arrival"); axR.set_ylabel("residual COM oscillation")
axR.set_title(f"post-arrival excitation (fast T={T:g} < trap period {w:.1f})", fontsize=11.5)
axR.legend(fontsize=9.5, loc="upper right")
fig.tight_layout(); fig.savefig("sta_transport.png", dpi=130, bbox_inches="tight")
print("wrote sta_transport.png")
