"""
Direction 1 (honest version): does geometry move the collapse threshold?

For several lump amplitudes eps, scan the ring mass and record the collapse
onset time. Overlaying collapse-time-vs-mass across geometries answers it
directly: if the curves coincide the necklace-collapse threshold is a LOCAL
belly-ring property (universal, since the belly is A=1 for every eps); if they
separate, geometry tunes it.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nls_lumpy_torus import build_operators, beam_along_geodesic, make_stepper, mass

OUT = "/home/thorax/nls_lumpy_torus"
Nx, Nth = 112, 80
dt, Tmax = 1e-3, 2.5
nsteps = int(Tmax / dt)
amps = [3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]
epss = [0.5, 1.0, 2.0, 4.0]


def run(grid, amp):
    th = grid["th"]
    seed = 1e-4 * (np.cos(5 * th) + np.cos(7 * th) + np.cos(9 * th))
    U = beam_along_geodesic(grid, xc=0.0, amp=amp, wx=0.30, k=6).reshape(Nx, Nth)
    U = (U * (1 + seed)).ravel()
    m0 = mass(U, grid["Mdiag"])
    step, stats = make_stepper(grid, dt=dt, sigma=-1.0, p=2)
    for n in range(1, nsteps + 1):
        U = step(U)
        pk = float(np.max(np.abs(U) ** 2))
        if not np.isfinite(pk) or pk > 80 or stats["picard_iters"][-1] >= 59:
            return m0, n * dt
    return m0, None


fig, ax = plt.subplots(figsize=(8, 5.2))
cols = ["#2b6cb0", "#1c2230", "#d69e2e", "#b83280"]
for eps, col in zip(epss, cols):
    g = build_operators(Nx, Nth, eps=eps)
    ms, ts = [], []
    for a in amps:
        m, t = run(g, a)
        ms.append(m); ts.append(t if t else np.nan)
    ms = np.array(ms); ts = np.array(ts)
    ax.plot(ms, ts, "o-", color=col, ms=5, label=f"ε={eps}")
    print(f"  eps={eps}: masses {np.round(ms,1)}  collapse_t {np.round(ts,2)}", flush=True)
ax.set_xlabel(r"ring mass  $M=\int A|u|^2$"); ax.set_ylabel("collapse onset time")
ax.set_title("Collapse-time vs mass across geometries — do the curves coincide?")
ax.grid(alpha=0.3); ax.legend(title="lump amplitude")
fig.tight_layout(); fig.savefig(f"{OUT}/mc_geometry.png", dpi=120)
print("wrote mc_geometry.png", flush=True)
