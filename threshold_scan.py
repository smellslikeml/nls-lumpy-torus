"""
Mass-critical collapse threshold for the focusing equatorial ring.

Scan the ring amplitude (hence mass M = int A|u|^2). A small fixed azimuthal seed
makes the (necklace) instability reproducible. Record the collapse onset time;
below a critical mass the ring never collapses (within the window), above it the
collapse time falls off -- an empirical mass-critical threshold M_c.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nls_lumpy_torus import build_operators, beam_along_geodesic, make_stepper, mass

OUT = "/home/thorax/nls_lumpy_torus"
grid = build_operators(128, 96)
Mdiag = grid["Mdiag"]
th = grid["th"]; Nx, Nth = grid["Nx"], grid["Nth"]
seed = 1e-4 * (np.cos(5 * th) + np.cos(7 * th) + np.cos(9 * th))   # fixed azimuthal seed
dt = 1e-3; Tmax = 3.0; nsteps = int(Tmax / dt)

amps = [2.5, 3.0, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5]
masses, tcoll = [], []
for amp in amps:
    U = beam_along_geodesic(grid, xc=0.0, amp=amp, wx=0.30, k=6).reshape(Nx, Nth)
    U = (U * (1 + seed)).ravel()
    m0 = mass(U, Mdiag)
    step, stats = make_stepper(grid, dt=dt, sigma=-1.0, p=2)
    tc = None
    for nstep in range(1, nsteps + 1):
        U = step(U)
        pk = float(np.max(np.abs(U) ** 2))
        if not np.isfinite(pk) or pk > 80 or stats["picard_iters"][-1] >= 59:
            tc = nstep * dt; break
    masses.append(m0); tcoll.append(tc)
    print(f"  amp={amp:4.2f}  mass={m0:6.2f}  collapse={'%.2f'%tc if tc else 'none (stable to %.1f)'%Tmax}",
          flush=True)

masses = np.array(masses)
fig, ax = plt.subplots(figsize=(7.5, 5))
coll = np.array([t if t else np.nan for t in tcoll])
ax.plot(masses, coll, "o-", color="#b83280", ms=6)
stable = np.isnan(coll)
if stable.any():
    ax.plot(masses[stable], np.full(stable.sum(), Tmax), "s", color="#2b6cb0", ms=7,
            label=f"no collapse to t={Tmax}")
# threshold = midpoint between last-stable and first-collapsing mass
if stable.any() and (~stable).any():
    Mc = 0.5 * (masses[stable].max() + masses[~stable].min())
    ax.axvline(Mc, ls="--", c="0.5")
    ax.text(Mc, Tmax * 0.9, f"  $M_c\\approx${Mc:.1f}", color="0.3")
ax.set_xlabel(r"ring mass  $M=\int A|u|^2$"); ax.set_ylabel("collapse onset time")
ax.set_title("Mass-critical threshold of the focusing ring (necklace collapse)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/threshold.png", dpi=120)
print("wrote threshold.png")
