"""
Self-trapping the quasimode on the stable elliptic equator. Same ring beam
(narrow transverse x, extended along theta, momentum k) evolved linearly
(whispering-gallery mode: breathes in the centrifugal well) vs with focusing
nonlinearity (self-traps into a tighter, higher-peak, steadier quasimode).
theta-symmetric -> 1-D-in-x -> subcritical -> stable (no collapse at amp=3).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, beam_along_geodesic, run
from render import animate_torus

OUT = "/home/thorax/nls_lumpy_torus"
Lx = np.pi
grid = build_operators(Nx=160, Nth=128)
x, A = grid["x"], grid["A"]
Nx, Nth = grid["Nx"], grid["Nth"]


def xwidth_peak(U):
    f = np.abs(U.reshape(Nx, Nth)) ** 2
    Px = f.sum(axis=1) * A * grid["dx"] * grid["dth"]
    d = (x - 0.0 + Lx / 2) % Lx - Lx / 2
    return np.sqrt(np.sum(Px * d ** 2) / (Px.sum() + 1e-300)), float(f.max())


res = {}
for name, sigma, gif in [("linear (whispering-gallery)", 0.0, "nls_selftrap_linear_torus.gif"),
                         ("focusing (self-trapped)", -1.0, "nls_selftrap_focusing_torus.gif")]:
    print(f"\n=== {name} ===", flush=True)
    U0 = beam_along_geodesic(grid, xc=0.0, amp=3.0, wx=0.30, k=6)
    U, hist, snaps, stats = run(grid, U0, dt=1e-3, T=2.5, sigma=sigma, p=2,
                                n_snapshots=80, verbose=True)
    wp = np.array([xwidth_peak(U) for U in snaps["U"]])
    res[name] = dict(t=np.array(snaps["t"]), w=wp[:, 0], pk=wp[:, 1])
    print(f"  mass drift {(hist['mass'][-1]-hist['mass'][0])/hist['mass'][0]:+.1e}, "
          f"max picard {max(stats['picard_iters'])}", flush=True)
    animate_torus(grid, snaps, f"{OUT}/{gif}", fps=18)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, c in zip(res, ["C7", "C1"]):
    ax[0].plot(res[name]["t"], res[name]["w"], color=c, label=name)
    ax[1].plot(res[name]["t"], res[name]["pk"], color=c, label=name)
ax[0].set_title(r"transverse ($x$) width of the ring"); ax[0].set_ylabel("width")
ax[1].set_title(r"peak $|u|^2$")
for a in ax:
    a.set_xlabel("t"); a.grid(alpha=0.3); a.legend()
fig.suptitle("Self-trapping the equatorial quasimode: focusing holds it tighter & higher than the linear mode")
fig.tight_layout(); fig.savefig(f"{OUT}/selftrap_comparison.png", dpi=120)
plt.close(fig)
print("wrote selftrap_comparison.png", flush=True)
print("ALL DONE", flush=True)
