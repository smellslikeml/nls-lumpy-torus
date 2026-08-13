"""
Side-by-side animation of the collapse-arrest run (collapse_arrest.py):
LEFT  a bare localized hump self-focuses and collapses (frozen once it blows up);
RIGHT the same hump under fast strong belly-breathing survives as a stable breather.
Chart view |u|^2 on the (x, theta) belly patch; shared color scale.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

OUT = "/home/thorax/nls_lumpy_torus"
d = np.load(f"{OUT}/collapse_arrest_snaps.npz")
Nx, Nth = int(d["Nx"]), int(d["Nth"])
x, th = d["x"], d["th"]
bt, bU = d["bare_t"], d["bare_U"]
mt, mU = d["man_t"], d["man_U"]
btc, mtc = float(d["bare_tc"]), float(d["man_tc"])
amp, delta, Om = float(d["amp"]), float(d["delta"]), float(d["Om"])

# zoom to the belly neighbourhood of the hump (theta ~ pi, x ~ 0) for a clearer view
def field(U):
    return np.abs(U.reshape(Nx, Nth)) ** 2

VMAX = 20.0                          # low enough that the settled breather stays visible;
T_SHOW = 1.2                         # the initial hump (~34) and the collapse saturate to max
idx = [i for i, t in enumerate(mt) if t <= T_SHOW]
frames = []
for i in idx:
    t = mt[i]
    mf = field(mU[i])
    bj = min(max(int(np.searchsorted(bt, t, side="right")) - 1, 0), len(bt) - 1)
    bf = field(bU[bj])
    collapsed = (btc > 0 and t >= btc)

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.5))
    for a, f, ttl, col in ((ax[0], bf, "bare ring", "#b83280"),
                           (ax[1], mf, f"managed  δ={delta:.1f}, Ω={Om:.0f}", "#2b6cb0")):
        im = a.pcolormesh(th, x, f, shading="auto", cmap="inferno", vmin=0, vmax=VMAX)
        a.set_xlabel(r"$\theta$"); a.set_ylabel(r"$x$")
        a.set_xticks([0, np.pi, 2 * np.pi]); a.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
        a.set_yticks([-np.pi / 2, 0, np.pi / 2]); a.set_yticklabels(["neck", "belly", "neck"])
        a.set_title(ttl, color=col, fontsize=11)
    if collapsed:
        ax[0].text(np.pi, 0.0, f"COLLAPSED\nt_c={btc:.2f}", color="white", ha="center", va="center",
                   fontsize=13, fontweight="bold")
    peak_now = bf.max() if not collapsed else float("inf")
    fig.suptitle(rf"Localized hump on the belly: collapse vs dynamically-arrested breather   ·   t={t:.2f}",
                 fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$|u|^2$", pad=0.02)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

imageio.mimsave(f"{OUT}/nls_collapse_arrest.gif", frames, fps=14, loop=0)
print(f"wrote nls_collapse_arrest.gif ({len(frames)} frames)  bare t_c={btc:.3f}  "
      f"managed {'arrested' if mtc < 0 else 't_c=%.3f' % mtc}")
