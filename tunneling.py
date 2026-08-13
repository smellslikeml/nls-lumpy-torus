"""H4 — geometry-controlled tunneling. The two-lobe `double_lump` surface is a double
well on the ring: the centrifugal potential V_k = k^2/A^2 has a well in each belly lobe,
separated by the neck barriers. The two lowest states form a near-degenerate doublet
(symmetric / antisymmetric across the lobes) whose splitting IS the inter-lobe tunneling
rate — and it collapses exponentially as the necks deepen. Pure geometric control of a
tunneling two-level system; the single-well torus has no such doublet.

Found by the nls_torus agent-toolkit (fast 1-D eigensolver on the §02 reduction), not
hand-designed. Regenerate: python3 tunneling.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nls_torus.geometry import make_geometry


def ladder(name, kw, k=8, Nx=800):
    geo = make_geometry(name, **kw)
    x = np.linspace(geo.x0, geo.x0 + geo.Lx, Nx, endpoint=False)
    dx = geo.Lx / Nx
    A = np.asarray(geo.A_of(x), float)
    V = k ** 2 / A ** 2
    H = (np.diag(2 / dx ** 2 + V)
         + np.diag(-1 / dx ** 2 * np.ones(Nx - 1), 1)
         + np.diag(-1 / dx ** 2 * np.ones(Nx - 1), -1))
    H[0, -1] = H[-1, 0] = -1 / dx ** 2          # periodic ring
    E, psi = np.linalg.eigh(H)
    return x, V, E, psi, float(geo.curvature(0.0))


epss = [0.5, 1.0, 2.0, 4.0, 8.0]
splits = [ladder("double_lump", {"eps": e})[2] for e in epss]
splits = [E[1] - E[0] for E in splits]
ctrl = (lambda r: r[2][1] - r[2][0])(ladder("lumpy_torus", {"eps": 1.0}))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.5))

# ---- left: the double well + the tunneling doublet -------------------------------------
x, V, E, psi, K = ladder("double_lump", {"eps": 6.0})
roll = len(x) // 4                                   # centre the view on a neck barrier
x2 = np.linspace(-np.pi / 2, np.pi / 2, len(x))
V, psi = np.roll(V, roll), np.roll(psi, roll, axis=0)
axL.plot(x2, V, color="#1c2230", lw=2.0, zorder=3)
axL.fill_between(x2, V, V.min() - 3, color="#eef1f5", zorder=0)
axL.set_ylim(V.min() - 1, V.min() + 26)
axL.text(0.5, 0.96, "two belly wells  +  neck barriers", transform=axL.transAxes,
         va="top", ha="center", fontsize=10.5, color="#5c6672")
axL.set_xlabel("meridian $x$  (one period, centred on a neck)")
axL.set_ylabel("$V_k = k^2/A^2$")
axL.set_title("double_lump: a double well on the ring", fontsize=11.5)
ax2 = axL.twinx()                                    # signed doublet on its own scale
for n, col, lab in [(0, "#2b6cb0", "symmetric $\\psi_0$"),
                    (1, "#b83280", "antisymmetric $\\psi_1$")]:
    p = psi[:, n]; p = p / np.abs(p).max()
    if p[np.argmax(np.abs(p))] < 0:
        p = -p
    ax2.plot(x2, p, color=col, lw=1.9, zorder=4, label=lab)
ax2.axhline(0, color="#aab2bd", lw=.8, zorder=2)
ax2.set_ylim(-4.6, 4.6); ax2.set_yticks([])
ax2.legend(fontsize=9.5, loc="lower center", ncol=2, frameon=False)

# ---- right: splitting vs neck depth (exponential control) ------------------------------
axR.semilogy(epss, splits, "o-", color="#2b6cb0", lw=2.0, ms=7, zorder=3,
             label="double_lump doublet $E_1-E_0$ (tunneling)")
axR.axhline(ctrl, color="#b83280", ls="--", lw=1.6,
            label=f"lumpy (single well): ordinary spacing $\\approx${ctrl:.1f}")
for e, s in zip(epss, splits):
    axR.annotate(f"{s:.1e}", (e, s), textcoords="offset points", xytext=(6, 6),
                 fontsize=8.5, color="#3a4658")
axR.set_xlabel("neck depth  $\\epsilon$  (barrier height $\\to$)")
axR.set_ylabel("ground-doublet splitting")
axR.set_title("tunneling is exponentially geometry-tunable", fontsize=11.5)
axR.grid(True, which="both", alpha=.25)
axR.legend(fontsize=9, loc="upper right")

fig.tight_layout()
fig.savefig("tunneling.png", dpi=130, bbox_inches="tight")
print("wrote tunneling.png ; splittings:", [f"{s:.2e}" for s in splits], "control:", f"{ctrl:.2f}")
