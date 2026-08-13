"""Genus-2 tunnelling sweep — H4 on true higher-genus topology.

On the two-lobe torus (H4) the ground-doublet splitting was the inter-lobe tunnelling rate.
Here the same idea on a genuine genus-2 surface: the first excited Laplace-Beltrami mode is
the "which-handle" doublet (one sign per handle, node at the connecting neck), so lambda_1
is the inter-handle splitting. Thin the neck (separate the two tori) and it falls — the
handles decouple toward two independent zero modes. Geometry controls tunnelling across a
handle, verified genus-2 at every point.

Regenerate:  python3 genus2_tunneling.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nls_torus import mesh as mo

ds = [1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
rows = []
mesh_for_fig = None
for d in ds:
    V, F = mo.two_tori_genus2(d=d, ngrid=80)
    chi, g = mo.euler_genus(V, F)
    vals, vecs, _ = mo.laplace_spectrum(V, F, k=5)
    lam1 = float(vals[1])                       # which-handle doublet splitting (lam0 ~ 0)
    isolated = float(vals[2] / max(vals[1], 1e-9))   # gap to next mode (>>1 => a real doublet)
    rows.append(dict(d=d, genus=g, lam0=float(vals[0]), lam1=lam1, iso=isolated))
    if abs(d - 1.30) < 1e-6:
        mesh_for_fig = (V, F, vecs[:, 1], lam1)
    print(f"d={d:.2f}  genus={g}  lam0={vals[0]:+.2e}  lam1(splitting)={lam1:.4f}  "
          f"lam2/lam1={isolated:.1f}")

ok = all(r["genus"] == 2 for r in rows)
print(f"\nVERIFICATION all meshes genus-2: {ok} ; all doublets isolated (lam2/lam1>2): "
      f"{all(r['iso'] > 2 for r in rows)}")

# ---- figure ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11.8, 4.6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
V, F, mode, lam1 = mesh_for_fig
mode = mode / np.abs(mode).max()
tris = Poly3DCollection(V[F], linewidths=0)
tris.set_array(mode[F].mean(axis=1)); tris.set_cmap("RdBu_r"); tris.set_clim(-1, 1)
ax1.add_collection3d(tris)
lim = np.abs(V).max()
ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
ax1.set_box_aspect((1, 0.5, 0.7)); ax1.view_init(elev=52, azim=-60); ax1.set_axis_off()
ax1.set_title("genus-2 which-handle doublet\n(one sign per handle, node at the neck)", fontsize=11)

ax2 = fig.add_subplot(1, 2, 2)
dd = [r["d"] for r in rows]; ll = [r["lam1"] for r in rows]
ax2.plot(dd, ll, "o-", color="#2b6cb0", lw=2.0, ms=7)
for r in rows:
    ax2.annotate(f"{r['lam1']:.3f}", (r["d"], r["lam1"]), textcoords="offset points",
                 xytext=(4, 7), fontsize=8.5, color="#3a4658")
ax2.set_xlabel("torus separation  $d$  (thinner connecting neck $\\to$)")
ax2.set_ylabel("inter-handle splitting  $\\lambda_1$")
ax2.set_title("neck geometry tunes handle-to-handle tunnelling", fontsize=11.5)
ax2.grid(True, alpha=.25)
fig.tight_layout(); fig.savefig("genus2_tunneling.png", dpi=130, bbox_inches="tight")
print("wrote genus2_tunneling.png")
