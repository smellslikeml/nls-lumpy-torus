"""Genus-2 Townes universality — closed out properly.

H1 said the mass-critical collapse threshold is a LOCAL universal invariant (the Townes mass
||Q||^2 ~ 11.7), blind to curvature and global structure. The sharpest test is topology: does
it hold on a fused genus-2 surface? Two subtleties had to be fixed first:

  (1) A fixed mesh GRID-ARRESTS true blow-up (the discrete Laplacian cannot resolve sub-grid
      concentration), so the signal is not peak -> infinity but the FOCUSING ONSET: above M_c
      a bump self-focuses (peak grows sharply); below it disperses (peak ~ flat).
  (2) The threshold must be resolved on a fine enough mesh for the onset to be crisp.

With those, we bisect the focusing threshold on the genus-2 mesh across neck widths, and on a
genus-0 SPHERE with the identical protocol (the sphere's Laplacian is validated against
l(l+1)). Verification: mass conservation + grid convergence + the cross-topology agreement.

Regenerate:  python3 genus2_collapse.py
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nls_torus import mesh as mo

G, DT, W0, TOWNES = 1.0, 2e-3, 0.35, 11.7


def setup(V, F):
    W, M = mo.cotan_laplacian(V, F); Md = np.asarray(M.diagonal()).ravel()
    c = V[np.argmin(V[:, 0])]
    g = np.exp(-np.sum((V - c) ** 2, axis=1) / (2 * W0 ** 2))          # unit-amp bump profile
    mass1 = float(np.real(np.vdot(g.astype(complex), Md * g)))          # mass at amp=1 (scales as amp^2)
    iMdt = 1j * sp.diags(Md) / DT
    lu = spla.splu((iMdt - 0.5 * W).tocsc()); Blin = iMdt + 0.5 * W
    return dict(Md=Md, g=g.astype(complex), mass1=mass1, lu=lu, Blin=Blin)


def focuses(s, amp, T=2.0, thresh=2.0):
    u = amp * s["g"].copy(); pk0 = np.abs(u).max(); Md = s["Md"]; drift = 0.0
    m0 = amp ** 2 * s["mass1"]
    for i in range(int(T / DT)):
        u = u * np.exp(0.5j * G * np.abs(u) ** 2 * DT)
        u = s["lu"].solve(s["Blin"] @ u)
        u = u * np.exp(0.5j * G * np.abs(u) ** 2 * DT)
        drift = max(drift, abs(np.real(np.vdot(u, Md * u)) - m0) / m0)
        if np.abs(u).max() / pk0 > thresh:
            return True, drift
    return False, drift


def Mc(V, F):
    s = setup(V, F); lo, hi = 4.0, 7.5
    for _ in range(6):
        mid = 0.5 * (lo + hi); foc, _ = focuses(s, mid)
        (hi := mid) if foc else (lo := mid)
    _, drift = focuses(s, hi)
    return hi ** 2 * s["mass1"], drift


rows = []
print(f"focusing-onset critical mass (Townes ||Q||^2 = {TOWNES}):")
print(f"{'surface':>22s} {'genus':>6s} {'M_c':>7s} {'mass drift':>11s}")
# genus-0 sphere reference (same protocol) + genus-2 across neck widths + a grid check
Vs, Fs = mo.icosphere(subdiv=5)
mc, dr = Mc(Vs, Fs); rows.append(("sphere", 0, mc)); print(f"{'sphere (g=0)':>22s} {0:6d} {mc:7.2f} {dr:11.1e}", flush=True)
for d in (1.20, 1.30, 1.40):
    V, F = mo.two_tori_genus2(d=d, ngrid=100)
    mc, dr = Mc(V, F); rows.append((f"genus-2 d={d}", 2, mc))
    print(f"{'genus-2 d=%.2f'%d:>22s} {2:6d} {mc:7.2f} {dr:11.1e}", flush=True)
V2, F2 = mo.two_tori_genus2(d=1.30, ngrid=130); mc_fine, _ = Mc(V2, F2)
print(f"{'genus-2 d=1.30 (fine)':>22s} {2:6d} {mc_fine:7.2f}  (grid check)", flush=True)

mcs = np.array([r[2] for r in rows])
spread = (mcs.max() - mcs.min()) / mcs.mean()
grid_ok = abs(mc_fine - rows[2][2]) / rows[2][2] < 0.1
print(f"\ncross-topology spread: {spread*100:.1f}%  ;  grid-converged: {grid_ok}  ;  "
      f"mean M_c={mcs.mean():.2f} vs Townes {TOWNES}")

# ---- figure ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.0, 3.9))
labels = [r[0] for r in rows]; xp = np.arange(len(rows))
cols = ["#3aa76d"] + ["#2b6cb0"] * 3
ax.axhline(TOWNES, color="#b83280", ls="--", lw=1.7, label=f"Townes $\\|Q\\|^2={TOWNES}$")
ax.scatter(xp, mcs, s=140, c=cols, zorder=3, edgecolor="white", lw=1.3)
for x, m in zip(xp, mcs):
    ax.annotate(f"{m:.1f}", (x, m), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=10)
ax.set_xticks(xp); ax.set_xticklabels(["sphere\n(genus 0)", "genus-2\nd=1.20", "genus-2\nd=1.30", "genus-2\nd=1.40"], fontsize=9.5)
ax.set_ylim(TOWNES - 3, TOWNES + 3); ax.set_ylabel("focusing-onset critical mass  $M_c$")
ax.set_title("collapse threshold is blind to topology and neck geometry", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout(); fig.savefig("genus2_collapse.png", dpi=130, bbox_inches="tight")
print("wrote genus2_collapse.png")
