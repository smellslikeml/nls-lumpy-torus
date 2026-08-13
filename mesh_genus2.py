"""First physics on a genus-2 surface — past what surfaces of revolution can express.

Builds a connected genus-2 mesh, shows its low Laplace-Beltrami spectrum (the "shape of a
two-handled drum") and the handle-localized doublet, then evolves a focusing NLS wavepacket
on the mesh with a Crank-Nicolson stepper (sparse cotan operator) — verified by mass
conservation. This is the mesh extension carrying the project's physics onto new topology.

Regenerate:  python3 mesh_genus2.py
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nls_torus import mesh as mo

# ---- build + verify the genus-2 surface ----------------------------------------------
V, F = mo.two_tori_genus2(d=1.35, ngrid=72)
chi, g = mo.euler_genus(V, F)
print(f"genus-2 mesh: {len(V)} verts, {len(F)} faces, chi={chi}, genus={g}")
assert g == 2, "generator did not produce genus 2"

vals, vecs, (W, M) = mo.laplace_spectrum(V, F, k=8)
print("low -Delta spectrum:", np.round(vals, 4))
gap = vals[1] - vals[0]
print(f"ground state (constant) lambda0={vals[0]:.2e}; first gap lambda1={vals[1]:.4f}")

# ---- focusing NLS on the mesh: localized bump on one handle -> mass-conserved evolution
# Strang split: exact pointwise phase rotation (nonlinear) + prefactored linear CN. Both
# substeps are norm-preserving, so mass is conserved to machine precision and each step is
# two triangular solves against a factorization built once.
Md = np.asarray(M.diagonal()).ravel()
c1 = V[np.argmin(V[:, 0])]                       # a point on the left handle
r2 = np.sum((V - c1) ** 2, axis=1)
u = (2.2 * np.exp(-r2 / (2 * 0.45 ** 2))).astype(complex)
dt, g_nl, nsteps = 4e-3, 4.0, 320

iMdt = 1j * sp.diags(Md) / dt
Alin = (iMdt - 0.5 * W).tocsc(); Blin = (iMdt + 0.5 * W)
lu = spla.splu(Alin)                             # factor the constant linear CN operator once


def mass(u):
    return float(np.real(np.vdot(u, Md * u)))


m0 = mass(u)
masses, peaks = [m0], [float(np.abs(u).max())]
for n in range(nsteps):
    u = u * np.exp(0.5j * g_nl * np.abs(u) ** 2 * dt)   # half nonlinear (exact, |u| preserved)
    u = lu.solve(Blin @ u)                              # full linear CN (unitary in M-metric)
    u = u * np.exp(0.5j * g_nl * np.abs(u) ** 2 * dt)   # half nonlinear
    masses.append(mass(u)); peaks.append(float(np.abs(u).max()))
drift = max(abs(m - m0) for m in masses) / m0
print(f"NLS evolution: {nsteps} steps, mass drift {drift:.2e}, peak {peaks[0]:.2f} -> {peaks[-1]:.2f}")
print(f"VERIFICATION mass_conserved={drift < 1e-6}")

# ---- figure ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11.8, 4.7))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
mode = vecs[:, 1]; mode = mode / np.abs(mode).max()
fc = mode[F].mean(axis=1)                          # per-face color from the eigenmode
tris = Poly3DCollection(V[F], linewidths=0)
tris.set_array(fc); tris.set_cmap("RdBu_r"); tris.set_clim(-1, 1)
ax1.add_collection3d(tris)
lim = np.abs(V).max()
ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
ax1.set_box_aspect((1, 0.5, 0.7)); ax1.view_init(elev=52, azim=-60); ax1.set_axis_off()
ax1.set_title(f"genus-2 surface, first excited mode\n$-\\Delta$: $\\lambda_1={vals[1]:.3f}$ "
              "(the two-handle doublet)", fontsize=11)

ax2 = fig.add_subplot(1, 2, 2)
t = np.arange(len(masses)) * dt
ax2.plot(t, np.array(masses) / m0 - 1, color="#2b6cb0", lw=1.9)
ax2.set_xlabel("time"); ax2.set_ylabel("mass drift  $M(t)/M_0 - 1$", color="#2b6cb0")
ax2.tick_params(axis="y", labelcolor="#2b6cb0")
ax2.set_title(f"focusing NLS on the mesh — mass conserved to {drift:.0e}", fontsize=11)
ax2b = ax2.twinx()
ax2b.plot(t, peaks, color="#b83280", lw=1.6, ls="--")
ax2b.set_ylabel("peak amplitude", color="#b83280"); ax2b.tick_params(axis="y", labelcolor="#b83280")
fig.tight_layout(); fig.savefig("mesh_genus2.png", dpi=130, bbox_inches="tight")
print("wrote mesh_genus2.png")
