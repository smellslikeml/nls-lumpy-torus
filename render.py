"""
Render the NLS-on-lumpy-torus solution: the diagnostics the post omits
(mass/energy conservation) + the evolution of |u|^2 on the (x,theta) chart and
on a representative lumpy-torus embedding.

Embedding note: the field u(x,theta) is metric-exact (solved with ds^2 =
dx^2 + A^2 dtheta^2). The 3-D torus is an *illustrative* canvas -- poloidal
angle phi = 2*pi*(x-x0)/Lx, tube radius modulated by A(x) so the intrinsic
"lumpiness" (A in [0.707,1]) is visible -- matching the post's figure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from nls_lumpy_torus import (build_operators, initial_condition,
                             beam_along_geodesic, run, profile_A)

OUT = "/home/thorax/nls_lumpy_torus"


# ------------------------------------------------------------------ 3D embedding
def torus_embedding(x, th, R=2.2):
    """Crescent immersion built on the metric-faithful profile (A(x), z(x)),
    z=int sqrt(1-A'^2) dx, over one period bent closed. The field coordinate x
    (belly x=0 <-> neck x=+-pi/2) maps to the toroidal angle Phi(x); theta is the
    poloidal angle. A(x) is the poloidal tube radius, so the tube is fat at the
    belly (A=1) and pinched at the neck (A=0.707) -> one bulge + one waist."""
    A = profile_A(x)                                    # (Nx,)
    Ap = -np.sin(2 * x) / (4 * A)
    g = np.sqrt(np.clip(1 - Ap ** 2, 0, 1))
    z = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    L = z[-1] - z[0] + float(np.mean(np.diff(z)))       # full period incl. seam step
    Phi = 2.0 * np.pi * (z - z[0]) / L                  # toroidal, closes over one period
    PHI, TH = np.meshgrid(Phi, th, indexing="ij")
    AA, _ = np.meshgrid(A, th, indexing="ij")
    rho = R + AA * np.cos(TH)
    X = rho * np.cos(PHI)
    Y = rho * np.sin(PHI)
    Z = AA * np.sin(TH)
    return X, Y, Z


def _wrap(M):                                            # close periodic seams
    M = np.concatenate([M, M[:1, :]], axis=0)
    M = np.concatenate([M, M[:, :1]], axis=1)
    return M


# ------------------------------------------------------------------ conservation
def plot_conservation(hist, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    t = hist["t"]
    dm = (hist["mass"] - hist["mass"][0]) / hist["mass"][0]
    de = (hist["energy"] - hist["energy"][0]) / (abs(hist["energy"][0]) + 1e-300)
    ax[0].plot(t, dm); ax[0].set_title("relative mass drift  (M-M0)/M0")
    ax[0].set_xlabel("t"); ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[1].plot(t, de, color="C1"); ax[1].set_title("relative energy drift  (E-E0)/|E0|")
    ax[1].set_xlabel("t"); ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[2].plot(t, hist["maxabs2"], color="C3"); ax[2].set_title(r"peak $|u|^2$  (concentration)")
    ax[2].set_xlabel("t")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ chart GIF
def animate_chart(grid, snaps, path, fps=15):
    x, th = grid["x"], grid["th"]
    Nx, Nth = grid["Nx"], grid["Nth"]
    fields = [np.abs(U.reshape(Nx, Nth)) ** 2 for U in snaps["U"]]
    vmax = max(f.max() for f in fields)
    frames = []
    for k, (t, f) in enumerate(zip(snaps["t"], fields)):
        fig, a = plt.subplots(figsize=(6.4, 4.2))
        im = a.pcolormesh(th, x, f, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        a.set_xlabel(r"$\theta$"); a.set_ylabel(r"$x$ (poloidal)")
        a.set_title(rf"$|u|^2$ on the $(x,\theta)$ chart   t={t:.2f}")
        fig.colorbar(im, ax=a, label=r"$|u|^2$")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print("wrote", path, f"({len(frames)} frames)")


# ------------------------------------------------------------------ torus GIF
def animate_torus(grid, snaps, path, fps=15, stride=1, vmax=None):
    from matplotlib import cm, colors
    x, th = grid["x"], grid["th"]
    Nx, Nth = grid["Nx"], grid["Nth"]
    X, Y, Z = torus_embedding(x, th)
    if stride > 1:                                       # subsample for speed at high res
        X, Y, Z = X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride]
    Xw, Yw, Zw = _wrap(X), _wrap(Y), _wrap(Z)
    fields = [np.abs(U.reshape(Nx, Nth)) ** 2 for U in snaps["U"]]
    vmax = vmax if vmax is not None else max(f.max() for f in fields)
    norm = colors.Normalize(vmin=0, vmax=vmax)
    cmap = matplotlib.colormaps["inferno"]
    frames = []
    for t, f in zip(snaps["t"], fields):
        if stride > 1:
            f = f[::stride, ::stride]
        fc = cmap(norm(_wrap(f)))
        fig = plt.figure(figsize=(6.4, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xw, Yw, Zw, facecolors=fc, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_box_aspect((1, 1, 0.42)); ax.set_axis_off()
        ax.view_init(elev=42, azim=-60)
        ax.set_title(rf"$|u|^2$ on the lumpy torus   t={t:.2f}")
        m = cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.6, label=r"$|u|^2$")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print("wrote", path, f"({len(frames)} frames)")


# ------------------------------------------------------------- transverse profile
def transverse_profile(grid, U, xc, Lx=np.pi):
    """x-marginal of |u|^2 (mass-weighted): peak |u|^2 and RMS width about xc."""
    Nx, Nth = grid["Nx"], grid["Nth"]
    f = np.abs(U.reshape(Nx, Nth)) ** 2
    Px = f.sum(axis=1) * grid["A"] * grid["dx"] * grid["dth"]   # mass per x-node
    x = grid["x"]
    dxc = (x - xc + Lx / 2.0) % Lx - Lx / 2.0
    width = np.sqrt(np.sum(Px * dxc ** 2) / (Px.sum() + 1e-300))
    return float(f.max()), float(width)


def plot_geodesic_comparison(results, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, col in zip(results, ["C0", "C3"]):
        r = results[name]
        ax[0].plot(r["t"], r["peak"], color=col, label=name)
        ax[1].plot(r["t"], r["width"], color=col, label=name)
    ax[0].set_title(r"peak $|u|^2$  (concentration)")
    ax[1].set_title(r"transverse RMS width in $x$  (spreading)")
    for a in ax:
        a.set_xlabel("t"); a.grid(alpha=0.3); a.legend()
    fig.suptitle("Beam on the elliptic (stable) vs hyperbolic (neck) geodesic — focusing cubic NLS")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    grid = build_operators(Nx=80, Nth=160)
    sigma, p = -1.0, 2
    cases = [("elliptic (x=0)", 0.0, "nls_geodesic_elliptic.gif"),
             ("hyperbolic neck (x=pi/2)", np.pi / 2.0, "nls_geodesic_hyperbolic.gif")]
    results = {}
    for name, xc, gif in cases:
        print(f"\n=== beam along {name} geodesic ===")
        U0 = beam_along_geodesic(grid, xc=xc, amp=0.9, wx=0.25, k=6)
        U, hist, snaps, stats = run(grid, U0, dt=2e-3, T=3.0, sigma=sigma, p=p,
                                    n_snapshots=60, verbose=True)
        print(f"  max picard iters = {max(stats['picard_iters'])}")
        pk = np.array([transverse_profile(grid, Us, xc) for Us in snaps["U"]])
        results[name] = dict(t=np.array(snaps["t"]), peak=pk[:, 0], width=pk[:, 1])
        animate_torus(grid, snaps, f"{OUT}/{gif}")
    plot_geodesic_comparison(results, f"{OUT}/geodesic_comparison.png")
