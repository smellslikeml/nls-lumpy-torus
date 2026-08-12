"""
Frontier prototype 3: quantized vortices on a curved superfluid shell.

A defocusing condensate (Gross-Pitaevskii, sigma=+1) supports quantized vortices;
on a curved surface a vortex feels a geometric force from the Gaussian curvature.
We launch a vortex-antivortex pair at the belly and track the cores: the geometry
bends their motion and they migrate toward the negative-curvature necks -- the
lumpy-torus toy of curved-shell BECs (NASA Cold Atom Lab bubble traps).
"""
import numpy as np
from nls_lumpy_torus import build_operators, run, profile_A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
grid = build_operators(96, 160)
x, th = grid["x"], grid["th"]
Nx, Nth = grid["Nx"], grid["Nth"]
X, TH = np.meshgrid(x, th, indexing="ij")
rho0, xi = 1.0, 0.16


def vortex_phase(xv, tv, q):
    dx = X - xv
    dt = np.angle(np.exp(1j * (TH - tv)))
    return q * np.arctan2(dt, dx), np.sqrt(dx ** 2 + dt ** 2)


# vortex-antivortex pair at the belly (x=0), separated in theta
ph1, r1 = vortex_phase(0.0, np.pi - 0.55, +1)
ph2, r2 = vortex_phase(0.0, np.pi + 0.55, -1)
U0 = (np.sqrt(rho0) * np.tanh(r1 / xi) * np.tanh(r2 / xi) * np.exp(1j * (ph1 + ph2))).ravel()

U, hist, snaps, stats = run(grid, U0, dt=2e-3, T=8.0, sigma=+1.0, p=2,
                            n_snapshots=110, verbose=False)
print(f"  mass drift {(hist['mass'][-1]-hist['mass'][0])/hist['mass'][0]:+.1e}, "
      f"max picard {max(stats['picard_iters'])}", flush=True)


def cores(U):                                    # phase-singularity detection (robust to sound)
    ph = np.angle(U.reshape(Nx, Nth))

    def wd(a, b):
        return np.angle(np.exp(1j * (b - a)))     # wrapped phase difference
    # circulation around each plaquette (i,j)->(i,j+1)->(i+1,j+1)->(i+1,j)
    w = (wd(ph, np.roll(ph, -1, 1))
         + wd(np.roll(ph, -1, 1), np.roll(np.roll(ph, -1, 0), -1, 1))
         + wd(np.roll(np.roll(ph, -1, 0), -1, 1), np.roll(ph, -1, 0))
         + wd(np.roll(ph, -1, 0), ph))
    ip, jp = np.unravel_index(np.argmax(w), w.shape)    # +1 vortex
    im, jm = np.unravel_index(np.argmin(w), w.shape)    # -1 vortex
    return [(x[ip], th[jp]), (x[im], th[jm])]


t = np.array(snaps["t"])
C = [cores(U) for U in snaps["U"]]
c0 = np.array([c[0] for c in C]); c1 = np.array([c[1] for c in C])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
K = 2 * np.cos(2 * x) / (3 + np.cos(2 * x)) + np.sin(2 * x) ** 2 / (3 + np.cos(2 * x)) ** 2
im = ax[0].pcolormesh(th, x, np.broadcast_to(K[:, None], (Nx, Nth)), cmap="RdBu_r",
                      vmin=-1, vmax=0.5, shading="auto")
ax[0].plot(c0[:, 1], c0[:, 0], ".", color="k", ms=2)
ax[0].plot(c1[:, 1], c1[:, 0], ".", color="0.4", ms=2)
ax[0].axhline(0, color="0.5", lw=.6); ax[0].axhline(np.pi / 2, color="0.5", ls=":", lw=.8)
ax[0].axhline(-np.pi / 2, color="0.5", ls=":", lw=.8)
ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel("x")
ax[0].set_yticks([-np.pi / 2, 0, np.pi / 2]); ax[0].set_yticklabels(["neck", "belly", "neck"])
ax[0].set_title("vortex tracks on the curvature-colored surface")
fig.colorbar(im, ax=ax[0], label="Gaussian curvature K")
ax[1].plot(t, np.abs(c0[:, 0]), color="k", label="core 1  |x|")
ax[1].plot(t, np.abs(c1[:, 0]), color="0.5", label="core 2  |x|")
ax[1].axhline(np.pi / 2, ls=":", c="#b83280"); ax[1].text(0.2, np.pi / 2 - 0.2, "neck", color="#b83280")
ax[1].axhline(0, ls=":", c="#2b6cb0"); ax[1].text(0.2, 0.05, "belly", color="#2b6cb0")
ax[1].set_xlabel("t"); ax[1].set_ylabel("|x| of core"); ax[1].set_title("cores migrate from belly toward the necks")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Vortices on a curved superfluid: Gaussian curvature steers the cores toward the necks",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/vortices.png", dpi=120)
print("wrote vortices.png", flush=True)
