"""
Semiclassical meridian beam: a narrow, high-momentum Gaussian beam launched at
the belly (x=0) and propagating poloidally (in x) around the meridian loop,
running over the bumps.

Semiclassical regime: momentum q large, transverse width w0 ~ small, so the
diffraction (Rayleigh) length ~ q*w0^2 exceeds the bump spacing ~pi/2. Then the
geometry controls the transverse (theta) width: it should COMPRESS at the necks
(A min, the Jacobi field J=A is smallest -> meridians converge) and EXPAND over
the belly (A max). Linear (sigma=0) to isolate the pure geodesic focusing.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, run, profile_A
from render import animate_torus, animate_chart

OUT = "/home/thorax/nls_lumpy_torus"
Lx = np.pi

# --- semiclassical parameters ---
Nx, Nth = 160, 224
q = 20            # poloidal momentum (even integer for x-period pi)
wx = 0.55         # along-meridian extent (packet)
wth = 0.32        # transverse width  (Rayleigh q*wth^2 ~ 2.0 > pi/2 bump spacing)
dt, T = 4e-4, 0.4
sigma = 0.0

grid = build_operators(Nx, Nth)
x, th = grid["x"], grid["th"]
X, TH = np.meshgrid(x, th, indexing="ij")
dthc = np.angle(np.exp(1j * (TH - np.pi)))
U0 = (np.exp(-(X ** 2) / (2 * wx ** 2))
      * np.exp(-dthc ** 2 / (2 * wth ** 2)) * np.exp(1j * q * X)).ravel()

U, hist, snaps, stats = run(grid, U0, dt=dt, T=T, sigma=sigma, p=2,
                            n_snapshots=100, verbose=True)
print(f"  mass drift {(hist['mass'][-1]-hist['mass'][0])/hist['mass'][0]:+.1e}, "
      f"max picard {max(stats['picard_iters'])}")


def diagnostics(U):
    f = np.abs(U.reshape(Nx, Nth)) ** 2
    fx = f.sum(axis=1)                                   # x-marginal
    ang = np.angle(np.sum(fx * np.exp(1j * 2 * np.pi * x / Lx)))
    xcm = ang / (2 * np.pi) * Lx
    i_br = int(np.argmax(fx))                            # brightest x-slice
    P = f[i_br, :]
    d = np.angle(np.exp(1j * (th - np.pi)))
    wtheta = np.sqrt(np.sum(P * d ** 2) / (P.sum() + 1e-300))
    return xcm, float(profile_A(x[i_br])), wtheta, float(f.max())


t = np.array(snaps["t"])
D = np.array([diagnostics(U) for U in snaps["U"]])
xcm, A_br, wth_t, peak = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
phys = A_br * wth_t                                       # physical transverse width

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, phys, "C0", lw=1.8, label="physical transverse width  $A\\,w_\\theta$")
ax0b = ax[0].twinx()
ax0b.plot(t, A_br, "C7", ls="--", lw=1, label="A at beam (1=belly, 0.707=neck)")
ax0b.set_ylabel("A(beam)"); ax[0].set_ylabel("width")
ax[0].set_title("Beam compresses at the necks (A min), expands over the belly (A max)")
ax[0].legend(loc="upper left"); ax0b.legend(loc="upper right"); ax[0].grid(alpha=0.3)
ax[1].plot(t, peak, "C3", lw=1.8)
ax[1].set_ylabel(r"peak $|u|^2$"); ax[1].set_xlabel("t"); ax[1].grid(alpha=0.3)
ax[1].set_title(r"peak intensity (high when focused at a neck)")
fig.tight_layout(); fig.savefig(f"{OUT}/semiclassical_breathing.png", dpi=120)
plt.close(fig)
print("wrote semiclassical_breathing.png")

animate_chart(grid, snaps, f"{OUT}/nls_semiclassical_chart.gif", fps=20)
animate_torus(grid, snaps, f"{OUT}/nls_semiclassical_torus.gif", fps=20)
