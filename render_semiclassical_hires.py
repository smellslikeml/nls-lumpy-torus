"""
Higher-resolution semiclassical meridian beam. Pushes further into the
semiclassical corner (q=40, finer 300x300 grid, smaller dt) so the beam is more
ray-like and crisper. Linear (sigma=0); mass/energy conserve to machine
precision. The transverse width still tracks A(beam) -- compress at necks,
expand over the belly -- now with a sharper beam for the first ~1-2 transits
before diffraction (Rayleigh ~ q*w0^2 ~ one loop) takes over.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, run, profile_A
from render import animate_torus, animate_chart

OUT = "/home/thorax/nls_lumpy_torus"
Lx = np.pi

Nx, Nth = 300, 300
q = 40
wx = 0.45
wth = 0.30
dt, T = 1e-4, 0.15
sigma = 0.0

print(f"building {Nx}x{Nth} operators + factorizing (one-time)...", flush=True)
grid = build_operators(Nx, Nth)
x, th = grid["x"], grid["th"]
X, TH = np.meshgrid(x, th, indexing="ij")
dthc = np.angle(np.exp(1j * (TH - np.pi)))
U0 = (np.exp(-(X ** 2) / (2 * wx ** 2))
      * np.exp(-dthc ** 2 / (2 * wth ** 2)) * np.exp(1j * q * X)).ravel()

U, hist, snaps, stats = run(grid, U0, dt=dt, T=T, sigma=sigma, p=2,
                            n_snapshots=90, verbose=True)
print(f"  mass drift {(hist['mass'][-1]-hist['mass'][0])/hist['mass'][0]:+.1e}, "
      f"energy drift {(hist['energy'][-1]-hist['energy'][0])/(abs(hist['energy'][0])+1e-300):+.1e}, "
      f"max picard {max(stats['picard_iters'])}", flush=True)


def diagnostics(U):
    f = np.abs(U.reshape(Nx, Nth)) ** 2
    fx = f.sum(axis=1)
    ang = np.angle(np.sum(fx * np.exp(1j * 2 * np.pi * x / Lx)))
    i_br = int(np.argmax(fx))
    P = f[i_br, :]
    d = np.angle(np.exp(1j * (th - np.pi)))
    wtheta = np.sqrt(np.sum(P * d ** 2) / (P.sum() + 1e-300))
    return ang / (2 * np.pi) * Lx, float(profile_A(x[i_br])), wtheta, float(f.max())


t = np.array(snaps["t"])
D = np.array([diagnostics(U) for U in snaps["U"]])
xcm, A_br, wth_t, peak = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
phys = A_br * wth_t

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, phys, "C0", lw=1.8, label=r"physical transverse width $A\,w_\theta$")
ax0b = ax[0].twinx()
ax0b.plot(t, A_br, "C7", ls="--", lw=1, label="A at beam (1=belly, 0.707=neck)")
ax0b.set_ylabel("A(beam)"); ax[0].set_ylabel("width")
ax[0].set_title(f"Semiclassical q={q}, {Nx}x{Nth}: compress at necks, expand over belly")
ax[0].legend(loc="upper left"); ax0b.legend(loc="upper right"); ax[0].grid(alpha=0.3)
ax[1].plot(t, peak, "C3", lw=1.8)
ax[1].set_ylabel(r"peak $|u|^2$"); ax[1].set_xlabel("t"); ax[1].grid(alpha=0.3)
ax[1].set_title("peak intensity (high when focused at a neck)")
fig.tight_layout(); fig.savefig(f"{OUT}/semiclassical_hires_breathing.png", dpi=120)
plt.close(fig)
print("wrote semiclassical_hires_breathing.png", flush=True)

animate_chart(grid, snaps, f"{OUT}/nls_semiclassical_hires_chart.gif", fps=20)
animate_torus(grid, snaps, f"{OUT}/nls_semiclassical_hires_torus.gif", fps=20, stride=2)
print("ALL DONE", flush=True)
