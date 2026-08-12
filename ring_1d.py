"""
Two canonical 1-D regimes on the belly ring (A=1), by exact split-step Fourier.
  (a) Benjamin-Feir modulational instability (focusing): a uniform state is
      unstable; a tiny modulation grows into a soliton, with Fermi-Pasta-Ulam
      recurrence back toward the uniform state.
  (b) Dark solitons (defocusing): a pair of black solitons (density notches with
      pi phase jumps) are STABLE and stationary -- the defocusing counterpart of
      the focusing self-trapping.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Nth = 512
th = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
n = np.fft.fftfreq(Nth, d=1.0 / Nth)


def split_step(u0, sigma, T, Nt):
    dt = T / Nt
    Lh = np.exp(-1j * n ** 2 * dt / 2)     # i u_t = -u_thth  ->  exp(-i n^2 dt)
    u = u0.astype(complex).copy()
    carpet = np.empty((Nt, Nth))
    for i in range(Nt):
        u = np.fft.ifft(Lh * np.fft.fft(u))
        u = u * np.exp(-1j * sigma * np.abs(u) ** 2 * dt)
        u = np.fft.ifft(Lh * np.fft.fft(u))
        carpet[i] = np.abs(u) ** 2
    return carpet


fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

# (a) Benjamin-Feir MI, focusing sigma=-1
a = 1.0
u0 = a * (1 + 0.03 * np.cos(th))            # uniform + seed of the m=1 unstable mode
T = 22.0; Nt = 1400
cA = split_step(u0, -1.0, T, Nt)
im0 = ax[0].imshow(cA, origin="lower", aspect="auto", cmap="inferno",
                   extent=[0, 2 * np.pi, 0, T], vmax=np.percentile(cA, 99.5))
ax[0].set_title("Benjamin–Feir MI (focusing): uniform state → soliton, FPU recurrence")
ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel("t")
ax[0].set_xticks([0, np.pi, 2 * np.pi]); ax[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
fig.colorbar(im0, ax=ax[0], label=r"$|u|^2$")

# (b) dark solitons, defocusing sigma=+1: two black solitons (periodic)
w = 0.18
u0 = np.tanh((th - np.pi / 2) / w) * np.tanh((th - 3 * np.pi / 2) / w)
T = 18.0; Nt = 1200
cB = split_step(u0.astype(complex), +1.0, T, Nt)
im1 = ax[1].imshow(cB, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, 2 * np.pi, 0, T])
ax[1].set_title("Dark solitons (defocusing): two stationary density notches")
ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel("t")
ax[1].set_xticks([0, np.pi, 2 * np.pi]); ax[1].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
fig.colorbar(im1, ax=ax[1], label=r"$|u|^2$")

fig.suptitle("Modulational instability & dark solitons on the belly ring", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/ring_modulational.png", dpi=120)
print("wrote ring_modulational.png")
print(f"  BF: initial |u|^2={a**2:.2f}, peak reached {cA.max():.2f} (soliton) ; "
      f"dark: background {cB[0].max():.2f}, min {cB.min():.3f}")
