"""
Quantum revivals (a Talbot carpet) on the belly ring.

The belly (x=0, A=1) is a unit-circumference ring; a wavepacket released on it
evolves under i u_t = -u_thth, eigenvalues n^2. The phases e^{-i n^2 t} realign
at t = 2*pi (full revival) and at rational fractions (fractional revivals),
producing the self-similar Talbot carpet -- the mechanism behind the dispersive
recurrence seen in the full runs. Spectral (exact) evolution.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Nth = 1024
th = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
n = np.fft.fftfreq(Nth, d=1.0 / Nth)                    # integer wavenumbers
d = np.angle(np.exp(1j * (th - np.pi)))
u0 = np.exp(-d ** 2 / (2 * 0.13 ** 2)).astype(complex)  # localized packet at theta=pi
uk = np.fft.fft(u0)

Nt = 900
ts = np.linspace(0, 2 * np.pi, Nt)                       # one full revival period
carpet = np.empty((Nt, Nth))
for i, t in enumerate(ts):
    u = np.fft.ifft(np.exp(-1j * n ** 2 * t) * uk)
    carpet[i] = np.abs(u) ** 2

fig, ax = plt.subplots(figsize=(8.5, 6))
im = ax.imshow(carpet, origin="lower", aspect="auto", cmap="magma",
               extent=[0, 2 * np.pi, 0, 2 * np.pi],
               vmax=np.percentile(carpet, 99.5))
ax.set_xlabel(r"$\theta$ (around the belly ring)")
ax.set_ylabel(r"time $t$")
ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$ (full revival)"])
ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax.set_title("Talbot carpet: revivals of a packet on the belly ring")
fig.colorbar(im, ax=ax, label=r"$|u|^2$")
fig.tight_layout()
fig.savefig(f"{OUT}/spacetime_carpet.png", dpi=120)
print("wrote spacetime_carpet.png")
