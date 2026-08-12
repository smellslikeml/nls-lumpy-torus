"""
Direction 6: driven-dissipative regime -> geometry-pinned Kerr solitons.

The Lugiato-Lefever equation (a driven, damped NLS) is the model of soliton
microcombs in ring resonators -- and a microtoroid IS a lumpy torus. A lump
enters as a local shift of the detuning; it acts as a pinning site that captures
a drifting dissipative soliton. Split-step; a soliton seeded off-centre drifts to
the geometric defect and locks there.
   d_t u = -(1 + i*alpha(theta)) u + i|u|^2 u + i d_theta^2 u + F
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Nth = 512
th = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
n = np.fft.fftfreq(Nth, d=1.0 / Nth)

alpha0, F = 4.0, 1.5                        # detuning, drive (cavity-soliton regime)
dth_c = np.angle(np.exp(1j * (th - np.pi)))
dalpha = -1.0 * np.exp(-dth_c ** 2 / (2 * 0.4 ** 2))    # geometric defect: local detuning dip at theta=pi
alpha = alpha0 + dalpha

T, Nt = 200.0, 60000
dt = T / Nt
Lin = np.exp((-(1 + 1j * alpha0) - 1j * n ** 2) * (dt / 2))   # constant linear part (Fourier)
# seed ~ exact cavity soliton (sech on the lower-branch background)
d0 = np.angle(np.exp(1j * (th - np.pi / 2)))
u = (0.42 + 2.6 / np.cosh(d0 / 0.42)).astype(complex)

carpet = np.empty((Nt // 100, Nth)); ci = 0
for i in range(Nt):
    u = np.fft.ifft(Lin * np.fft.fft(u))
    u = u + dt * (-1j * dalpha * u + 1j * np.abs(u) ** 2 * u + F)   # defect + Kerr + drive
    u = np.fft.ifft(Lin * np.fft.fft(u))
    if i % 100 == 0:
        carpet[ci] = np.abs(u) ** 2; ci += 1
carpet = carpet[:ci]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
im = ax[0].imshow(carpet, origin="lower", aspect="auto", cmap="inferno",
                  extent=[0, 2 * np.pi, 0, T], vmax=np.percentile(carpet, 99.7))
ax[0].axvline(np.pi, color="c", ls=":", lw=1)
ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel("t")
ax[0].set_xticks([0, np.pi, 2 * np.pi]); ax[0].set_xticklabels(["0", r"$\pi$ (defect)", r"$2\pi$"])
ax[0].set_title("a Kerr soliton drifts to the geometric defect and locks")
fig.colorbar(im, ax=ax[0], label=r"$|u|^2$")
ax[1].plot(th, np.abs(u) ** 2, color="#b83280", lw=1.6, label=r"$|u|^2$ (final)")
ax[1].plot(th, F ** 2 + 0 * th, color="0.7", ls="--", lw=1)
ax2 = ax[1].twinx(); ax2.plot(th, alpha, color="#2b6cb0", lw=1, alpha=.6)
ax2.set_ylabel("detuning α(θ)", color="#2b6cb0")
ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel(r"$|u|^2$")
ax[1].set_xticks([0, np.pi, 2 * np.pi]); ax[1].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax[1].set_title("pinned dissipative soliton at the defect"); ax[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Lugiato–Lefever on the lumpy torus: a lump pins a dissipative Kerr soliton", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/lle.png", dpi=120)
print(f"wrote lle.png  (final peak |u|^2 = {np.max(np.abs(u)**2):.2f} at theta = {th[np.argmax(np.abs(u)**2)]:.2f})")
