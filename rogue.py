"""
Frontier prototype 10: rogue waves (Peregrine limit) on the belly ring.

The focusing belly ring is Benjamin-Feir unstable (gallery 09). Seed a uniform
condensate with a whisper at the longest-wavelength unstable mode and the
modulational instability grows it into an Akhmediev breather: a wave that rises
"from nowhere", peaks far above the background, and recurs (Fermi-Pasta-Ulam). As
the seeded wavenumber drops well below the MI band edge the breather approaches the
PEREGRINE soliton -- the canonical rogue-wave prototype with peak amplitude 3x the
background. Geometry is the knob: A(x)/the effective coupling set the MI gain, hence
how tall and how often the rogue events come.

1-D focusing NLS on the ring:  i u_t + u_thth + g |u|^2 u = 0  (split-step Fourier).
MI gain sigma^2 = Q^2 (2 g a0^2 - Q^2); max at Q=sqrt(g) a0. With g=8, a0=1 the ring
mode Q=1 sits well below the band edge (Q_max~2.8) -> a tall, near-Peregrine breather.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE = "#b83280", "#2b6cb0"
Nth = 256
th = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
kf = np.fft.fftfreq(Nth, d=1.0 / Nth)              # integer ring wavenumbers
a0, g = 1.0, 8.0
T, Nt = 5.0, 50000
dt = T / Nt

u = a0 * (1.0 + 1e-3 * np.cos(th))                 # seed the Q=1 (longest) unstable mode
Lh = np.exp(-1j * kf ** 2 * dt / 2)                # linear half-step: i u_t + u_thth = 0

carpet = np.empty((Nt // 50, Nth)); ci = 0
tmax_amp, times = [], []
for i in range(Nt):
    u = np.fft.ifft(Lh * np.fft.fft(u))
    u = u * np.exp(1j * g * np.abs(u) ** 2 * dt)   # focusing nonlinear step
    u = np.fft.ifft(Lh * np.fft.fft(u))
    if i % 50 == 0:
        carpet[ci] = np.abs(u) ** 2; ci += 1
    tmax_amp.append(np.max(np.abs(u))); times.append(i * dt)
carpet = carpet[:ci]
tmax_amp = np.array(tmax_amp); times = np.array(times)

# first rogue peak
ipk = int(np.argmax(tmax_amp)); tpk = times[ipk]
peak_amp = tmax_amp[ipk]
# spatial profile at the peak (recompute state at tpk)
u2 = a0 * (1.0 + 1e-3 * np.cos(th))
npk = ipk
for i in range(npk):
    u2 = np.fft.ifft(Lh * np.fft.fft(u2)); u2 = u2 * np.exp(1j * g * np.abs(u2) ** 2 * dt)
    u2 = np.fft.ifft(Lh * np.fft.fft(u2))
prof = np.abs(u2)
print(f"first rogue peak |u|={peak_amp:.3f} (x background a0={a0}) at t={tpk:.3f}; "
      f"Peregrine bound = 3.0", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
im = ax[0].imshow(carpet, origin="lower", aspect="auto", cmap="inferno",
                  extent=[0, 2 * np.pi, 0, T])
ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel("t")
ax[0].set_xticks([0, np.pi, 2 * np.pi]); ax[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax[0].set_title("a rogue event rises from the ring, then recurs (FPU)")
fig.colorbar(im, ax=ax[0], label=r"$|u|^2$")

ax[1].plot(times, tmax_amp, color=PINK, lw=1.5)
ax[1].axhline(3 * a0, color=BLUE, ls="--", lw=1.3, label="Peregrine bound  3×")
ax[1].axhline(a0, color="0.6", ls=":", lw=1.0, label="background")
ax[1].plot(tpk, peak_amp, "o", color=PINK, ms=7)
ax[1].annotate(f"peak {peak_amp:.2f}×", (tpk, peak_amp), textcoords="offset points",
               xytext=(8, -4), fontsize=9, color=PINK)
ax[1].set_xlabel("t"); ax[1].set_ylabel(r"max$_\theta$ |u|  (background units)")
ax[1].set_title("peak amplitude approaches the Peregrine 3× limit")
ax[1].legend(loc="upper right", fontsize=9); ax[1].grid(alpha=0.3)

fig.suptitle("Rogue waves on the belly ring: modulational instability → near-Peregrine breather",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/rogue.png", dpi=120)
print("wrote rogue.png", flush=True)
