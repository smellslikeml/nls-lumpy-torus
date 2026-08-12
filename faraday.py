"""
Direction 5: a time-dependent metric drives parametric (Faraday) amplification.

Periodically modulating the geometry modulates the effective coupling on the
belly ring, g(t)=g0(1+delta cos Omega t). Driving near twice a Bogoliubov
frequency (Omega ~ 2 omega_n) makes that density mode grow exponentially from
noise -- Faraday waves. (A breathing torus is the same parametric structure; the
analogous monotonic expansion is the NLS analogue of cosmological particle
production.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Nth = 512
th = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
n = np.fft.fftfreq(Nth, d=1.0 / Nth)
rng = np.random.default_rng(1)

a = 1.0                                   # uniform background amplitude
g0, delta = 1.0, 0.5                      # defocusing background, modulation depth
nres = 3
omega = nres * np.sqrt(nres ** 2 + 2 * g0 * a ** 2)   # Bogoliubov freq of mode n
Omega = 2 * omega                          # parametric resonance
T, Nt = 24.0, 12000
dt = T / Nt
u = a * (1 + 0.01 * np.cos(nres * th)) + 1e-4 * (rng.standard_normal(Nth) + 1j * rng.standard_normal(Nth))

carpet = np.empty((Nt // 20, Nth)); amp3 = []; times = []; ci = 0
Lh = np.exp(-1j * n ** 2 * dt / 2)         # i u_t = -u_thth
for i in range(Nt):
    t = i * dt
    g = g0 * (1 + delta * np.cos(Omega * t))
    u = np.fft.ifft(Lh * np.fft.fft(u))
    u = u * np.exp(-1j * g * np.abs(u) ** 2 * dt)
    u = np.fft.ifft(Lh * np.fft.fft(u))
    if i % 20 == 0:
        carpet[ci] = np.abs(u) ** 2; ci += 1
    uk = np.fft.fft(u) / Nth
    amp3.append(np.abs(uk[nres])); times.append(t)
carpet = carpet[:ci]
amp3 = np.array(amp3); times = np.array(times)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
im = ax[0].imshow(carpet, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, 2 * np.pi, 0, T], vmax=carpet.max())
ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel("t")
ax[0].set_xticks([0, np.pi, 2 * np.pi]); ax[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax[0].set_title(f"Faraday pattern grows (mode n={nres})")
fig.colorbar(im, ax=ax[0], label=r"$|u|^2$")
ax[1].semilogy(times, amp3 + 1e-12, color="#b83280")
ax[1].set_xlabel("t"); ax[1].set_ylabel(f"amplitude of mode n={nres} (log)")
ax[1].set_title(r"exponential parametric growth ($\Omega=2\omega_n$)"); ax[1].grid(alpha=0.3)
fig.suptitle("Breathing torus → Faraday parametric amplification on the belly ring", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/faraday.png", dpi=120)
print(f"wrote faraday.png  (mode {nres}: {amp3[0]:.1e} -> {amp3.max():.1e})")
