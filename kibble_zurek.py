"""
Frontier prototype 13: Kibble-Zurek scaling from a quench on the belly ring.

Ramp the belly ring from modulationally STABLE to UNSTABLE at a finite rate (turn the
focusing coupling g up through the instability onset over a quench time tau_Q). Near
the transition the uniform condensate's response freezes; the modulational instability
then imprints a pattern whose characteristic wavenumber k* is set by WHEN freeze-out
happened. Faster quench -> freeze deeper in the unstable regime (higher gain, shorter
wavelength) -> MORE defects; slower quench -> freeze just past threshold -> few, long-
wavelength defects. That is the Kibble-Zurek mechanism: k* ~ tau_Q^{-b}.

Implementation notes that make the scaling clean:
  * a LONG ring (large radius) so k* spans a decade of resolved wavenumbers;
  * a large g_f so the MI band sqrt(g) is wide;
  * the frozen wavenumber is read as the spectral peak WITHIN the physical MI band
    q <= 1.3*sqrt(g_f) (searching all q lets a high-q noise spike masquerade as k*);
  * measured at freeze-out (first time max|u| departs the uniform state), averaged
    over noise realizations.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"
L = 20 * np.pi
Nx = 2048
xg = np.linspace(0, L, Nx, endpoint=False)
q = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)           # physical wavenumbers
a0, g_f = 1.0, 40.0
dt = 2e-4
qband = 1.3 * np.sqrt(g_f)                             # search the MI band only
inband = (q > 0) & (q <= qband)
qpos = q[inband]


def kstar(tauQ, seed):
    rng = np.random.default_rng(seed)
    u = a0 + 3e-3 * (rng.standard_normal(Nx) + 1j * rng.standard_normal(Nx))
    Lh = np.exp(-1j * q ** 2 * dt / 2)
    n = int((tauQ + 8.0) / dt)
    for i in range(n):
        g = g_f * min(i * dt / tauQ, 1.0)
        u = np.fft.ifft(Lh * np.fft.fft(u))
        u = u * np.exp(1j * g * np.abs(u) ** 2 * dt)
        u = np.fft.ifft(Lh * np.fft.fft(u))
        if np.max(np.abs(u)) > 1.35 * a0:              # instability has taken hold
            drho = np.abs(u) ** 2 - np.mean(np.abs(u) ** 2)
            P = np.abs(np.fft.fft(drho)) ** 2
            return qpos[np.argmax(P[inband])], u
    return np.nan, u


tauQs = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
nseed = 16
kbar, kerr = [], []
for tq in tauQs:
    ks = np.array([kstar(tq, s)[0] for s in range(nseed)])
    ks = ks[np.isfinite(ks) & (ks > 0)]
    kbar.append(ks.mean()); kerr.append(ks.std() / np.sqrt(max(len(ks), 1)))
    print(f"tau_Q={tq:5.2f}  <k*>={kbar[-1]:.3f} +- {kerr[-1]:.3f}  (n={len(ks)})", flush=True)
kbar = np.array(kbar); kerr = np.array(kerr)

b, c = np.polyfit(np.log(tauQs), np.log(kbar), 1)
print(f"KZ exponent  k* ~ tau_Q^({b:.3f})", flush=True)

kf_fast, u_fast = kstar(0.05, 0)
kf_slow, u_slow = kstar(6.4, 0)

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
axA = ax[0]
axA.plot(xg, np.abs(u_fast) ** 2, color=PINK, lw=0.9, label=f"fast τ_Q=0.05  (k*={kf_fast:.1f})")
axA.plot(xg, np.abs(u_slow) ** 2, color=BLUE, lw=1.4, label=f"slow τ_Q=6.4  (k*={kf_slow:.1f})")
axA.set_xlabel(r"$\theta$ (long ring)"); axA.set_ylabel(r"$|u|^2$ at freeze-out")
axA.set_xlim(0, L)
axA.set_title("faster quench freezes in a finer pattern (more defects)")
axA.legend(fontsize=8.6)

axB = ax[1]
axB.errorbar(tauQs, kbar, yerr=kerr, fmt="o", color=GOLD, ms=6, capsize=3, label="⟨k*⟩ (16 seeds)")
axB.plot(tauQs, np.exp(c) * tauQs ** b, color="k", lw=1.6,
         label=rf"fit  $k^*\!\sim\tau_Q^{{{b:.2f}}}$")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel(r"quench time  $\tau_Q$"); axB.set_ylabel(r"frozen wavenumber  $k^*$")
axB.set_title("Kibble-Zurek power-law scaling")
axB.legend(fontsize=9); axB.grid(alpha=0.3, which="both")

fig.suptitle("Kibble-Zurek on the belly ring: a finite-rate quench freezes in defects, k* ~ τ_Q^{-b}",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/kibble_zurek.png", dpi=120)
print("wrote kibble_zurek.png", flush=True)
