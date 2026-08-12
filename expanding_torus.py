"""
Frontier prototype 3: an EXPANDING torus -> cosmological particle production.

Quench the ring radius (scale factor) a(t) upward and the phonon modes on the belly
redshift, omega_k(t) = c k / a(t): the adiabatic vacuum can no longer follow, and
the expansion pumps phonon PAIRS out of nothing -- the acoustic analogue of particle
production by an expanding universe. This is the NLS/BEC realisation demonstrated by
Viermann et al., "Quantum field simulator for dynamics in curved spacetime",
Nature 611, 260 (2022): a 2-D BEC whose interaction/trap is ramped to mimic an
expanding FRW spacetime.

Protocol: a fast (near-sudden) expansion at t_q, then free evolution. Each mode ends
in a two-mode squeezed vacuum with |beta_k|^2 produced pairs; while it evolves, the
squeezed quadratures rotate, so the equal-time density structure factor
    S_k(t) = <|delta rho_k|^2>  = (|alpha_k| - |beta_k|)^2 ... (|alpha_k| + |beta_k|)^2
oscillates in k at fixed t -- SAKHAROV OSCILLATIONS (the acoustic analogue of the CMB
acoustic peaks), riding on the k-independent |beta_k|^2 plateau.

Solved exactly by linear Bogoliubov theory (the massless-scalar-on-FRW model, the
purest cosmology analogue, and exactly the theory Viermann et al. compare data to):
each comoving mode is a parametric oscillator

        v_k'' + omega_k(t)^2 v_k = 0,   omega_k = c k / a(t),

started in its adiabatic (WKB) vacuum and integrated (RK4) through the quench, with
the two-mode-squeeze Wronskian |v v'* - v* v'| = 1 held to machine precision. The
produced pairs |beta_k|^2 and the observed density structure factor
S_k(t_obs) = 2 omega_f |v_k(t_obs)|^2 (=1 in vacuum) are read straight off v_k, v_k'.
(The companion direct-GPE demonstration of parametric production from noise is
faraday.py, the periodic-drive member of this same time-dependent-geometry theme.)

Signatures:
  * small k : the quench is sudden -> plateau |beta|^2 -> (a_f-a_i)^2/(4 a_i a_f),
              a SCALE-INVARIANT spectrum (the inflationary analogue);
  * S_k(t) oscillates about 1+2|beta|^2  ->  SAKHAROV oscillations;
  * large k : the quench looks adiabatic -> production shuts off, S_k -> 1 (vacuum).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"

# ------------------------------------------------- FRW scale factor: a fast expansion
c = 1.0
a_i, a_f = 1.0, 3.0                       # scale factor triples at the quench
t_q, tau_c = 1.0, 0.10                    # quench time and sharpness (small -> sudden)
T, Nt = 3.0, 30000                        # observe the structure factor at t_obs = T
tg = np.linspace(0.0, T, Nt)
dt = tg[1] - tg[0]
t_obs = T


def scale_factor(t):
    z = (t - t_q) / tau_c
    s = 0.5 * (1.0 + np.tanh(z))
    a = a_i + (a_f - a_i) * s
    adot = (a_f - a_i) * 0.5 / tau_c * (1.0 / np.cosh(z)) ** 2
    return a, adot


a_t, ad_t = scale_factor(tg)

# --------------------------------------------------- (1) linear mode integration (RK4)
kmax = 34
ks = np.arange(1, kmax + 1)               # comoving integer momenta on the ring
omega_i = c * ks / a_i                     # in  frequencies (static, t<t_q)
omega_f = c * ks / a_f                     # out frequencies (static, t>t_q)

# adiabatic-vacuum initial data:  v = 1/sqrt(2w),  v' = -i sqrt(w/2)
v = (1.0 / np.sqrt(2.0 * omega_i)).astype(complex)
w = (-1j * np.sqrt(omega_i / 2.0)).astype(complex)


def Omega2(t):
    a, _ = scale_factor(t)
    return (c * ks / a) ** 2


def deriv(vv, ww, t):
    return ww, -Omega2(t) * vv


for i in range(Nt - 1):
    t = tg[i]
    k1v, k1w = deriv(v, w, t)
    k2v, k2w = deriv(v + dt / 2 * k1v, w + dt / 2 * k1w, t + dt / 2)
    k3v, k3w = deriv(v + dt / 2 * k2v, w + dt / 2 * k2w, t + dt / 2)
    k4v, k4w = deriv(v + dt * k3v, w + dt * k3w, t + dt)
    v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)

# produced pairs and the observed density (field-quadrature) structure factor
N_k = np.abs(w) ** 2 / (2 * omega_f) + omega_f / 2 * np.abs(v) ** 2 - 0.5   # = |beta_k|^2
N_k = np.maximum(N_k, 0.0)
S_k = 2.0 * omega_f * np.abs(v) ** 2                                        # =1 in vacuum
wronski = np.max(np.abs(np.abs(v * np.conj(w) - np.conj(v) * w) - 1.0))     # |v v'* - v* v'| = 1
plateau = (a_f - a_i) ** 2 / (4 * a_i * a_f)                                # sudden, k-indep.

print(f"[linear]  Wronskian residual (should be ~0): {wronski:.1e}")
print(f"[linear]  sudden plateau |beta|^2 = {plateau:.4f}")
print(f"[linear]  |beta|^2: k=1 {N_k[0]:.3f}  k=8 {N_k[7]:.3f}  k={kmax} {N_k[-1]:.2e}")
print(f"[linear]  S_k(t_obs): min {S_k.min():.3f}  max {S_k.max():.3f}  "
      f"(vacuum 1.0; oscillates about 1+2|beta|^2={1+2*plateau:.2f})")

# ------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9))

# panel A: the protocol -- a fast expansion redshifts the modes
axA = ax[0]
axA2 = axA.twinx()
axA2.plot(tg, a_t, color="k", lw=2.3, label="scale factor a(t)")
axA2.set_ylabel("scale factor  a(t)")
axA2.set_ylim(a_i - 0.25, a_f + 0.25)
axA.axvline(t_q, color="0.6", lw=1.0, ls=":")
axA.text(t_q + 0.08, 0.3, "quench", fontsize=8.5, color="0.4")
for k, col in zip((3, 8, 20), (PINK, BLUE, GOLD)):
    axA.plot(tg, c * k / a_t, color=col, lw=1.8, label=f"$\\omega_k$, k={k}")
axA.set_yscale("log")
axA.set_xlabel("t")
axA.set_ylabel(r"frequency   $\omega_k = ck/a(t)$   (log)")
axA.set_xlim(0, T); axA.set_ylim(0.25, 45)
axA.set_title("expansion redshifts the modes  ·  vacuum can't follow")
h1, l1 = axA.get_legend_handles_labels()
h2, l2 = axA2.get_legend_handles_labels()
axA.legend(h2 + h1, l2 + l1, loc="upper right", fontsize=8.4, framealpha=.92)

# panel B: observed structure factor -- Sakharov oscillations about the plateau
axB = ax[1]
axB.axhline(1.0, color="0.6", ls=":", lw=1.2, label="vacuum (shot noise)")
axB.plot(ks, 1 + 2 * N_k, color="0.5", ls="--", lw=1.3,
         label=r"oscillation centre  $1+2|\beta_k|^2$")
axB.plot(ks, S_k, "-o", color=PINK, ms=4.6, lw=1.7,
         label=r"Bogoliubov  $S_k(t_{\rm obs})$")
axB.set_xlabel("comoving mode number  k")
axB.set_ylabel(r"structure factor  $S_k / S_k^{\rm vac}$")
axB.set_xlim(0, kmax + 1)
axB.set_ylim(0, max(S_k.max() * 1.12, 3.2))
axB.set_title("Sakharov oscillations in the structure factor")
axB.grid(alpha=0.25)
axB.legend(loc="upper right", fontsize=8.1, framealpha=.92, ncol=1)

fig.suptitle("Expanding torus → cosmological particle production "
             "(NLS analogue of Viermann et al., Nature 2022)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/expanding_torus.png", dpi=120)
print("wrote expanding_torus.png")
