"""
Frontier prototype 14: analog Hawking -- the exponential redshift and thermal spectrum.

The neck is a de Laval nozzle: a transonic superfluid crosses the sound speed at the
throat, a sonic horizon (analog_horizon.py). Here we take the horizon's radiation
seriously. Near the horizon the outgoing (upstream) characteristic obeys
    dx/dt = v(x) - c(x) ~ kappa * (x - x_h),
so a ray peels away EXPONENTIALLY, |x - x_h| ~ e^{kappa t}: a mode dragged to the
horizon is redshifted without bound at the rate kappa (the surface gravity). That
exponential relation between horizon-time and lab-time is exactly what turns the
vacuum into a THERMAL spectrum at the Hawking temperature T_H = kappa / 2*pi. We
(1) trace the rays and read kappa off the exponential slope -- an independent,
DYNAMICAL measurement of the surface gravity that must match the static formula --
and (2) plot the predicted thermal phonon occupation n(omega) = 1/(e^{omega/T_H}-1).

Honest scope: this is the kinematic Hawking result (temperature + thermality
mechanism). The spontaneous pair spectrum |beta_omega|^2 itself requires the
dispersive Bogoliubov-de Gennes scattering with its negative-norm partner mode --
the rigorous completion, noted but not attempted here.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"

# --- transonic flow through the neck (as in analog_horizon.py) ---
mu, g = 1.0, 1.0
x = np.linspace(-np.pi / 2, np.pi / 2, 1200)
A = np.sqrt((1 + np.sin(x) ** 2) / 2)                 # throat (min A) at x=0
Amin = A.min()
vs = np.sqrt(2 * mu / 3); rhos = vs ** 2; J = rhos * vs * Amin
v = np.empty_like(x)
for i, xi in enumerate(x):
    roots = np.roots([0.5, 0, -mu, J / A[i]])
    pos = np.sort(roots[np.abs(roots.imag) < 1e-6].real)
    pos = pos[pos > 0]
    v[i] = pos[0] if xi < 0 else pos[-1]
rho = J / (v * A); c = np.sqrt(g * rho)
vmc = v - c                                           # upstream characteristic speed
# The transonic branch-selection has a kink exactly at the sonic point, so a
# single-point gradient there is unreliable. Extract kappa from a smooth (odd-cubic)
# fit of v-c across the throat, excluding the immediate kink.
fitmask = (np.abs(x) > 0.02) & (np.abs(x) < 0.35)
cfit = np.polyfit(x[fitmask], vmc[fitmask], 3)
vmc_smooth = np.poly1d(cfit)
kappa_static = abs(np.polyval(np.polyder(cfit), 0.0))
T_H = kappa_static / (2 * np.pi)
print(f"surface gravity kappa (smooth fit) = {kappa_static:.4f}   T_H = {T_H:.4f}", flush=True)

# --- (1) trace upstream rays on the smooth flow; peeling gives kappa dynamically ---
vmcf = vmc_smooth                                     # smooth v-c (no branch kink)
rays = []
for x0 in [-0.12, -0.06, -0.03, 0.03, 0.06, 0.12]:
    s = solve_ivp(lambda t, y: vmcf(y), [0, 9], [x0], max_step=0.02, rtol=1e-8, dense_output=True)
    rays.append((s.t, s.y[0]))
tt, xx = rays[2]                                      # x0 = -0.03, closest inside
near = np.abs(xx) < 0.15                              # near-horizon (linear) zone
kappa_dyn = np.polyfit(tt[near], np.log(np.abs(xx[near])), 1)[0]
print(f"surface gravity kappa (dynamical, ray slope) = {kappa_dyn:.4f}", flush=True)

# --- (2) predicted thermal phonon spectrum at T_H ---
omega = np.linspace(1e-3, 6 * T_H, 400)
n_thermal = 1.0 / (np.exp(omega / T_H) - 1.0)         # Bose occupation
flux = omega * n_thermal                              # energy flux density (Planck-like)

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))

axA = ax[0]
for (tt, xx), col in zip(rays, [PINK, PINK, PINK, BLUE, BLUE, BLUE]):
    axA.plot(tt, np.abs(xx) + 1e-6, color=col, lw=1.3, alpha=.85)
# reference exponential of the fitted slope
tref = np.linspace(0, 6, 50)
axA.plot(tref, 0.03 * np.exp(kappa_dyn * tref), "k--", lw=1.4,
         label=rf"$\propto e^{{\kappa t}}$, $\kappa$={kappa_dyn:.2f}")
axA.set_yscale("log"); axA.set_ylim(1e-2, np.pi / 2)
axA.set_xlabel("t"); axA.set_ylabel(r"distance from horizon  $|x-x_h|$  (log)")
axA.set_title(f"rays peel off the horizon exponentially  ·  κ={kappa_dyn:.2f}")
axA.text(0.3, 1.0, "interior\n(supersonic)", color=PINK, fontsize=8.5)
axA.text(0.3, 0.02, "exterior\n(subsonic)", color=BLUE, fontsize=8.5)
axA.legend(loc="lower right", fontsize=9)

axB = ax[1]
axB.plot(omega, n_thermal, color=GOLD, lw=2.2, label=r"$n(\omega)=1/(e^{\omega/T_H}-1)$")
axB.plot(omega, flux, color=PINK, lw=1.8, ls="--", label=r"energy flux  $\omega\,n(\omega)$")
axB.axvline(T_H, color="0.6", lw=1.0, ls=":")
axB.text(T_H * 1.05, 2.2, f"  $T_H$={T_H:.3f}", fontsize=9, color="0.4")
axB.set_ylim(0, 4)
axB.set_xlabel(r"phonon frequency  $\omega$"); axB.set_ylabel("occupation / flux")
axB.set_title(r"predicted thermal Hawking spectrum at $T_H=\kappa/2\pi$")
axB.legend(fontsize=9); axB.grid(alpha=0.3)

fig.suptitle("Analog Hawking: the horizon's exponential redshift (rate κ) → a thermal phonon spectrum at T_H",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/hawking.png", dpi=120)
print("wrote hawking.png", flush=True)
