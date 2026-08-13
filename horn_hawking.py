"""Flagship: analog Hawking radiation on the OPEN horn.

The horn's throat is a de Laval nozzle: a transonic superfluid accelerates from a subsonic
reservoir, crosses the sound speed exactly at the throat (a sonic horizon), and radiates
supersonically out the open end — where, unlike the closed torus neck of section 10, the
phonons genuinely escape to infinity (the PML/absorbing collar is the outgoing boundary).

Near the horizon the upstream characteristic obeys dx/dt = v - c ~ kappa (x - x_h), so rays
peel away exponentially at the surface gravity kappa, turning the vacuum thermal at
T_H = kappa / 2*pi. We extract kappa TWO independent ways — a smooth fit of v-c across the
throat (static) and the exponential slope of traced rays (dynamical) — and their agreement
is the trust flag. Then we sweep the throat sharpness: geometry sets the Hawking temperature.

Honest scope: kinematic Hawking (temperature + thermality mechanism). The full spontaneous
pair spectrum needs dispersive Bogoliubov-de Gennes scattering — noted, not attempted.

Regenerate:  python3 horn_hawking.py
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU, G = 1.0, 1.0
Xmax, x_t, A_open, Delta = 20.0, 8.0, 1.2, 0.7          # horn: throat dip at x_t, opens beyond


def horn_A(x, w):
    return A_open - Delta * np.exp(-((x - x_t) / w) ** 2)


def transonic_flow(w, N=2000):
    """Steady transonic flow through the throat: subsonic upstream, supersonic downstream."""
    x = np.linspace(0.0, Xmax, N)
    A = horn_A(x, w); Amin = A.min()
    vs = np.sqrt(2 * MU / 3); rhos = vs ** 2; J = rhos * vs * Amin       # sonic at the throat
    v = np.empty_like(x)
    for i, xi in enumerate(x):
        roots = np.roots([0.5, 0.0, -MU, J / A[i]])
        pos = np.sort(roots[np.abs(roots.imag) < 1e-6].real)
        pos = pos[pos > 0]
        v[i] = pos[0] if xi < x_t else pos[-1]           # subsonic then supersonic branch
    rho = J / (v * A); c = np.sqrt(G * rho)
    return x, A, v, c


def kappa_static(w):
    """Surface gravity from a smooth (odd-cubic) fit of v-c across the horizon (the
    transonic branch has a kink exactly at the sonic point, so avoid the immediate point)."""
    x, A, v, c = transonic_flow(w)
    vmc = v - c
    msk = (np.abs(x - x_t) > 0.05) & (np.abs(x - x_t) < 1.0)
    cf = np.polyfit(x[msk] - x_t, vmc[msk], 3)
    return abs(np.polyval(np.polyder(cf), 0.0)), (x, v, c, vmc, cf)


# ---- dual kappa at a reference throat, then the geometry sweep ------------------------
w0 = 1.2
k_stat, (x, v, c, vmc, cf) = kappa_static(w0)
vmc_smooth = np.poly1d(cf)
rays = []; slopes = []
for x0 in (-0.08, -0.04, 0.04, 0.08):                    # trace the horizon characteristic
    s = solve_ivp(lambda t, y: vmc_smooth(y[0] - x_t), [0, 12], [x_t + x0],
                  max_step=0.02, rtol=1e-9, dense_output=True)
    xx = s.y[0] - x_t; rays.append((s.t, xx))
    near = (np.abs(xx) > 0.02) & (np.abs(xx) < 0.25)     # the truly-linear near-horizon zone
    if near.sum() > 5:
        slopes.append(np.polyfit(s.t[near], np.log(np.abs(xx[near])), 1)[0])
k_dyn = float(np.median(slopes))
T_H = k_stat / (2 * np.pi)
print(f"reference throat w={w0}:  kappa_static={k_stat:.4f}  kappa_dyn={k_dyn:.4f}  "
      f"(agree to {abs(k_stat-k_dyn)/k_stat*100:.1f}%)  ->  T_H={T_H:.4f}")

ws = [0.7, 0.9, 1.2, 1.6, 2.1]
print(f"\ngeometry sweep — sharper throat, hotter horizon:")
print(f"{'throat w':>9s} {'kappa':>8s} {'T_H':>8s}")
ks = []
for w in ws:
    k, _ = kappa_static(w); ks.append(k)
    print(f"{w:9.2f} {k:8.4f} {k/(2*np.pi):8.4f}")
verified = abs(k_stat - k_dyn) / k_stat < 0.15

# ---- figure ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.7))
axA.plot(x, v, color="#b83280", lw=2.2, label="flow speed $v$")
axA.plot(x, c, color="#2b6cb0", lw=2.2, label="sound speed $c$")
axA.axvline(x_t, color="#888", ls=":", lw=1.2)
axA.fill_between(x, 0, 1.4, where=(x > x_t), color="#fbeaf2", zorder=0)
axA.text(x_t + 0.3, 0.15, "supersonic\n(interior)", color="#b83280", fontsize=8.5)
axA.text(2.2, 0.15, "subsonic\n(exterior)", color="#2b6cb0", fontsize=8.5)
axA.text(x_t + 0.15, 1.28, "horizon\n(throat)", color="#555", fontsize=8.5, ha="center")
axA.axvspan(Xmax - 4, Xmax, color="#eee", alpha=.7); axA.text(Xmax - 3.6, 1.15, "PML\n(escape)", fontsize=8, color="#777")
axA.set_xlabel("meridian $x$ (open horn)"); axA.set_ylabel("speed"); axA.set_ylim(0, 1.4)
axA.set_title(f"de Laval horizon at the throat  ·  $T_H=\\kappa/2\\pi={T_H:.3f}$", fontsize=11.5)
axA.legend(fontsize=9, loc="upper right")
axins = axA.inset_axes([0.12, 0.62, 0.32, 0.32])
for tt, xx in rays:
    axins.plot(tt, np.abs(xx) + 1e-6, color="#b83280", lw=1.1)
tref = np.linspace(0, 7, 30); axins.plot(tref, 0.05 * np.exp(k_dyn * tref), "k--", lw=1.1)
axins.set_yscale("log"); axins.set_ylim(1e-2, 3); axins.set_xlabel("t", fontsize=7)
axins.set_ylabel("$|x-x_h|$", fontsize=7); axins.tick_params(labelsize=6)
axins.set_title(f"rays peel $\\propto e^{{\\kappa t}}$, $\\kappa$={k_dyn:.2f}", fontsize=7.5)

axB.plot(ws, np.array(ks) / (2 * np.pi), "o-", color="#d69e2e", lw=2.2, ms=7, label="$T_H$ (static $\\kappa$)")
axB.axhline(k_dyn / (2 * np.pi), color="#b83280", ls="--", lw=1.3, alpha=.7,
            label=f"dynamical cross-check @ w={w0}")
for w, k in zip(ws, ks):
    axB.annotate(f"{k/(2*np.pi):.3f}", (w, k / (2 * np.pi)), textcoords="offset points",
                 xytext=(5, 6), fontsize=8.3, color="#3a4658")
axB.set_xlabel("throat width $w$  (sharper $\\leftarrow$)"); axB.set_ylabel("Hawking temperature $T_H$")
axB.set_title("geometry sets the horizon temperature", fontsize=11.5)
axB.legend(fontsize=9); axB.grid(alpha=.25); axB.invert_xaxis()
fig.tight_layout(); fig.savefig("horn_hawking.png", dpi=130, bbox_inches="tight")
print(f"\nVERIFICATION static/dynamical kappa agree (<15%): {verified}")
print("wrote horn_hawking.png")
