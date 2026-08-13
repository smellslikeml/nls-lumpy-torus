"""
Frontier prototype 2: the neck as a de Laval nozzle -> an analog black hole.

A superfluid (Gross-Pitaevskii) flowing along the tube must speed up through the
small-A neck (continuity: rho*v*A = const). Where the flow speed v crosses the
sound speed c = sqrt(g*rho), sound can no longer travel upstream -- a SONIC
HORIZON, provided by the geometry alone (the throat), with an analog Hawking
temperature T_H = |d(v-c)/dx|_horizon / (2 pi). We solve the stationary transonic
flow (continuity + Bernoulli) and trace the sound characteristics: the v-c rays
cannot escape upstream past the throat -- the horizon of an analog black hole.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
mu, g = 1.0, 1.0
x = np.linspace(-np.pi / 2, np.pi / 2, 800)
A = np.sqrt((1 + np.sin(x) ** 2) / 2)             # de Laval nozzle: throat (min A) at x=0
Amin = A.min()
vs = np.sqrt(2 * mu / 3); rhos = vs ** 2; J = rhos * vs * Amin     # transonic flux

v = np.empty_like(x)
for i, xi in enumerate(x):
    roots = np.roots([0.5, 0, -mu, J / A[i]])     # 1/2 v^3 - mu v + J/A = 0
    pos = np.sort(roots[np.abs(roots.imag) < 1e-6].real)
    pos = pos[pos > 0]
    v[i] = pos[0] if xi < 0 else pos[-1]           # subsonic upstream, supersonic downstream
rho = J / (v * A); c = np.sqrt(g * rho); M = v / c
# surface gravity = |d(v-c)/dx| at the horizon, from a local linear fit that
# excludes the sonic-point branch-selection kink (a single-point gradient there
# under-reads it); robust to the fit window at kappa ~ 0.71.
_m = (np.abs(x) > 0.02) & (np.abs(x) < 0.15)
kappa = abs(np.polyfit(x[_m], (v - c)[_m], 1)[0])
T_H = kappa / (2 * np.pi)

# ---- sound characteristics: dx/dt = v -/+ c ----
vf, cf = interp1d(x, v, fill_value="extrapolate"), interp1d(x, c, fill_value="extrapolate")
def rays(sign, x0s, T=6.0):
    out = []
    for x0 in x0s:
        s = solve_ivp(lambda t, y: vf(y) + sign * cf(y), [0, T], [x0], max_step=0.02,
                      dense_output=True, rtol=1e-7)
        out.append((s.t, s.y[0]))
    return out

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# panel 1: v and c cross at the throat
ax[0].fill_between(x, -0.1, 1.2, where=(x > 0), color="#f7e6ef", alpha=.6)
ax[0].plot(x, v, color="#b83280", lw=2, label="flow speed v")
ax[0].plot(x, c, color="#2b6cb0", lw=2, label="sound speed c")
ax[0].axvline(0, color="k", ls="--", lw=1)
ax[0].text(0.05, 1.12, "  horizon\n  (throat)", fontsize=9)
ax[0].text(0.55, 0.15, "supersonic\n(interior)", ha="center", fontsize=9, color="#b83280")
ax[0].text(-0.9, 0.15, "subsonic\n(exterior)", ha="center", fontsize=9, color="#2b6cb0")
ax[0].set_xlim(-np.pi / 2, np.pi / 2); ax[0].set_ylim(0, 1.2)
ax[0].set_xticks([-np.pi / 2, 0, np.pi / 2]); ax[0].set_xticklabels([r"$-\pi/2$", "neck", r"$\pi/2$"])
ax[0].set_xlabel("x (along the tube)"); ax[0].set_ylabel("speed")
ax[0].set_title(f"sonic horizon at the neck   ·   $T_H$ = {T_H:.3f}")
ax[0].legend(loc="upper left")
# panel 2: characteristics -> the horizon traps the upstream (v-c) rays
for t, xr in rays(-1, np.linspace(-1.3, 1.3, 15)):
    ax[1].plot(t, xr, color="#b83280", lw=.9)
for t, xr in rays(+1, np.linspace(-1.3, -0.2, 4)):
    ax[1].plot(t, xr, color="#2b6cb0", lw=.7, alpha=.5)
ax[1].axhline(0, color="k", ls="--", lw=1)
ax[1].text(4.2, 0.04, "horizon", fontsize=9)
ax[1].set_ylim(-np.pi / 2, np.pi / 2); ax[1].set_xlabel("t")
ax[1].set_yticks([-np.pi / 2, 0, np.pi / 2]); ax[1].set_yticklabels([r"$-\pi/2$", "neck", r"$\pi/2$"])
ax[1].set_ylabel("x")
ax[1].set_title("upstream (v−c) sound rays cannot cross the throat")
fig.suptitle("The neck as an analog black hole: a geometry-provided sonic horizon", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/analog_horizon.png", dpi=120)
print(f"wrote analog_horizon.png  surface gravity kappa={kappa:.3f}, T_H={T_H:.3f}")
