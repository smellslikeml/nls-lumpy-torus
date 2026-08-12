"""
The classical geodesic flow -- the skeleton the quasimodes live on.

Geodesic flow on the surface of revolution is INTEGRABLE (Clairaut: L = A^2 dθ/ds
conserved). Unit-speed geodesics obey  (dx/ds)^2 = 1 - L^2/A(x)^2  -- 1-D motion
in the SAME effective potential L^2/A^2 that appears quantum-mechanically. Phase
portrait in (x, dx/ds):
  L in (A_min, A_max): librating orbits trapped near the belly  -> ELLIPTIC ISLAND
  L = A_min = 0.707   : the neck orbit                          -> SEPARATRIX
  L < A_min           : orbits circulating over the necks
The equatorial geodesic (x=0, L=1) is the elliptic fixed point; the neck geodesic
is hyperbolic. This is the classical mirror of the quantum well/barrier.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Amin, Amax = np.sqrt(0.5), 1.0


def A_of(x):
    return np.sqrt((1 + np.cos(x) ** 2) / 2)


def Ap_of(x):
    return -np.sin(2 * x) / (4 * A_of(x))


# profile z(x) for the crescent embedding
xg = np.linspace(-np.pi / 2, np.pi / 2, 3000)
g = np.sqrt(np.clip(1 - Ap_of(xg) ** 2, 0, 1))
zg = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(xg))])
Lz = zg[-1] + float(np.mean(np.diff(zg)))
z_interp = interp1d(xg, zg)
R = 2.2


def crescent_pt(x, th):
    xm = ((x + np.pi / 2) % np.pi) - np.pi / 2
    a = A_of(xm); Phi = 2 * np.pi * (z_interp(xm) - zg[0]) / Lz
    return ((R + a * np.cos(th)) * np.cos(Phi), (R + a * np.cos(th)) * np.sin(Phi), a * np.sin(th))


def geodesic(L, s_max=16.0):
    def rhs(s, y):
        x, xd, th = y
        a = A_of(x)
        return [xd, Ap_of(x) * L ** 2 / a ** 3, L / a ** 2]
    xd0 = np.sqrt(max(1 - L ** 2, 1e-9))
    sol = solve_ivp(rhs, [0, s_max], [0.0, xd0, 0.0], max_step=0.01, dense_output=True, rtol=1e-8)
    return sol.y[0], sol.y[2]


fig = plt.figure(figsize=(13, 5.3))

# ---- (1) phase portrait ----
ax = fig.add_subplot(1, 2, 1)
xx = np.linspace(-np.pi / 2, np.pi / 2, 600); AA = A_of(xx)
for L in np.linspace(0.73, 0.995, 9):                    # elliptic island (librating)
    val = 1 - L ** 2 / AA ** 2; m = val >= 0
    ax.plot(xx[m], np.sqrt(val[m]), color="#2b6cb0", lw=.9)
    ax.plot(xx[m], -np.sqrt(val[m]), color="#2b6cb0", lw=.9)
val = 1 - Amin ** 2 / AA ** 2                             # separatrix
ax.plot(xx, np.sqrt(np.clip(val, 0, None)), "k", lw=2)
ax.plot(xx, -np.sqrt(np.clip(val, 0, None)), "k", lw=2)
for L in np.linspace(0.15, 0.66, 5):                     # circulating over the necks
    val = 1 - L ** 2 / AA ** 2
    ax.plot(xx, np.sqrt(val), color="#b83280", lw=.9)
    ax.plot(xx, -np.sqrt(val), color="#b83280", lw=.9)
ax.plot(0, 0, "o", color="#2b6cb0", ms=7)
for xv in (-np.pi / 2, np.pi / 2):
    ax.plot(xv, 0, "x", color="k", ms=8, mew=2)
ax.text(0, 0.05, " elliptic (belly)", color="#2b6cb0", fontsize=9)
ax.text(np.pi / 2, 0.05, "hyperbolic\n(neck) ", color="k", ha="right", fontsize=9)
ax.text(0, 0.9, "circulating over necks", color="#b83280", ha="center", fontsize=9)
ax.text(0, 0.42, "librating island", color="#2b6cb0", ha="center", fontsize=9)
ax.set_xlim(-np.pi / 2, np.pi / 2); ax.set_ylim(-1.05, 1.05)
ax.set_xticks([-np.pi / 2, 0, np.pi / 2]); ax.set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
ax.set_xlabel("x"); ax.set_ylabel(r"$dx/ds$"); ax.set_title("geodesic phase portrait (integrable)")

# ---- (2) two geodesics on the crescent ----
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
# faint crescent surface
xs = np.linspace(-np.pi / 2, np.pi / 2, 160); ts = np.linspace(0, 2 * np.pi, 80)
Xg, Tg = np.meshgrid(xs, ts, indexing="ij")
Sx, Sy, Sz = crescent_pt(Xg, Tg)
ax2.plot_surface(Sx, Sy, Sz, color="0.85", rstride=2, cstride=2, linewidth=0,
                 antialiased=False, shade=True, alpha=0.25)
for L, col, lab in [(0.9, "#2b6cb0", "librating (near belly)"), (0.45, "#b83280", "circulating (over necks)")]:
    gx, gt = geodesic(L)
    px, py, pz = crescent_pt(gx, gt)
    ax2.plot(px, py, pz, color=col, lw=1.6, label=lab)
ax2.set_box_aspect((1, 1, 0.5)); ax2.set_axis_off(); ax2.view_init(elev=42, azim=-55)
ax2.legend(loc="upper center", fontsize=9); ax2.set_title("geodesics on the crescent")

fig.suptitle("Classical geodesic flow: an elliptic island around the belly, a separatrix at the neck",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/geodesic_phase_space.png", dpi=120)
print("wrote geodesic_phase_space.png")
