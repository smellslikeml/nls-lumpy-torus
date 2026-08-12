"""
Frontier prototype 1: geometric dispersion engineering of a WG resonator.

A microtoroid's soliton-comb performance is set by its modal dispersion
D_int(m) = omega_m - (omega_0 + D1 m). Here omega_m = E0(m), the ground transverse
state of the reduced operator H_m = -(1/A) d(A d) + m^2/A^2 -- so the lump profile
A(x) IS the dispersion knob. We inverse-design A(x) (a few harmonics) to FLATTEN
D_int over a mode band (a broadband, low-dispersion resonator), by minimising the
mode-frequency departure from an equidistant comb grid.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Nx = 220
x = np.linspace(-np.pi / 2, np.pi / 2, Nx, endpoint=False)
dx = x[1] - x[0]
harm = np.array([np.cos(2 * x), np.cos(4 * x), np.cos(6 * x)])   # design harmonics (period pi)
ms = np.arange(8, 29)                                             # mode band
m0 = ms[len(ms) // 2]


def A_of(a):
    A = 1.0 + a @ harm
    return np.clip(A, 0.35, None)


def E0(m, A):
    Aface = 0.5 * (A + np.roll(A, -1)); Am = np.roll(Aface, 1)
    r = np.r_[np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx]
    c = np.r_[np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)]
    v = np.r_[(Am + Aface) / dx, -Aface / dx, -Aface / dx]
    S = sp.coo_matrix((v, (r, c)), shape=(Nx, Nx)).toarray() + np.diag(m ** 2 / A * dx)
    return eigh(S, np.diag(A * dx), eigvals_only=True, subset_by_index=[0, 0])[0]


def dint(A):
    E = np.array([E0(m, A) for m in ms])
    # remove the equidistant (linear) part -> integrated dispersion
    c1, c0 = np.polyfit(ms - m0, E, 1)
    return E - (c0 + c1 * (ms - m0))


def loss(a):
    A = A_of(a)
    pen = 1e3 * np.sum(np.clip(0.35 - (1 + a @ harm), 0, None) ** 2)   # keep A>0.35
    return np.sum(dint(A) ** 2) + pen


base = A_of(np.zeros(3))                    # flat cylinder A=1 (reference)
res = minimize(loss, np.zeros(3), method="Nelder-Mead",
               options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
Aopt = A_of(res.x)
Db, Do = dint(base), dint(Aopt)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
ax[0].plot(ms, Db, "o-", color="0.5", label=f"flat cylinder (spread {np.ptp(Db):.1f})")
ax[0].plot(ms, Do, "o-", color="#2b6cb0", label=f"engineered A(x) (spread {np.ptp(Do):.2f})")
ax[0].axhline(0, color="0.7", lw=.8)
ax[0].set_xlabel("mode number m"); ax[0].set_ylabel(r"integrated dispersion $D_{\rm int}(m)$")
ax[0].set_title(f"dispersion flattened {np.ptp(Db)/max(np.ptp(Do),1e-9):.0f}× over the band")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(x, base, "0.5", lw=1.5, label="flat A=1")
ax[1].plot(x, Aopt, "#b83280", lw=2, label="inverse-designed A(x)")
ax[1].set_xticks([-np.pi / 2, 0, np.pi / 2]); ax[1].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
ax[1].set_xlabel("x"); ax[1].set_ylabel("A(x)"); ax[1].set_title("the profile that does it")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Geometric dispersion engineering: inverse-design the lump profile for a flat WG comb grid",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/dispersion_design.png", dpi=120)
print(f"wrote dispersion_design.png  spread {np.ptp(Db):.2f} -> {np.ptp(Do):.3f}  (a={np.round(res.x,3)})")
