"""
Elliptic <-> hyperbolic bifurcation: the geometry knob.

Tunable lump family  A(x;eps) = sqrt((1 + eps cos^2 x)/(1+eps)),  eps in (-1, inf),
with A(belly)=1 fixed. The curvatures are exactly
    K_belly(eps) = eps/(1+eps),     K_neck(eps) = -eps ,
so the belly parallel is elliptic for eps>0 and hyperbolic for eps<0; the belly
and neck geodesics EXCHANGE stability at eps=0 (the flat cylinder). Our lumpy
torus is eps=1 (K_belly=+1/2, K_neck=-1). Quantum-mechanically the effective
potential V_k=k^2/A^2 flips from a well at the belly to a barrier there.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"


def A_eps(x, eps):
    return np.sqrt((1 + eps * np.cos(x) ** 2) / (1 + eps))


Nx = 500
x = np.linspace(-np.pi / 2, np.pi / 2, Nx, endpoint=False)
dx = x[1] - x[0]


def nbound(eps, k=6):                       # # belly-bound states of V_k for this eps
    A = A_eps(x, eps)
    Aface = 0.5 * (A + np.roll(A, -1)); Am = np.roll(Aface, 1)
    r = np.r_[np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx]
    c = np.r_[np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)]
    v = np.r_[(Am + Aface) / dx, -Aface / dx, -Aface / dx]
    Kx = sp.coo_matrix((v, (r, c)), shape=(Nx, Nx)).toarray()
    S = Kx + np.diag(k ** 2 / A * dx); M = np.diag(A * dx)
    E = eigh(S, M, eigvals_only=True)
    Vmin, Vmax = k ** 2 / A.max() ** 2, k ** 2 / A.min() ** 2
    return int(np.sum((E > Vmin - 1e-6) & (E < Vmax)))    # trapped below the barrier


fig = plt.figure(figsize=(13, 4.6))

# ---- (1) stability exchange ----
ax = fig.add_subplot(1, 3, 1)
eps = np.linspace(-0.9, 4, 400)
ax.plot(eps, eps / (1 + eps), color="#2b6cb0", lw=2, label=r"$K_{\rm belly}=\epsilon/(1+\epsilon)$")
ax.plot(eps, -eps, color="#b83280", lw=2, label=r"$K_{\rm neck}=-\epsilon$")
ax.axhline(0, color="0.6", lw=.8); ax.axvline(0, color="0.6", lw=.8, ls=":")
ax.plot(1, 0.5, "o", color="#2b6cb0"); ax.plot(1, -1, "o", color="#b83280")
ax.text(1.05, 0.55, "our torus\n($\\epsilon$=1)", fontsize=8)
ax.text(-0.85, 1.2, "belly\nhyperbolic", color="#2b6cb0", fontsize=8)
ax.text(2.2, 1.1, "belly elliptic", color="#2b6cb0", fontsize=8)
ax.set_ylim(-2, 2.2); ax.set_xlabel(r"lump amplitude $\epsilon$"); ax.set_ylabel("Gaussian curvature")
ax.set_title("belly & neck exchange stability at $\\epsilon=0$"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)

# ---- (2) effective potential flips ----
ax2 = fig.add_subplot(1, 3, 2)
k = 6
for eps, lab, col in [(1.0, r"$\epsilon=1$ (well at belly)", "#2b6cb0"),
                      (0.0001, r"$\epsilon\to0$ (flat)", "0.5"),
                      (-0.5, r"$\epsilon=-0.5$ (barrier at belly)", "#b83280")]:
    ax2.plot(x, k ** 2 / A_eps(x, eps) ** 2, color=col, lw=2, label=lab)
ax2.set_xticks([-np.pi / 2, 0, np.pi / 2]); ax2.set_xticklabels([r"$-\pi/2$", "belly", r"$\pi/2$"])
ax2.set_xlabel("x"); ax2.set_ylabel(r"$V_k=k^2/A^2$")
ax2.set_title(r"the centrifugal potential inverts ($k=6$)"); ax2.legend(fontsize=8); ax2.grid(alpha=.3)

# ---- (3) trapped-mode count vanishes at the bifurcation ----
ax3 = fig.add_subplot(1, 3, 3)
epss = np.linspace(0.02, 4, 40)
nb = [nbound(e) for e in epss]
ax3.plot(epss, nb, "o-", color="#2b6cb0", ms=3)
ax3.axvline(0, color="0.6", ls=":")
ax3.set_xlabel(r"lump amplitude $\epsilon$"); ax3.set_ylabel("# whispering-gallery bound states (k=6)")
ax3.set_title("trapping switches on as the belly becomes elliptic"); ax3.grid(alpha=.3)

fig.suptitle("Elliptic$\\leftrightarrow$hyperbolic bifurcation of the equatorial geodesic", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/bifurcation.png", dpi=120)
print("wrote bifurcation.png")
print(f"  bound states (k=6): eps=0.1 -> {nbound(0.1)}, eps=1 -> {nbound(1)}, eps=3 -> {nbound(3)}")
