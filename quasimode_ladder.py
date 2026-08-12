"""
The centrifugal-barrier reduction that grounds the whole experiment.

Separating u = e^{ikθ} φ(x) turns the Laplace-Beltrami operator into a 1-D
Schrodinger operator, self-adjoint in the metric measure A dx:
    H_k φ = -(1/A) d_x(A d_x φ) + V_k(x) φ ,   V_k(x) = k^2 / A(x)^2 .
V_k is a WELL at the belly (A=1 -> V=k^2) with BARRIERS at the necks
(A=0.707 -> V=2k^2). Its bound states are the whispering-gallery / equatorial
quasimodes; above the barrier the states circulate over the necks. This is the
same structure as the centrifugal barrier l(l+1)/r^2 in the radial equation.
Ground-state transverse width ~ k^{-1/2} (the semiclassical beam width).
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"


def A_of(x):
    return np.sqrt((1 + np.cos(x) ** 2) / 2)


Nx = 500
x = np.linspace(-np.pi / 2, np.pi / 2, Nx, endpoint=False)
dx = x[1] - x[0]
A = A_of(x)
Aface = 0.5 * (A + np.roll(A, -1)); Am = np.roll(Aface, 1)
r = np.r_[np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx]
c = np.r_[np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)]
v = np.r_[(Am + Aface) / dx, -Aface / dx, -Aface / dx]
Kx = sp.coo_matrix((v, (r, c)), shape=(Nx, Nx)).toarray()   # A-weighted stiffness
M = np.diag(A * dx)


def modes(k):
    S = Kx + np.diag(k ** 2 / A * dx)      # stiffness + potential k^2/A (weighted)
    E, V = eigh(S, M)                       # generalized, ascending; V is M-orthonormal
    return E, V


fig = plt.figure(figsize=(13, 5.2))

# ---- (1) potential well + bound-state ladder for k=6 ----
ax = fig.add_subplot(1, 2, 1)
k = 6
Vk = k ** 2 / A ** 2
barrier = 2 * k ** 2
ax.plot(x, Vk, "k", lw=2)
ax.axhline(barrier, ls="--", c="0.5", lw=1)
ax.text(0.0, barrier + 2, "barrier top (necks)", ha="center", fontsize=8, color="0.4")
E, V = modes(k)
bound = E[E < barrier]
for n, En in enumerate(bound):
    phi = V[:, n]; phi = phi / np.max(np.abs(phi)) * 5.5      # scale for display
    col = plt.cm.viridis(n / max(1, len(bound) - 1))
    ax.hlines(En, -np.pi / 2, np.pi / 2, color=col, lw=0.8, alpha=.6)
    ax.plot(x, En + phi, color=col, lw=1.4)
ax.set_ylim(k ** 2 - 4, barrier + 8)
ax.set_xlim(-np.pi / 2, np.pi / 2)
ax.set_xticks([-np.pi / 2, 0, np.pi / 2]); ax.set_xticklabels([r"$-\pi/2$", "belly\n0", r"$\pi/2$"])
ax.set_ylabel(r"energy  /  $V_k(x)=k^2/A^2$"); ax.set_xlabel("x")
ax.set_title(f"k={k}: {len(bound)} whispering-gallery bound states in the belly well")

# ---- (2) ground state narrows ~ k^{-1/2} ----
ax2 = fig.add_subplot(1, 2, 2)
for k in (3, 6, 12, 24):
    E, V = modes(k)
    dens = V[:, 0] ** 2
    dens = dens / np.trapz(dens, x)
    ax2.plot(x, dens, lw=1.8, label=f"k={k}")
ax2.set_xlim(-0.9, 0.9)
ax2.set_xlabel("x (transverse to the equator)")
ax2.set_ylabel(r"$|\varphi_0^{(k)}(x)|^2$")
ax2.set_title(r"ground-state quasimode concentrates $\sim k^{-1/2}$ on the belly")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle(r"Whispering-gallery quasimodes: $e^{ik\theta}\varphi(x)$ trapped in the centrifugal well $k^2/A^2$",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/quasimode_ladder.png", dpi=120)
print("wrote quasimode_ladder.png")
for k in (3, 6, 12, 24):
    E, _ = modes(k)
    nb = int(np.sum(E < 2 * k ** 2))
    print(f"  k={k:2d}: {nb} bound states; E0={E[0]:.2f} (well bottom {k**2}, barrier {2*k**2})")
