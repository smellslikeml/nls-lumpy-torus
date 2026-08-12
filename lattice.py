"""
Direction 3: corrugation as an intrinsic optical lattice / disorder.

A periodic chain of lumps makes V_k = k^2/A^2 a periodic potential -> whispering-
gallery BLOCH BANDS with gaps. Random lump depths make it a disordered potential
-> ANDERSON LOCALIZATION of the modes. No external potential -- the geometry is
the lattice.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
k = 6


def A_of(x, eps=1.0):
    return np.sqrt((1 + eps * np.cos(x) ** 2) / (1 + eps))


# ---- Bloch band structure of one lump cell (period a=pi) ----
a = np.pi
Nc = 240
xc = np.linspace(-np.pi / 2, np.pi / 2, Nc, endpoint=False)
dx = xc[1] - xc[0]
A = A_of(xc)
Aface = 0.5 * (A + np.roll(A, -1)); Am = np.roll(Aface, 1)


def bloch_H(q):
    # A-weighted stiffness with Bloch phase on the wrap edge (Nc-1 -> 0)
    S = np.zeros((Nc, Nc), complex)
    for i in range(Nc):
        j = (i + 1) % Nc
        w = Aface[i] / dx
        ph = np.exp(1j * q * a) if j < i else 1.0          # wrap edge carries e^{iqa}
        S[i, j] += -w * ph; S[j, i] += -w * np.conj(ph)
        S[i, i] += w; S[j, j] += w
    S += np.diag(k ** 2 / A * dx)
    return S


qs = np.linspace(-1, 1, 121)
M = np.diag(A * dx)
bands = np.array([eigh(bloch_H(np.pi * qq), M, eigvals_only=True)[:5] for qq in qs])

# ---- Anderson: disordered chain of cells ----
rng = np.random.default_rng(3)
Mcell = 24
eps_cells = rng.uniform(0.3, 3.0, Mcell)                   # random lump depths
xs = np.concatenate([xc + m * a for m in range(Mcell)])
Ad = np.concatenate([A_of(xc, e) for m, e in enumerate(eps_cells)])
Ntot = xs.size
Af = 0.5 * (Ad + np.roll(Ad, -1)); Amm = np.roll(Af, 1)
r = np.r_[np.arange(Ntot), np.arange(Ntot), (np.arange(Ntot) + 1) % Ntot]
c = np.r_[np.arange(Ntot), (np.arange(Ntot) + 1) % Ntot, np.arange(Ntot)]
v = np.r_[(Amm + Af) / dx, -Af / dx, -Af / dx]
Kx = sp.coo_matrix((v, (r, c)), shape=(Ntot, Ntot)).tocsr() + sp.diags(k ** 2 / Ad * dx)
Md = sp.diags(Ad * dx)
Ed, Vd = eigsh(Kx, k=12, M=Md, sigma=k ** 2 + 1, which="LM")
order = np.argsort(Ed); Ed, Vd = Ed[order], Vd[:, order]
mode = Vd[:, 0]; mode = mode / np.max(np.abs(mode))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for b in range(5):
    ax[0].plot(qs, bands[:, b], color="#2b6cb0", lw=1.6)
ax[0].set_xlabel(r"Bloch momentum $q$ (units $\pi/a$)"); ax[0].set_ylabel("energy")
ax[0].set_title(f"whispering-gallery Bloch bands (k={k})"); ax[0].grid(alpha=0.3)
ax[1].plot(xs / a, mode ** 2, color="#b83280", lw=1)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 2)
ax[1].set_xlabel("cell index  $x/a$"); ax[1].set_ylabel(r"$|\varphi|^2$ (log)")
ax[1].set_title("Anderson-localized mode (disordered lumps)"); ax[1].grid(alpha=0.3)
fig.suptitle("Corrugation as an intrinsic lattice: Bloch bands (ordered) → Anderson localization (disordered)",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/lattice.png", dpi=120)
print("wrote lattice.png")
