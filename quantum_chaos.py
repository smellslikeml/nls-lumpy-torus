"""
Frontier prototype 12: quantum chaos on the lump-perturbed surface.

The chaos panel showed the CLASSICAL picture: breaking axisymmetry with a theta-lump
shatters the integrable geodesic tori into a chaotic sea. Here is its quantum face.
Diagonalise the 2-D Laplace-Beltrami operator plus a generic theta-dependent lump
potential H = -Delta_g + lambda*V(x,theta), V a symmetry-free mix of Fourier modes
(so angular momentum is not conserved and no reflection symmetry survives to split
the spectrum). As lambda grows, the nearest-neighbour level-
spacing statistics cross over from Poisson (level clustering, integrable) to the
Wigner-GOE surmise (level REPULSION -- the universal fingerprint of quantum chaos),
and the eigenfunctions go from clean whispering-gallery order to irregular, scarred
states.

Symmetrised generalised eigenproblem (lumped mass M diagonal):
    K phi = E M phi   ->   (D K D + diag(V)) psi = E psi,   D = diag(1/sqrt(M)),  psi = sqrt(M) phi.
"""
import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"

Nx, Nth = 56, 56
grid = build_operators(Nx, Nth)
K = grid["K"].toarray()
Mdiag = grid["Mdiag"]
x, th = grid["x"], grid["th"]
X, TH = np.meshgrid(x, th, indexing="ij")
# A GENERIC lump: no residual x->-x, theta->-theta, or (x,theta)->(-x,-theta) symmetry
# (phase-offset, mixed sin/cos), so the chaotic spectrum is a single symmetry sector
# -> true GOE, not a symmetry-superposition that mimics Poisson.
Vshape = (np.cos(2 * X - 3 * TH) + 0.6 * np.cos(X + 2 * TH + 0.5)
          + 0.4 * np.sin(3 * X - TH + 0.9)).ravel()
D = 1.0 / np.sqrt(Mdiag)
DKD = (D[:, None] * K) * D[None, :]                        # symmetric M^{-1/2} K M^{-1/2}


def spectrum(lam):
    H = DKD + np.diag(lam * Vshape)
    E, V = eigh(H)
    phi = D[:, None] * V                                    # back to physical field
    return E, phi


def unfolded_spacings(E, lo=0.15, hi=0.75):
    """Unfold the bulk of the spectrum with a smooth polynomial fit of the counting
    function, then return nearest-neighbour spacings (mean ~ 1)."""
    E = np.sort(E)
    n = len(E)
    i0, i1 = int(lo * n), int(hi * n)
    Eb = E[i0:i1]
    counts = np.arange(i0, i1)
    c = np.polyfit(Eb, counts, 14)
    unf = np.polyval(c, Eb)
    s = np.diff(unf)
    return s[s > 0]


lams = {"near-integrable (λ=2.5)": 2.5, "chaotic (λ=120)": 120.0}
spac = {}
E_by = {}
phi_by = {}
for name, lam in lams.items():
    E, phi = spectrum(lam)
    spac[name] = unfolded_spacings(E)
    E_by[name] = E; phi_by[name] = phi
    # crude repulsion diagnostic: fraction of small spacings
    print(f"{name:26s}  P(s<0.3) = {np.mean(spac[name] < 0.3):.3f}  "
          f"(Poisson~0.26, GOE~0.07)", flush=True)

sgrid = np.linspace(0, 3, 200)
poisson = np.exp(-sgrid)
goe = (np.pi / 2) * sgrid * np.exp(-np.pi * sgrid ** 2 / 4)

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

# panel A: near-integrable spacings -> Poisson
axA = ax[0]
axA.hist(spac["near-integrable (λ=2.5)"], bins=np.linspace(0, 3, 26), density=True,
         color=BLUE, alpha=.55, label="λ=2.5 (near-integrable)")
axA.plot(sgrid, poisson, color="k", lw=1.8, label="Poisson $e^{-s}$")
axA.plot(sgrid, goe, color="0.6", lw=1.4, ls="--", label="GOE")
axA.set_xlabel("normalised spacing  s"); axA.set_ylabel("P(s)")
axA.set_title("weak lump: clustering (± k degeneracy)")
axA.legend(fontsize=8.5)

# panel B: chaotic spacings -> GOE (level repulsion)
axB = ax[1]
axB.hist(spac["chaotic (λ=120)"], bins=np.linspace(0, 3, 26), density=True,
         color=PINK, alpha=.55, label="λ=120 (chaotic)")
axB.plot(sgrid, goe, color="k", lw=1.8, label="GOE $\\frac{\\pi}{2}s\\,e^{-\\pi s^2/4}$")
axB.plot(sgrid, poisson, color="0.6", lw=1.4, ls="--", label="Poisson")
axB.set_xlabel("normalised spacing  s"); axB.set_ylabel("P(s)")
axB.set_title("strong lump: level REPULSION (Wigner-GOE)")
axB.legend(fontsize=8.5)

# panel C: an irregular/scarred eigenfunction of the chaotic surface
axC = ax[2]
E, phi = E_by["chaotic (λ=120)"], phi_by["chaotic (λ=120)"]
idx = int(0.55 * len(E))                                   # a mid-spectrum state
dens = (np.abs(phi[:, idx]) ** 2).reshape(Nx, Nth)
im = axC.pcolormesh(th, x, dens, shading="auto", cmap="magma")
axC.set_xlabel(r"$\theta$"); axC.set_ylabel("x")
axC.set_yticks([-np.pi / 2, 0, np.pi / 2]); axC.set_yticklabels(["neck", "belly", "neck"])
axC.set_xticks([0, np.pi, 2 * np.pi]); axC.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
axC.set_title(f"an irregular high-lying eigenstate  (n={idx})")
fig.colorbar(im, ax=axC, label=r"$|\psi|^2$")

fig.suptitle("Quantum chaos: a θ-lump drives the spectrum from level clustering to GOE repulsion — the quantum face of §10's chaos",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/quantum_chaos.png", dpi=120)
print("wrote quantum_chaos.png", flush=True)
