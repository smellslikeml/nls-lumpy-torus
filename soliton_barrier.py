"""
Direction 4: soliton-geometry interaction on the reduced belly-to-neck line.

The k-sector reduces to 1-D focusing NLS with the geometric potential V_k=k^2/A^2
(well at belly, barrier at neck). A bright soliton given a small kick LIBRATES in
the belly well (trapped); given a larger kick it climbs over the neck barrier and
CIRCULATES -- the nonlinear analogue of the librating/circulating geodesics.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
k = 4
Nx = 400
x = np.linspace(-np.pi / 2, np.pi / 2, Nx, endpoint=False)
dx = x[1] - x[0]
A = np.sqrt((1 + np.cos(x) ** 2) / 2)
Aface = 0.5 * (A + np.roll(A, -1)); Am = np.roll(Aface, 1)
r = np.r_[np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx]
c = np.r_[np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)]
v = np.r_[(Am + Aface) / dx, -Aface / dx, -Aface / dx]
Kx = sp.coo_matrix((v, (r, c)), shape=(Nx, Nx)).tocsr()
Mdiag = A * dx
Vk = k ** 2 / A ** 2 * dx                          # potential (lumped, weighted)
Mmat = sp.diags(Mdiag)


def evolve(p, T=3.0, Nt=3000, sigma=-1.0, eta=3.0):
    dt = T / Nt
    L = (1j * Mmat - 0.5 * dt * (Kx + sp.diags(Vk))).tocsc()
    lu = splu(L)
    rhs_op = (1j * Mmat + 0.5 * dt * (Kx + sp.diags(Vk)))
    u = (eta / np.cosh(eta * x) * np.exp(1j * p * x)).astype(complex)   # soliton + kick
    carpet = np.empty((Nt // 6, Nx)); ci = 0
    for n in range(Nt):
        base = rhs_op @ u
        w = u.copy()
        for _ in range(4):
            uh = 0.5 * (w + u)
            w = lu.solve(base + sigma * dt * (Mdiag * (np.abs(uh) ** 2 * uh)))
        u = w
        if n % 6 == 0:
            carpet[ci] = np.abs(u) ** 2; ci += 1
    return carpet[:ci]


fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for a, p, title in [(ax[0], 2.0, "small kick → trapped (librates in the well)"),
                    (ax[1], 6.0, "large kick → circulates over the necks")]:
    cp = evolve(p)
    im = a.imshow(cp, origin="lower", aspect="auto", cmap="inferno",
                  extent=[-np.pi / 2, np.pi / 2, 0, 3.0], vmax=np.percentile(cp, 99.5))
    a.set_xticks([-np.pi / 2, 0, np.pi / 2]); a.set_xticklabels([r"$-\pi/2$", "belly", r"$\pi/2$"])
    a.set_xlabel("x"); a.set_title(title)
ax[0].set_ylabel("t")
fig.suptitle(r"A bright soliton in the geometric well $V_k=k^2/A^2$: trapped vs over-the-neck (k=4)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/soliton_barrier.png", dpi=120)
print("wrote soliton_barrier.png")
