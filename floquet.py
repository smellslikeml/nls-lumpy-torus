"""
Frontier prototype 8: Floquet engineering -> a TIME route to topological edge modes.

Breathe the lump chain periodically instead of twisting it in space. Model the
whispering-gallery chain as a driven tight-binding lattice with a two-step cycle:
for half a period the intra-lump bonds are on (angle th1), for the other half the
inter-lump bonds are (angle th2) -- exactly what a breathing lump chain does to its
hopping. The one-period (Floquet) operator
      U = R_inter(th2) . R_intra(th1)
has QUASIENERGY bands; on an open chain, edge modes appear in BOTH gaps -- at
quasienergy 0 (a static-like edge mode) AND at the Floquet-only value eps=pi (an
"anomalous" pi-mode with no static analogue). This is the time-domain sibling of the
chiral-twist edge modes built in space (topological.py): drive, don't twist.

Because the intra/inter bond operators are block-diagonal 2x2 (sigma_x) blocks, each
half-step is an exact block rotation exp(-i th sigma_x) = cos th - i sin th sigma_x,
so no numerical matrix exponential is needed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Ncell = 20
Ns = 2 * Ncell
intra_pairs = [(2 * n, 2 * n + 1) for n in range(Ncell)]          # (0,1),(2,3),...
inter_pairs = [(2 * n + 1, 2 * n + 2) for n in range(Ncell - 1)]  # (1,2),(3,4),...  ends free
site = np.arange(Ns)
center = (Ns - 1) / 2.0


def rot(theta, pairs):
    """exp(-i theta * H) for H a sum of sigma_x on disjoint site pairs (ends -> identity)."""
    M = np.eye(Ns, dtype=complex)
    c, s = np.cos(theta), -1j * np.sin(theta)
    for a, b in pairs:
        M[a, a] = c; M[b, b] = c; M[a, b] = s; M[b, a] = s
    return M


def floquet(U_intra, th2):
    w, v = np.linalg.eig(rot(th2, inter_pairs) @ U_intra)
    eps = np.angle(w)
    edge = np.sum(np.abs(v) ** 2 * ((site[:, None] - center) / center), axis=0)   # -1..+1
    return eps, edge, v


def has_edge(eps, edge, target, sign=0):
    d = np.abs(np.angle(np.exp(1j * (eps - target))))            # circular distance to target
    mask = (d < 0.4) & (np.abs(edge) > 0.5)
    if sign > 0:
        mask &= edge > 0
    elif sign < 0:
        mask &= edge < 0
    cand = np.where(mask)[0]
    return cand[np.argmax(np.abs(edge[cand]))] if len(cand) else None


# search (th1, th2) for a point hosting BOTH a 0-edge and a pi-edge mode
th2s = np.linspace(0.0, 2 * np.pi, 240)
best = None
for th1 in np.linspace(0.15, 0.85, 15) * np.pi:
    Ui = rot(th1, intra_pairs)
    n_both, star = 0, None
    for th2 in th2s:
        eps, edge, _ = floquet(Ui, th2)
        if has_edge(eps, edge, 0.0) is not None and has_edge(eps, edge, np.pi) is not None:
            n_both += 1
            if star is None or abs(th2 - np.pi) < abs(star - np.pi):
                star = th2
    if n_both and (best is None or n_both > best[1]):
        best = (th1, n_both, star)
th1, nb, th2_star = best
print(f"chosen th1={th1/np.pi:.3f}pi  th2*={th2_star/np.pi:.3f}pi  (both-gap points: {nb})", flush=True)

Ui = rot(th1, intra_pairs)
EPS = np.zeros((len(th2s), Ns)); EDGE = np.zeros((len(th2s), Ns))
for i, th2 in enumerate(th2s):
    e, ed, _ = floquet(Ui, th2)
    EPS[i], EDGE[i] = e, ed

eps0, edge0, v0 = floquet(Ui, th2_star)
# pick the 0-mode on the LEFT edge and the pi-mode on the RIGHT edge so they are
# visually distinct (each gap hosts a mode on both edges by symmetry)
_iz = has_edge(eps0, edge0, 0.0, sign=-1)
i_zero = _iz if _iz is not None else has_edge(eps0, edge0, 0.0)
_ip = has_edge(eps0, edge0, np.pi, sign=+1)
i_pi = _ip if _ip is not None else has_edge(eps0, edge0, np.pi)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc = ax[0].scatter(np.repeat(th2s, Ns), EPS.ravel(), c=EDGE.ravel(), cmap="coolwarm", s=4, vmin=-1, vmax=1)
for y in (-np.pi, 0, np.pi):
    ax[0].axhline(y, color="0.6", lw=0.8, ls=":")
ax[0].axvline(th2_star, color="k", lw=0.9, ls="--")
ax[0].set_xlabel(r"inter-lump drive angle  $\theta_2$")
ax[0].set_ylabel(r"quasienergy  $\varepsilon$")
ax[0].set_yticks([-np.pi, 0, np.pi]); ax[0].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
ax[0].set_xticks([0, np.pi, 2 * np.pi]); ax[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax[0].set_title(rf"quasienergy bands ($\theta_1$={th1/np.pi:.2f}$\pi$): edge modes in 0 & $\pi$ gaps")
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("edge localization  (−1 left … +1 right)")

if i_zero is not None:
    ax[1].plot(site, np.abs(v0[:, i_zero]) ** 2, color="#2b6cb0", lw=1.9,
               label=rf"$\varepsilon\approx0$ edge mode")
if i_pi is not None:
    ax[1].plot(site, np.abs(v0[:, i_pi]) ** 2, color="#b83280", lw=1.9,
               label=r"$\varepsilon\approx\pi$ mode (anomalous, no static analogue)")
ax[1].set_xlabel("lump site"); ax[1].set_ylabel(r"$|\psi|^2$")
ax[1].set_title(rf"edge modes at $\theta_2$={th2_star/np.pi:.2f}$\pi$")
ax[1].legend(); ax[1].grid(alpha=0.3)

fig.suptitle(r"Floquet engineering: periodically breathing the lump chain makes edge modes in TIME "
             r"(a $\pi$-mode has no static analogue)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/floquet.png", dpi=120)
zt = f"{eps0[i_zero]:+.3f}/{edge0[i_zero]:+.2f}" if i_zero is not None else "none"
pt = f"{eps0[i_pi]:+.3f}/{edge0[i_pi]:+.2f}" if i_pi is not None else "none"
print(f"wrote floquet.png  (0-mode eps/edge={zt};  pi-mode eps/edge={pt})", flush=True)
