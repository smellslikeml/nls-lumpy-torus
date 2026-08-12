"""
Frontier prototype 4: a chiral lump twist -> topological whispering-gallery bands.

Twist the lumps helically, A(x - c theta): the on-site WG energy k^2/A^2(x - c theta)
becomes, for a chain of lumps n and theta-momentum k_theta, the Harper on-site term
2 lambda cos(2 pi alpha n + k_theta) with alpha proportional to the twist rate c.
So the twisted-lump chain maps onto the Harper-Hofstadter model -> topological
bands with CHIRAL EDGE MODES traversing the gaps (robust one-way transport), a
geometric route to topological photonics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
N = 40                      # lump sites (open chain in x -> a strip with two edges)
alpha = 1.0 / 3.0          # flux per plaquette from the twist
t_hop, lam = 1.0, 1.0
kth = np.linspace(0, 2 * np.pi, 240)

E = np.zeros((len(kth), N)); edge = np.zeros((len(kth), N))
for i, k in enumerate(kth):
    H = np.diag(2 * lam * np.cos(2 * np.pi * alpha * np.arange(N) + k))
    H += np.diag(-t_hop * np.ones(N - 1), 1) + np.diag(-t_hop * np.ones(N - 1), -1)
    w, v = np.linalg.eigh(H)
    E[i] = w
    n = np.arange(N)
    edge[i] = (v ** 2 * (n[:, None] - (N - 1) / 2)).sum(0) / ((N - 1) / 2)   # +1 top edge, -1 bottom

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc = ax[0].scatter(np.repeat(kth, N), E.ravel(), c=edge.ravel(), cmap="coolwarm",
                   s=3, vmin=-1, vmax=1)
ax[0].set_xlabel(r"$\theta$-momentum $k_\theta$"); ax[0].set_ylabel("energy")
ax[0].set_xticks([0, np.pi, 2 * np.pi]); ax[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax[0].set_title(f"Hofstadter strip (flux {alpha:.2g}): edge modes cross the gaps")
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("edge localization  (−1 bottom … +1 top)")

# a couple of edge eigenmodes at a k_theta inside the lower gap
k = 2.6
H = np.diag(2 * lam * np.cos(2 * np.pi * alpha * np.arange(N) + k))
H += np.diag(-t_hop * np.ones(N - 1), 1) + np.diag(-t_hop * np.ones(N - 1), -1)
w, v = np.linalg.eigh(H)
# pick the two states with largest |edge| in the lower gap window
gapmask = (w > -3.2) & (w < -1.6)
idx = np.where(gapmask)[0]
idx = idx[np.argsort(-np.abs((v[:, idx] ** 2 * (np.arange(N)[:, None] - (N - 1) / 2)).sum(0)))][:2]
for j, col in zip(idx, ["#b83280", "#2b6cb0"]):
    ax[1].plot(np.arange(N), v[:, j] ** 2, color=col, lw=1.8,
               label=f"edge mode (E={w[j]:.2f})")
ax[1].set_xlabel("lump site n (x)"); ax[1].set_ylabel(r"$|\psi|^2$")
ax[1].set_title("gap states live on opposite edges"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Chiral lump twist → topological whispering-gallery bands (a geometric route to edge modes)",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/topological.png", dpi=120)
print("wrote topological.png")
