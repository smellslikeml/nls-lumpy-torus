"""
Frontier prototype 5: a neural operator on the manifold (closes the post's own
ML-for-PDE motivation).

We learn the forward map  lump profile A(x)  ->  modal dispersion D_int(m)  with a
small MLP trained on solver-generated data, then use the surrogate IN PLACE of the
eigensolver to inverse-design a flat-dispersion resonator -- orders of magnitude
faster, which is the point of learned surrogates for PDE design loops.
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
Nx = 150
x = np.linspace(-np.pi / 2, np.pi / 2, Nx, endpoint=False)
dx = x[1] - x[0]
harm = np.array([np.cos(2 * x), np.cos(4 * x), np.cos(6 * x)])
ms = np.arange(8, 29); m0 = ms[len(ms) // 2]
rng = np.random.default_rng(0)


def A_of(a):
    return np.clip(1.0 + a @ harm, 0.35, None)


def dint_true(a):                                # ground-truth via eigensolver
    A = A_of(a)
    Aface = 0.5 * (A + np.roll(A, -1)); Am = np.roll(Aface, 1)
    r = np.r_[np.arange(Nx), np.arange(Nx), (np.arange(Nx) + 1) % Nx]
    c = np.r_[np.arange(Nx), (np.arange(Nx) + 1) % Nx, np.arange(Nx)]
    vv = np.r_[(Am + Aface) / dx, -Aface / dx, -Aface / dx]
    K = sp.coo_matrix((vv, (r, c)), shape=(Nx, Nx)).toarray()
    E = np.array([eigh(K + np.diag(m ** 2 / A * dx), np.diag(A * dx),
                       eigvals_only=True, subset_by_index=[0, 0])[0] for m in ms])
    c1, c0 = np.polyfit(ms - m0, E, 1)
    return E - (c0 + c1 * (ms - m0))


# ---- data ----
print("generating training data...", flush=True)
X = rng.uniform(-0.55, 0.55, (700, 3))
t0 = time.time(); Y = np.array([dint_true(a) for a in X]); t_solver = (time.time() - t0) / len(X)
Xtr, Ytr, Xte, Yte = X[:600], Y[:600], X[600:], Y[600:]

net = MLPRegressor(hidden_layer_sizes=(128, 128), activation="tanh",
                   max_iter=3000, tol=1e-7, random_state=0).fit(Xtr, Ytr)
Ypred = net.predict(Xte)
r2 = 1 - np.sum((Yte - Ypred) ** 2) / np.sum((Yte - Yte.mean()) ** 2)
t1 = time.time(); _ = net.predict(Xte); t_net = (time.time() - t1) / len(Xte)

# ---- surrogate-driven inverse design (flatten) ----
def loss_net(a):
    return float(np.sum(net.predict(a.reshape(1, -1))[0] ** 2))
res = minimize(loss_net, np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-4, "maxiter": 3000})
D_designed = dint_true(res.x)                    # verify with the TRUE solver

fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
for i in range(4):
    ax[0].plot(ms, Yte[i], "o", color="0.5", ms=4)
    ax[0].plot(ms, Ypred[i], "-", color="#2b6cb0", lw=1.4)
ax[0].plot([], [], "o", color="0.5", label="solver (truth)")
ax[0].plot([], [], "-", color="#2b6cb0", label="neural operator")
ax[0].set_xlabel("mode number m"); ax[0].set_ylabel(r"$D_{\rm int}(m)$")
ax[0].set_title(f"surrogate learns the dispersion map  ($R^2$={r2:.3f})")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ms, dint_true(np.zeros(3)), "o-", color="0.5", label="flat cylinder")
ax[1].plot(ms, D_designed, "o-", color="#b83280",
           label=f"surrogate-designed (spread {np.ptp(D_designed):.1f})")
ax[1].axhline(0, color="0.7", lw=.8)
ax[1].set_xlabel("mode number m"); ax[1].set_ylabel(r"$D_{\rm int}(m)$")
ax[1].set_title(f"inverse-design via the surrogate  ({t_solver/t_net:.0f}× faster/eval)")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("Neural operator on the manifold: learn geometry → dispersion, then design with it",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/neural_operator.png", dpi=120)
print(f"wrote neural_operator.png  R2={r2:.3f}, speedup {t_solver/t_net:.0f}x, "
      f"design spread {np.ptp(dint_true(np.zeros(3))):.0f}->{np.ptp(D_designed):.1f}")
