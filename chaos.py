"""
Direction 2: break axisymmetry -> the geodesic flow goes from integrable to chaotic.

Metric ds^2 = dx^2 + B(x,theta)^2 dtheta^2 with B = A(x)*(1 + delta*cos^2 x*cos theta).
delta=0 is axisymmetric (Clairaut: p_theta conserved -> integrable, nested curves).
delta>0 breaks the symmetry -> a Poincare section (x, p_x) at theta=0 mod 2pi shows
KAM islands embedded in a chaotic sea -- the classical backdrop for quantum scarring.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
eps = 1.0


def A(x):
    return np.sqrt((1 + eps * np.cos(x) ** 2) / (1 + eps))


def Ap(x):
    return -eps * np.sin(2 * x) / (2 * A(x) * (1 + eps))


def section(delta, n_ic=22, s_max=260.0):
    def B(x, th):
        return A(x) * (1 + delta * np.cos(x) ** 2 * np.cos(th))

    def Bx(x, th):
        c = np.cos(x) ** 2 * np.cos(th)
        return Ap(x) * (1 + delta * c) + A(x) * delta * (-np.sin(2 * x) * np.cos(th))

    def Bth(x, th):
        return A(x) * delta * (-np.cos(x) ** 2 * np.sin(th))

    def rhs(s, y):
        x, th, px, pth = y
        b = B(x, th)
        return [px, pth / b ** 2, pth ** 2 * Bx(x, th) / b ** 3, pth ** 2 * Bth(x, th) / b ** 3]

    pts_x, pts_px = [], []
    for x0 in np.linspace(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, n_ic):
        px0 = 0.0
        pth0 = B(x0, 0.0) * np.sqrt(max(1 - px0 ** 2, 0))     # energy 1/2 (unit speed)
        sol = solve_ivp(rhs, [0, s_max], [x0, 0.0, px0, pth0], max_step=0.02,
                        dense_output=True, rtol=1e-9, atol=1e-10)
        th = sol.y[1]
        # theta increases monotonically; record crossings theta = 2*pi*m
        mmax = int(th[-1] // (2 * np.pi))
        for m in range(1, mmax + 1):
            target = 2 * np.pi * m
            idx = np.searchsorted(th, target)
            if idx <= 0 or idx >= len(th):
                continue
            s0, s1 = sol.t[idx - 1], sol.t[idx]
            frac = (target - th[idx - 1]) / (th[idx] - th[idx - 1] + 1e-30)
            sc = s0 + frac * (s1 - s0)
            xc, _, pxc, _ = sol.sol(sc)
            xc = ((xc + np.pi / 2) % np.pi) - np.pi / 2
            pts_x.append(xc); pts_px.append(pxc)
    return np.array(pts_x), np.array(pts_px)


fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for a, delta, title in [(ax[0], 0.0, r"$\delta=0$: axisymmetric (integrable)"),
                        (ax[1], 0.35, r"$\delta=0.35$: $\theta$-lumpy (chaotic)")]:
    X, PX = section(delta)
    a.plot(X, PX, ".", ms=1.3, color="#1c2230", alpha=0.6)
    a.set_xlim(-np.pi / 2, np.pi / 2); a.set_ylim(-1.05, 1.05)
    a.set_xticks([-np.pi / 2, 0, np.pi / 2]); a.set_xticklabels([r"$-\pi/2$", "belly", r"$\pi/2$"])
    a.set_xlabel("x"); a.set_title(title)
ax[0].set_ylabel(r"$p_x$")
fig.suptitle("Geodesic Poincaré section: axisymmetry protects integrability; a θ-lump destroys it", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/chaos_poincare.png", dpi=120)
print("wrote chaos_poincare.png")
