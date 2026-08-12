"""
3-D immersion of the lumpy torus over EXACTLY ONE period x in [-pi/2, pi/2].
After associating the ends (period pi), A has one maximum (x=0, the belly) and
one minimum (x=+-pi/2, a single circle after identification), giving exactly:
  - one ELLIPTIC closed geodesic  (belly,  A=1,    K=-A''/A=+0.5)
  - one HYPERBOLIC closed geodesic (neck,  A=0.707, K=-1)
Closing the single neck-belly-neck bump into a ring gives an asymmetric,
crescent-like torus (one bulge, one waist) -- not a symmetric multi-lump donut.

Metric ds^2 = dx^2 + A^2 dtheta^2, A=sqrt((1+cos^2 x)/2). Isometric immersion is
the surface of revolution (A cos th, A sin th, z), z=int sqrt(1-A'^2) dx.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

OUT = "/home/thorax/nls_lumpy_torus"


def A_of(x):
    return np.sqrt((1 + np.cos(x) ** 2) / 2)


def Ap_of(x):
    return -np.sin(2 * x) / (4 * A_of(x))


def K_of(x):
    c = np.cos(2 * x)
    return 2 * c / (3 + c) + np.sin(2 * x) ** 2 / (3 + c) ** 2


def z_of(x):
    g = np.sqrt(np.clip(1 - Ap_of(x) ** 2, 0, 1))
    return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])


cmap = matplotlib.colormaps["RdBu_r"]
norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.5)
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
thc = np.linspace(0, 2 * np.pi, 160)

fig = plt.figure(figsize=(15, 5.2))

# ---- (1) isometric immersion, ONE period (neck - belly - neck) ----
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
xt = np.linspace(-np.pi / 2, np.pi / 2, 260)
th = np.linspace(0, 2 * np.pi, 120)
Xt, TH = np.meshgrid(xt, th, indexing="ij")
At = A_of(Xt); zt = z_of(xt); Zt = np.broadcast_to(zt[:, None], Xt.shape)
Xs, Ys, Zs = At * np.cos(TH), At * np.sin(TH), Zt
fc = cmap(norm(np.broadcast_to(K_of(xt)[:, None], Xt.shape)))
ax1.plot_surface(Xs, Ys, Zs, facecolors=fc, rstride=1, cstride=2,
                 linewidth=0, antialiased=False, shade=False, alpha=1.0)
# geodesics: belly (x=0, elliptic) and the two identified necks (x=+-pi/2, hyperbolic)
ax1.plot(A_of(0) * np.cos(thc), A_of(0) * np.sin(thc), zt[len(xt) // 2] * np.ones_like(thc),
         color="limegreen", lw=3)
for zc in (zt[0], zt[-1]):
    ax1.plot(A_of(np.pi / 2) * np.cos(thc), A_of(np.pi / 2) * np.sin(thc),
             zc * np.ones_like(thc), color="magenta", lw=3)
ax1.set_box_aspect((1, 1, 1.6)); ax1.set_axis_off(); ax1.view_init(elev=10, azim=-70)
ax1.set_title("isometric immersion (one period)\nbelly=elliptic, ends=one neck (identified)", fontsize=10)

# ---- (2) bent-closed crescent-like torus, ONE period ----
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
xb = np.linspace(-np.pi / 2, np.pi / 2, 320)
zb = z_of(xb); Phi = 2 * np.pi * (zb - zb[0]) / (zb[-1] - zb[0])
thb = np.linspace(0, 2 * np.pi, 120)
Xb, THb = np.meshgrid(xb, thb, indexing="ij")
Ab = A_of(Xb); PHI = np.broadcast_to(Phi[:, None], Xb.shape); R = 2.2
Xt2 = (R + Ab * np.cos(THb)) * np.cos(PHI)
Yt2 = (R + Ab * np.cos(THb)) * np.sin(PHI)
Zt2 = Ab * np.sin(THb)
fc2 = cmap(norm(np.broadcast_to(K_of(xb)[:, None], Xb.shape)))
ax2.plot_surface(Xt2, Yt2, Zt2, facecolors=fc2, rstride=1, cstride=2,
                 linewidth=0, antialiased=False, shade=False)


def polar_circle(x0, Phi0, col):
    a = A_of(x0)
    ax2.plot((R + a * np.cos(thc)) * np.cos(Phi0), (R + a * np.cos(thc)) * np.sin(Phi0),
             a * np.sin(thc), color=col, lw=3)


polar_circle(0.0, np.pi, "limegreen")     # belly = elliptic (Phi=pi, far side)
polar_circle(-np.pi / 2, 0.0, "magenta")  # neck  = hyperbolic (Phi=0, the join)
ax2.set_box_aspect((1, 1, 0.55)); ax2.set_axis_off(); ax2.view_init(elev=38, azim=-55)
ax2.set_title("bent closed: crescent-like torus\n1 belly (elliptic) + 1 neck (hyperbolic)", fontsize=10)
cb = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.55, pad=0.02)
cb.set_label(r"Gaussian curvature  $K=-A''/A$")

# ---- (3) operator inputs over the one period ----
ax3 = fig.add_subplot(1, 3, 3)
xp = np.linspace(-np.pi / 2, np.pi / 2, 400)
ax3.plot(xp, A_of(xp), "k", label=r"$A(x)$  (metric radius)")
ax3.plot(xp, 1 / A_of(xp) ** 2, "C2", label=r"$1/A^2$  ($\theta$-coupling in $\Delta_g$)")
ax3.plot(xp, K_of(xp), "C3", lw=2, label=r"$K=-A''/A$")
ax3.axhline(0, color="0.6", lw=0.8)
ax3.axvline(0, ls=":", c="limegreen"); ax3.axvline(-np.pi / 2, ls=":", c="magenta")
ax3.axvline(np.pi / 2, ls=":", c="magenta")
ax3.text(0, 2.08, "elliptic", color="green", ha="center", fontsize=8)
ax3.text(np.pi / 2, 2.08, "hyperbolic", color="m", ha="right", fontsize=8)
ax3.set_xlabel("x  (one period, ends identified)"); ax3.set_title("what $\\Delta_g$ sees")
ax3.legend(fontsize=8, loc="center left"); ax3.grid(alpha=0.3)
ax3.set_xticks([-np.pi / 2, 0, np.pi / 2]); ax3.set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])

fig.savefig(f"{OUT}/curvature_view.png", dpi=120, bbox_inches="tight")
print("wrote curvature_view.png")
