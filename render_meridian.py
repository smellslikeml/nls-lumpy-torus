"""
Meridian-beam experiment: a beam concentrated transverse (theta) to the
meridian theta=0 and extended along it (x), running over the bumps.

A meridian crosses the positive-curvature belly (x=0, K=+0.5) and the
negative-curvature necks (x=+-pi/2, K=-1), so we expect the transverse (theta)
width to PINCH over the belly (focusing) and BULGE over the necks (defocusing)
-- the sign-changing-curvature (Hill/Floquet) case, unlike the stable
equatorial ring.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, beam_along_meridian, run
from render import animate_torus, animate_chart

OUT = "/home/thorax/nls_lumpy_torus"


def theta_width_at(grid, U, i0, thc=0.0):
    """RMS theta-width of the beam on the x-slice i0 (about thc)."""
    Nx, Nth = grid["Nx"], grid["Nth"]
    P = np.abs(U.reshape(Nx, Nth)[i0, :]) ** 2
    dthc = np.angle(np.exp(1j * (grid["th"] - thc)))
    return np.sqrt(np.sum(P * dthc ** 2) / (P.sum() + 1e-300))


if __name__ == "__main__":
    grid = build_operators(Nx=80, Nth=160)
    x = grid["x"]
    i_belly = int(np.argmin(np.abs(x - 0.0)))        # x=0, K=+0.5
    i_neck = int(np.argmin(np.abs(x - np.pi / 2)))   # x~+pi/2 neck, K=-1
    i_neck0 = 0                                       # x=-pi/2 (exact grid node)

    U0 = beam_along_meridian(grid, thc=0.0, amp=0.9, wth=0.4, q=4)
    U, hist, snaps, stats = run(grid, U0, dt=2e-3, T=2.5, sigma=-1.0, p=2,
                                n_snapshots=60, verbose=True)
    print(f"  max picard iters = {max(stats['picard_iters'])}")

    t = np.array(snaps["t"])
    w_belly = np.array([theta_width_at(grid, U, i_belly) for U in snaps["U"]])
    w_neck = np.array([0.5 * (theta_width_at(grid, U, i_neck)
                              + theta_width_at(grid, U, i_neck0))
                       for U in snaps["U"]])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, w_belly, "C0", label="belly  x=0  (K=+0.5)")
    ax[0].plot(t, w_neck, "C3", label="necks  x=$\\pm\\pi/2$  (K=$-1$)")
    ax[0].axhline(0.4, ls=":", c="k", lw=1, label="initial width")
    ax[0].set_title(r"transverse ($\theta$) width along the meridian")
    ax[0].set_xlabel("t"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(t, w_neck / w_belly, "C2")
    ax[1].axhline(1.0, ls=":", c="k", lw=1)
    ax[1].set_title(r"defocus ratio  width(neck)/width(belly)")
    ax[1].set_xlabel("t"); ax[1].grid(alpha=0.3)
    fig.suptitle("Meridian beam over the bumps: focusing at the belly vs defocusing at the necks")
    fig.tight_layout(); fig.savefig(f"{OUT}/meridian_focusing.png", dpi=120)
    plt.close(fig)
    print("wrote meridian_focusing.png")

    animate_chart(grid, snaps, f"{OUT}/nls_meridian_chart.gif")
    animate_torus(grid, snaps, f"{OUT}/nls_meridian_torus.gif")
