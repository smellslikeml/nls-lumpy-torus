"""
Correct diagnostic for the meridian beam. The transverse separation of nearby
meridians is the Jacobi field J(x) = A(x): FARTHEST apart at the belly (A=1,
max) and CLOSEST at the necks (A=0.707, min). So a meridian-aligned beam is
transversally WIDEST (defocused) at the belly and NARROWEST (focused) at the
necks -- the necks are the focusing waists. (Same as a sphere: meridians
diverge at the equator, converge at the poles, even though K>0 throughout.)

We show it with the PHYSICAL transverse width A(x)*w_theta, which is what the
Jacobi field controls; the coordinate width w_theta hides it because the metric
factor A divides back out.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, beam_along_meridian, run, profile_A

OUT = "/home/thorax/nls_lumpy_torus"
grid = build_operators(Nx=80, Nth=160)
x, th = grid["x"], grid["th"]
i_belly = int(np.argmin(np.abs(x - 0.0)))     # A=1
i_neck = 0                                     # x=-pi/2, A=0.707
A_belly, A_neck = float(profile_A(x[i_belly])), float(profile_A(x[i_neck]))


def wtheta(U, i0):
    P = np.abs(U.reshape(grid["Nx"], grid["Nth"])[i0, :]) ** 2
    d = np.angle(np.exp(1j * (th - 0.0)))
    return np.sqrt(np.sum(P * d ** 2) / (P.sum() + 1e-300))


# linear, wide beam -> pure geometry, stays coherent longer
U0 = beam_along_meridian(grid, thc=0.0, amp=1.0, wth=1.0, q=4)
U, hist, snaps, stats = run(grid, U0, dt=2e-3, T=1.5, sigma=0.0, p=2,
                            n_snapshots=60, verbose=False)
t = np.array(snaps["t"])
wb = np.array([wtheta(U, i_belly) for U in snaps["U"]])
wn = np.array([wtheta(U, i_neck) for U in snaps["U"]])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, wb, "C0", label="belly x=0")
ax[0].plot(t, wn, "C3", label="neck x=$-\\pi/2$")
ax[0].set_title(r"coordinate width $w_\theta$  (metric factor hidden)")
ax[1].plot(t, A_belly * wb, "C0", label=f"belly  (A={A_belly:.2f})")
ax[1].plot(t, A_neck * wn, "C3", label=f"neck   (A={A_neck:.2f})")
ax[1].set_title(r"physical width $A\,w_\theta$  — necks focus, belly defocuses")
for a in ax:
    a.set_xlabel("t"); a.legend(); a.grid(alpha=0.3)
fig.suptitle("Meridian beam: the Jacobi field J=A(x) compresses the beam at the necks")
fig.tight_layout(); fig.savefig(f"{OUT}/meridian_diag.png", dpi=120)
print("wrote meridian_diag.png")
print(f"physical width ratio belly/neck at end: "
      f"{A_belly*wb[-1] / (A_neck*wn[-1]):.2f}  (A ratio = {A_belly/A_neck:.2f})")
print(f"mass drift {(hist['mass'][-1]-hist['mass'][0])/hist['mass'][0]:+.1e}")
