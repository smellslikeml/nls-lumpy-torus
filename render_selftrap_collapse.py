"""
The dramatic amp~5 case on the elliptic equator: the focusing ring self-traps
into a tight soliton (transverse width locks) and then the ring's azimuthal
(symmetry-breaking) instability grows and drives a mass-critical collapse.
Steps with an early-stop guard and renders with a capped colormap so the locked
phase stays readable and the collapse shows as a saturated hot spot.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, beam_along_geodesic, make_stepper, mass
from render import animate_torus

OUT = "/home/thorax/nls_lumpy_torus"
Lx = np.pi
grid = build_operators(Nx=160, Nth=128)
x, A = grid["x"], grid["A"]
Nx, Nth = grid["Nx"], grid["Nth"]
Mdiag = grid["Mdiag"]


def xwidth_peak(U):
    f = np.abs(U.reshape(Nx, Nth)) ** 2
    Px = f.sum(axis=1) * A * grid["dx"] * grid["dth"]
    d = (x - 0.0 + Lx / 2) % Lx - Lx / 2
    return np.sqrt(np.sum(Px * d ** 2) / (Px.sum() + 1e-300)), float(f.max())


def azim_frac(U):
    """fraction of mass outside the k=6 sector -> azimuthal symmetry breaking."""
    F = np.fft.fft(U.reshape(Nx, Nth), axis=1)
    tot = np.sum(np.abs(F) ** 2)
    return 1.0 - np.sum(np.abs(F[:, 6]) ** 2) / (tot + 1e-300)


dt = 1e-3
step, stats = make_stepper(grid, dt=dt, sigma=-1.0, p=2)
U = beam_along_geodesic(grid, xc=0.0, amp=5.0, wx=0.30, k=6)
m0 = mass(U, Mdiag)

nsteps = 2500
snap_every = 20
snaps = {"t": [0.0], "U": [U.copy()]}
hist = {"t": [0.0], "w": [xwidth_peak(U)[0]], "pk": [xwidth_peak(U)[1]], "az": [azim_frac(U)]}
t_collapse = None
for n in range(1, nsteps + 1):
    U = step(U)
    t = n * dt
    if n % 10 == 0:
        w, pk = xwidth_peak(U)
        hist["t"].append(t); hist["w"].append(w); hist["pk"].append(pk)
        hist["az"].append(azim_frac(U))
    if n % snap_every == 0:
        snaps["t"].append(t); snaps["U"].append(U.copy())
    pk_now = float(np.max(np.abs(U) ** 2))
    if not np.isfinite(pk_now) or pk_now > 80 or stats["picard_iters"][-1] >= 59:
        t_collapse = t
        print(f"collapse onset at t={t:.3f}  peak={pk_now:.1f}", flush=True)
        snaps["t"].append(t); snaps["U"].append(U.copy())
        break
    if n % 200 == 0:
        print(f"  t={t:.2f}  width={hist['w'][-1]:.3f}  peak={hist['pk'][-1]:.2f}  "
              f"azim_frac={hist['az'][-1]:.1e}  mass_drift={(mass(U,Mdiag)-m0)/m0:+.1e}", flush=True)

for k in hist:
    hist[k] = np.array(hist[k])

fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(hist["t"], hist["w"], "C1"); ax[0].axhline(0.30, ls=":", c="k", lw=1)
ax[0].set_ylabel("transverse width"); ax[0].set_title("width LOCKS (self-trapped soliton) then collapses")
ax[1].semilogy(hist["t"], hist["pk"], "C3"); ax[1].set_ylabel(r"peak $|u|^2$ (log)")
ax[2].semilogy(hist["t"], np.maximum(hist["az"], 1e-16), "C0")
ax[2].set_ylabel("mass outside k=6"); ax[2].set_xlabel("t")
ax[2].set_title("azimuthal symmetry-breaking grows -> triggers collapse")
if t_collapse:
    for a in ax:
        a.axvline(t_collapse, ls="--", c="0.5", lw=1)
fig.suptitle("Equator ring, focusing amp=5: self-trapping then mass-critical (azimuthal) collapse")
fig.tight_layout(); fig.savefig(f"{OUT}/selftrap_collapse.png", dpi=120); plt.close(fig)
print("wrote selftrap_collapse.png", flush=True)

# cap colormap: locked soliton peaks ~25-30, so vmax=35 keeps it readable; collapse saturates
animate_torus(grid, snaps, f"{OUT}/nls_selftrap_collapse_torus.gif", fps=16, vmax=35.0)
print("ALL DONE", flush=True)
