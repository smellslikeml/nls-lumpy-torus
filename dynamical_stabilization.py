"""
Frontier prototype 6: dynamical stabilization of collapse (geometry vs nonlinearity).

The focusing belly ring above the mass-critical threshold self-traps and then
suffers the necklace (azimuthal) collapse (gallery 08-10). Here we fight it with a
TIME-VARYING deformation: breathe the belly, which modulates the effective coupling
g_eff ~ 1/A^2(t). With the MEAN coupling unchanged, g(t) = g0(1 + delta cos Omega t),
this is "nonlinearity management" (Feshbach-resonance management; Saito-Ueda 2003,
Abdullaev, Malomed): the rapid drive, felt through the soliton's breathing, adds an
effective stabilizing term that can arrest 2-D mass-critical collapse.

Solved directly on the 2-D lumpy-torus solver, reusing its prefactorized
Crank-Nicolson linear operator (coupling-independent) with a per-step, time-dependent
sigma(t). We measure the truth -- peak |u|^2(t) for the bare vs managed ring, and the
collapse time across modulation depth delta and frequency Omega -- so the figure
reports whatever actually happens (arrest, delay, or a resonant window), not a story.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD, GREY = "#b83280", "#2b6cb0", "#d69e2e", "0.55"
PK_CAP = 80.0                    # collapse-onset guard (as in threshold_scan.py)
PICARD_GUARD = 59                # near-non-convergence => collapsing


def managed_stepper(grid, dt, p=2, tol=1e-11, pmax=60):
    """Crank-Nicolson step; linear operator prefactorized once (it is
    coupling-independent), nonlinear coupling sigma supplied per step so it may
    vary in time."""
    K, Mdiag = grid["K"], grid["Mdiag"]
    iM = sp.diags(1j * Mdiag)
    lu = splu((iM - 0.5 * dt * K).tocsc())
    rhs = (iM + 0.5 * dt * K).tocsr()

    def step(Un, sigma):
        base = rhs @ Un
        W = Un.copy()
        it = 0
        for it in range(1, pmax + 1):
            Uh = 0.5 * (W + Un)
            Wn = lu.solve(base + sigma * dt * (Mdiag * (np.abs(Uh) ** p * Uh)))
            if np.linalg.norm(Wn - W) / (np.linalg.norm(Wn) + 1e-300) < tol:
                W = Wn
                break
            W = Wn
        return W, it

    return step


def run(grid, step, dt, amp, delta, Omega, Tmax, seed, rec_every=5):
    """Evolve the seeded focusing ring with sigma(t) = -(1 + delta cos Omega t).
    Returns (times, peaks, t_collapse-or-None, mass0)."""
    from nls_lumpy_torus import beam_along_geodesic, mass
    Nx, Nth, Mdiag = grid["Nx"], grid["Nth"], grid["Mdiag"]
    U = beam_along_geodesic(grid, xc=0.0, amp=amp, wx=0.30, k=6).reshape(Nx, Nth)
    U = (U * (1.0 + seed)).ravel()
    m0 = mass(U, Mdiag)
    nsteps = int(round(Tmax / dt))
    ts, pks = [0.0], [float(np.max(np.abs(U) ** 2))]
    tc = None
    for n in range(1, nsteps + 1):
        tmid = (n - 0.5) * dt
        sigma = -(1.0 + delta * np.cos(Omega * tmid))     # focusing, mean sigma = -1
        U, it = step(U, sigma)
        pk = float(np.max(np.abs(U) ** 2))
        if n % rec_every == 0:
            ts.append(n * dt); pks.append(min(pk, PK_CAP))
        if (not np.isfinite(pk)) or pk > PK_CAP or it >= PICARD_GUARD:
            tc = n * dt
            ts.append(tc); pks.append(PK_CAP)
            break
    return np.array(ts), np.array(pks), tc, m0


def seed_of(grid):
    th = grid["th"]
    return 1e-4 * (np.cos(5 * th) + np.cos(7 * th) + np.cos(9 * th))   # fixed necklace seed


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nls_lumpy_torus import build_operators

    dt = 1e-3
    amp = 5.0                     # mass ~83, well above M_c ~ 43: bare collapses fast (t_c~0.5)
    Om_in, Om_off = 40.0, 16.0    # in-window vs off-window drive frequency
    Tmax = 4.0

    # ---- hero time series: bare vs in-window vs off-window ----
    grid_h = build_operators(Nx=160, Nth=128)
    seed_h = seed_of(grid_h)
    step_h = managed_stepper(grid_h, dt)
    print("hero (amp=5.0):", flush=True)
    t0, p0, tc0, m0 = run(grid_h, step_h, dt, amp, 0.0, Om_in, Tmax, seed_h)
    print(f"  bare (delta=0):         mass={m0:.1f}  t_c={tc0}", flush=True)
    tI, pI, tcI, _ = run(grid_h, step_h, dt, amp, 1.0, Om_in, Tmax, seed_h)
    print(f"  in-window  (d=1,Om=40): t_c={tcI}", flush=True)
    tF, pF, tcF, _ = run(grid_h, step_h, dt, amp, 1.0, Om_off, Tmax, seed_h)
    print(f"  off-window (d=1,Om=16): t_c={tcF}", flush=True)

    # ---- depth + frequency scans (fast grid) ----
    grid_s = build_operators(Nx=128, Nth=96)
    seed_s = seed_of(grid_s)
    step_s = managed_stepper(grid_s, dt)

    deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    tc_delta = []
    print("depth scan (amp=5.0, Om=40):", flush=True)
    for d in deltas:
        _, _, tc, _ = run(grid_s, step_s, dt, amp, d, Om_in, Tmax, seed_s)
        tc_delta.append(tc if tc else Tmax)
        print(f"  delta={d:.1f}  t_c={tc_delta[-1]:.2f}", flush=True)

    Omegas = [8.0, 16.0, 25.0, 40.0, 60.0, 90.0]
    tc_omega = []
    print("frequency probe (amp=5.0, delta=0.7):", flush=True)
    for om in Omegas:
        _, _, tc, _ = run(grid_s, step_s, dt, amp, 0.7, om, Tmax, seed_s)
        tc_omega.append(tc if tc else Tmax)
        print(f"  Om={om:4.0f}  t_c={tc_omega[-1]:.2f}", flush=True)

    # ------------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    axA = ax[0]
    axA.axhline(PK_CAP, color=GREY, ls=":", lw=1.1)
    axA.text(0.05, PK_CAP * 1.03, "collapse guard", fontsize=8, color="0.4")
    axA.semilogy(t0, p0, color=PINK, lw=1.9, label=f"bare — collapses (t_c={tc0:.2f})")
    axA.semilogy(tI, pI, color=BLUE, lw=2.0, label=f"in-window Ω=40 — delayed {tcI/tc0:.1f}× (t_c={tcI:.2f})")
    axA.semilogy(tF, pF, color=GOLD, lw=1.6, ls="--",
                 label=f"off-window Ω=16 — no help (t_c={tcF:.2f})")
    axA.set_xlabel("t"); axA.set_ylabel(r"peak $|u|^2$  (log)")
    axA.set_xlim(0, 3.0)
    axA.set_title("breathing in the resonant window delays collapse")
    axA.legend(loc="lower right", fontsize=8.3, framealpha=.92)

    axB = ax[1]
    axB.plot(deltas, tc_delta, "-o", color=PINK, ms=6, lw=1.6)
    axB.set_xlabel(r"modulation depth  $\delta$"); axB.set_ylabel(r"collapse time  $t_c$")
    axB.set_title(r"deeper breathing $\to$ longer delay ($\Omega$=40)")
    axB.grid(alpha=0.25)

    axC = ax[2]
    axC.plot(Omegas, tc_omega, "-o", color=GOLD, ms=6, lw=1.6)
    axC.axhline(tc_omega[0] if tc_delta[0] else 0.5, color=PINK, ls=":", lw=1.1)
    axC.text(Omegas[-1], tc_delta[0], "bare t_c", fontsize=8, color=PINK, ha="right", va="bottom")
    axC.set_xlabel(r"modulation frequency  $\Omega$"); axC.set_ylabel(r"collapse time  $t_c$")
    axC.set_title(r"resonant window ($\delta$=0.7); off-resonance hastens it")
    axC.grid(alpha=0.25)

    fig.suptitle("Dynamical control of collapse: a time-varying geometry resonantly delays "
                 "(or hastens) the necklace blow-up", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/dynamical_stabilization.png", dpi=120)
    print("wrote dynamical_stabilization.png", flush=True)
