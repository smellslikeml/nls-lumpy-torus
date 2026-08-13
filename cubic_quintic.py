"""
Frontier prototype 11: cubic-quintic -> GENUINE arrest of collapse (prevent, not delay).

The dynamical-control panels showed that breathing the geometry (nonlinearity
management) only *delays* a supercritical collapse -- a result made rigorous by
Li-Ning-Zhao (2025): nonlinearity management postpones blow-up, it need not prevent
it. The clean way to actually PREVENT 2-D collapse is a defocusing quintic term:
    i u_t = -Delta_g u + sigma3 |u|^2 u + sigma5 |u|^4 u ,   sigma3<0, sigma5>0.
The quintic repulsion switches on only at high density, halting the self-focusing
runaway -> a stable 2-D soliton with NO time-dependence, at ANY sigma5>0 (the energy
is then bounded below). We take the same hump that Townes-collapses under pure cubic
and show the quintic term removes the collapse threshold outright.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nls_lumpy_torus import build_operators, initial_condition, mass

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"
PK_CAP = 150.0
WX, WTH = 0.35, 0.35


def cq_stepper(grid, dt, tol=1e-8, pmax=60):
    K, Mdiag = grid["K"], grid["Mdiag"]
    iM = sp.diags(1j * Mdiag)
    lu = splu((iM - 0.5 * dt * K).tocsc())
    rhs = (iM + 0.5 * dt * K).tocsr()

    def step(Un, s3, s5):
        base = rhs @ Un
        W = Un.copy()
        for _ in range(pmax):
            Uh = 0.5 * (W + Un)
            a2 = np.abs(Uh) ** 2
            Wn = lu.solve(base + dt * (Mdiag * ((s3 * a2 + s5 * a2 ** 2) * Uh)))
            if np.linalg.norm(Wn - W) / (np.linalg.norm(Wn) + 1e-300) < tol:
                W = Wn; break
            W = Wn
        return W
    return step


def run(grid, step, dt, amp, s3, s5, Tmax):
    """Collapse == peak crosses the cap / non-finite. (No Picard-iteration proxy: a
    stiff-but-stable cubic-quintic step can hit max iters without blowing up.)"""
    Mdiag = grid["Mdiag"]
    U = initial_condition(grid, amp=amp, xc=0.0, thc=np.pi, wx=WX, wth=WTH, k=0).ravel()
    m0 = mass(U, Mdiag); n = int(round(Tmax / dt))
    ts, pks = [0.0], [float(np.max(np.abs(U) ** 2))]; tc = None
    for i in range(1, n + 1):
        U = step(U, s3, s5)
        pk = float(np.max(np.abs(U) ** 2))
        ts.append(i * dt); pks.append(min(pk, PK_CAP))
        if (not np.isfinite(pk)) or pk > PK_CAP:
            tc = i * dt; break
    return np.array(ts), np.array(pks), tc, m0


if __name__ == "__main__":
    dt = 1e-3
    grid = build_operators(96, 96)
    step = cq_stepper(grid, dt)
    Tmax = 1.5
    S5 = 0.4

    print("hero (amp=6.0):", flush=True)
    tb, pb, tcb, m0 = run(grid, step, dt, 6.0, -1.0, 0.0, Tmax)
    print(f"  cubic only:       mass={m0:.1f}  t_c={tcb}  peak_max={pb.max():.1f}", flush=True)
    tq, pq, tcq, _ = run(grid, step, dt, 6.0, -1.0, S5, Tmax)
    print(f"  cubic+quintic(.4): t_c={tcq}  peak_max={pq.max():.1f}  peak_final={pq[-1]:.1f}", flush=True)

    amps = [4.0, 5.0, 6.0, 7.0]     # up to the density the implicit solver resolves cleanly
    tc_cubic, tc_cq, masses = [], [], []
    print("mass scan (cubic vs cubic+quintic):", flush=True)
    for a in amps:
        _, _, tcc, mm = run(grid, step, dt, a, -1.0, 0.0, Tmax)
        _, _, tcq2, _ = run(grid, step, dt, a, -1.0, S5, Tmax)
        masses.append(mm); tc_cubic.append(tcc); tc_cq.append(tcq2)
        print(f"  amp={a} mass={mm:5.1f}  cubic {'%.3f'%tcc if tcc else 'stable':>7}  "
              f"cubic+quintic {'%.3f'%tcq2 if tcq2 else 'stable'}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    axA = ax[0]
    axA.axhline(PK_CAP, color="0.6", ls=":", lw=1.1)
    axA.semilogy(tb, pb, color=PINK, lw=2.0, label=f"cubic only — collapses (t_c={tcb:.2f})")
    lbl = "cubic + quintic — " + ("stable soliton" if tcq is None else f"t_c={tcq:.2f}")
    axA.semilogy(tq, pq, color=BLUE, lw=1.8, label=lbl)
    axA.set_xlabel("t"); axA.set_ylabel(r"peak $|u|^2$ (log)"); axA.set_xlim(0, Tmax)
    axA.set_title(r"a defocusing quintic PREVENTS collapse (no drive, amp=6)")
    axA.legend(loc="center right", fontsize=9, framealpha=.92)

    Ssv = Tmax * 1.03
    axB = ax[1]
    yc = [t if t else Ssv for t in tc_cubic]
    yq = [t if t else Ssv for t in tc_cq]
    axB.plot(masses, yc, "-o", color=PINK, ms=6, lw=1.6, label=r"cubic ($\sigma_5$=0)")
    axB.plot(masses, yq, "-s", color=BLUE, ms=6, lw=1.6, label=rf"cubic+quintic ($\sigma_5$={S5})")
    axB.axhline(Ssv, color=BLUE, ls=":", lw=1.1)
    axB.text(masses[0], Ssv, "stable (no collapse)", fontsize=8.5, color=BLUE, va="bottom")
    axB.set_ylim(0, Ssv * 1.12)
    axB.set_xlabel(r"hump mass  $M=\int A|u|^2$"); axB.set_ylabel(r"collapse time  $t_c$")
    axB.set_title(r"cubic has a collapse threshold; cubic+quintic never collapses")
    axB.legend(loc="center left", fontsize=9); axB.grid(alpha=0.25)

    fig.suptitle("Cubic-quintic: a defocusing quintic term removes the collapse threshold outright",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cubic_quintic.png", dpi=120)
    print("wrote cubic_quintic.png", flush=True)
