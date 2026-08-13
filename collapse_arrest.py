"""
Frontier prototype 7: ARRESTING the collapse (nonlinearity management, done right).

The dynamical-control run (dynamical_stabilization.py) showed a time-varying geometry
only *delays* the azimuthal necklace collapse -- because that collapse is a fast
symmetry-breaking of an extended ring. Here we take the collapse management is meant
for: a single, localized 2-D hump on the belly, the mass-critical (Townes) collapse.
Just above the critical mass the collapse is slow, so a fast, strong breathing of the
belly -- g_eff(t) = g0(1 + delta cos Omega t) with mean unchanged -- has time to act,
and it turns the collapsing hump into a stable breather: the blow-up is ARRESTED
(Saito-Ueda / Feshbach-resonance management; the belly's curvature + trap help where
pure 2-D cubic management only delays).

Hero runs are saved to collapse_arrest_snaps.npz for the side-by-side animation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamical_stabilization import managed_stepper
from nls_lumpy_torus import build_operators, initial_condition, mass

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"
PK_CAP = 150.0
AMP, WX, WTH = 5.8, 0.35, 0.35        # localized hump, mass ~12.8 (just supercritical)
OM, DELTA = 100.0, 0.8                 # in-window fast strong drive that arrests


def run(grid, step, dt, amp, delta, Om, Tmax, seed_amp=0.0, snap_every=None):
    Mdiag = grid["Mdiag"]
    U = initial_condition(grid, amp=amp, xc=0.0, thc=np.pi, wx=WX, wth=WTH, k=0).ravel()
    m0 = mass(U, Mdiag)
    n = int(round(Tmax / dt))
    ts, pks = [0.0], [float(np.max(np.abs(U) ** 2))]
    snaps_t, snaps_U = [0.0], [U.copy()]
    tc = None
    for i in range(1, n + 1):
        sig = -(1.0 + delta * np.cos(Om * (i - 0.5) * dt))
        U, it = step(U, sig)
        pk = float(np.max(np.abs(U) ** 2))
        ts.append(i * dt); pks.append(min(pk, PK_CAP))
        if snap_every and i % snap_every == 0:
            snaps_t.append(i * dt); snaps_U.append(U.copy())
        if (not np.isfinite(pk)) or pk > PK_CAP or it >= 59:
            tc = i * dt
            snaps_t.append(i * dt); snaps_U.append(U.copy())
            break
    return np.array(ts), np.array(pks), tc, m0, np.array(snaps_t), snaps_U


if __name__ == "__main__":
    # ------------------------------------------------ hero (fine dt, save snapshots)
    dt_h = 5e-4
    grid = build_operators(128, 128)
    step_h = managed_stepper(grid, dt_h)
    snap_every = 30                      # every 0.015 in time
    print("hero (amp=5.8):", flush=True)
    tb, pb, tcb, m0, sbt, sbU = run(grid, step_h, dt_h, AMP, 0.0, OM, 3.0, snap_every=snap_every)
    print(f"  bare: mass={m0:.1f}  t_c={tcb}  peak_max={pb.max():.1f}", flush=True)
    tm, pm, tcm, _, smt, smU = run(grid, step_h, dt_h, AMP, DELTA, OM, 3.0, snap_every=snap_every)
    print(f"  managed(d=0.8,Om=100): t_c={tcm}  peak_max={pm.max():.1f}  peak_final={pm[-1]:.1f}", flush=True)

    np.savez(f"{OUT}/collapse_arrest_snaps.npz",
             Nx=grid["Nx"], Nth=grid["Nth"], x=grid["x"], th=grid["th"],
             bare_t=sbt, bare_U=np.array(sbU), bare_tc=tcb if tcb else -1,
             man_t=smt, man_U=np.array(smU), man_tc=tcm if tcm else -1,
             amp=AMP, delta=DELTA, Om=OM)
    print("  saved collapse_arrest_snaps.npz", flush=True)

    # ------------------------------------------------ scans (coarser dt, faster)
    dt_s = 1e-3
    step_s = managed_stepper(grid, dt_s)
    deltas = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    tc_d = []
    print("depth scan (Om=100):", flush=True)
    for d in deltas:
        _, _, tc, _, _, _ = run(grid, step_s, dt_s, AMP, d, OM, 1.5)
        tc_d.append(tc)
        print(f"  d={d:.1f}  {'t_c=%.3f' % tc if tc else 'SURVIVES'}", flush=True)

    Omegas = [30.0, 50.0, 70.0, 100.0, 150.0, 220.0]
    tc_o = []
    print("freq scan (delta=0.8):", flush=True)
    for om in Omegas:
        _, _, tc, _, _, _ = run(grid, step_s, dt_s, AMP, DELTA, om, 1.5)
        tc_o.append(tc)
        print(f"  Om={om:4.0f}  {'t_c=%.3f' % tc if tc else 'SURVIVES'}", flush=True)

    # ------------------------------------------------------------------- figure
    Tsurv = 1.5
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    axA = ax[0]
    axA.axhline(PK_CAP, color="0.6", ls=":", lw=1.1)
    axA.text(0.05, PK_CAP * 0.86, "collapse guard", fontsize=8, color="0.4")
    axA.semilogy(tb, pb, color=PINK, lw=2.0, label=f"bare — collapses (t_c={tcb:.2f})")
    lblm = "managed δ=0.8, Ω=100 — " + ("stable breather, arrested" if tcm is None else f"t_c={tcm:.2f}")
    axA.semilogy(tm, pm, color=BLUE, lw=1.7, label=lblm)
    axA.set_xlim(0, 3.0); axA.set_xlabel("t"); axA.set_ylabel(r"peak $|u|^2$  (log)")
    axA.set_title("localized hump: bare collapses, managed breathes")
    axA.legend(loc="upper right", fontsize=8.5, framealpha=.92)

    axB = ax[1]
    yb = [t if t else Tsurv * 1.03 for t in tc_d]
    axB.plot(deltas, yb, "-o", color=PINK, ms=6, lw=1.6, zorder=3)
    for d, y, t in zip(deltas, yb, tc_d):
        if t is None:
            axB.plot(d, y, "s", color=BLUE, ms=10, zorder=4)
    axB.axhline(Tsurv * 1.03, color=BLUE, ls=":", lw=1.1)
    axB.text(0.02, Tsurv * 1.03, "survives (arrested)", fontsize=8.4, color=BLUE, va="bottom")
    axB.set_xlabel(r"modulation depth  $\delta$"); axB.set_ylabel(r"collapse time  $t_c$")
    axB.set_title(r"deep enough breathing arrests it ($\Omega$=100)")
    axB.grid(alpha=0.25)

    axC = ax[2]
    yc = [t if t else Tsurv * 1.03 for t in tc_o]
    axC.plot(Omegas, yc, "-o", color=GOLD, ms=6, lw=1.6, zorder=3)
    for om, y, t in zip(Omegas, yc, tc_o):
        if t is None:
            axC.plot(om, y, "s", color=BLUE, ms=10, zorder=4)
    axC.axhline(Tsurv * 1.03, color=BLUE, ls=":", lw=1.1)
    axC.text(Omegas[0], Tsurv * 1.03, "survives (arrested)", fontsize=8.4, color=BLUE, va="bottom")
    axC.set_xlabel(r"modulation frequency  $\Omega$"); axC.set_ylabel(r"collapse time  $t_c$")
    axC.set_title(r"fast enough to average $\to$ arrested ($\delta$=0.8)")
    axC.grid(alpha=0.25)

    fig.suptitle("Arresting the collapse: fast strong breathing turns the Townes-collapsing hump "
                 "into a stable breather", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/collapse_arrest.png", dpi=120)
    print("wrote collapse_arrest.png", flush=True)
