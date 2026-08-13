"""Declarative experiments over the shared solver. Each returns a structured Result

    {experiment, params, metrics, verification, provenance, series, summary}

so a human — or an agent — reads the NUMBERS and the TRUST SIGNALS, not a narrative.
Register new experiments with @register; run one with run(name, **params).
"""
import subprocess
import numpy as np

from .operators import build_surface, ring_grid
from .steppers import CNStepper, RingStepper
from . import fields
from . import diagnostics as diag
from . import bogoliubov

REGISTRY = {}


def register(fn):
    REGISTRY[fn.__name__] = fn
    return fn


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=__file__.rsplit("/", 2)[0], stderr=subprocess.DEVNULL
                                       ).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ experiments
@register
def collapse(amp=6.0, eps=1.0, sigma5=0.0, drive_delta=0.0, drive_omega=40.0,
             Nx=96, Nth=96, dt=1e-3, T=1.5, pk_cap=150.0):
    """Localized hump under focusing cubic (+ optional quintic sigma5, + optional
    breathing drive g_eff=g0(1+delta cos Omega t)). Covers bare collapse, nonlinearity
    management (drive_delta>0), and genuine cubic-quintic arrest (sigma5>0) — one
    experiment, three regimes."""
    surf = build_surface(Nx, Nth, eps=eps)
    step = CNStepper(surf, dt, sigma5=sigma5)
    U = fields.localized_hump(surf, amp=amp, k=0)
    M, K = surf["Mdiag"], surf["K"]
    m0 = diag.mass(U, M)
    hm = [m0]; he = [diag.energy(U, K, M, -1.0, sigma5=sigma5)]
    pk_max = diag.peak(U); tc = None
    n = int(round(T / dt))
    for i in range(1, n + 1):
        s3 = -(1.0 + drive_delta * np.cos(drive_omega * (i - 0.5) * dt))
        U = step.step(U, sigma3=s3)
        pk = diag.peak(U); pk_max = max(pk_max, pk)
        hm.append(diag.mass(U, M)); he.append(diag.energy(U, K, M, -1.0, sigma5=sigma5))
        if (not np.isfinite(pk)) or pk > pk_cap:
            tc = i * dt; break
    collapsed = tc is not None
    cons = diag.conservation_drift(hm, he)
    ver = {"mass_drift": cons["mass_drift"], "mass_conserved": cons["mass_conserved"]}
    if drive_delta == 0.0:  # energy only conserved for the autonomous (undriven) run
        ver.update({k: cons[k] for k in ("energy_drift", "energy_conserved")})
    metrics = {"mass": m0, "collapsed": collapsed,
               "collapse_time": (tc if collapsed else None),
               "peak_max": pk_max, "peak_final": diag.peak(U)}
    summary = (f"mass {m0:.1f}: " + (f"COLLAPSED at t={tc:.3f}" if collapsed
               else f"stable to T={T} (peak settled to {diag.peak(U):.1f})"))
    return dict(metrics=metrics, verification=ver,
                series={"mass_hist": hm}, summary=summary)


@register
def expanding_cosmos(a_i=1.0, a_f=3.0, tau_c=0.10, t_q=1.0, kmax=34, c=1.0, T=3.0, Nt=30000):
    """Expanding-ring analog cosmology: quench the scale factor, integrate the
    Bogoliubov modes, read the |beta_k|^2 plateau and Sakharov oscillations."""
    ks = np.arange(1, kmax + 1)

    def a_of(t):
        return a_i + (a_f - a_i) * 0.5 * (1 + np.tanh((t - t_q) / tau_c))

    def omega_of_t(t):
        return c * ks / a_of(t)
    tg = np.linspace(0.0, T, Nt)
    r = bogoliubov.integrate_modes(ks, omega_of_t, tg, c=c, a_i=a_i, a_f=a_f)
    plateau = (a_f - a_i) ** 2 / (4 * a_i * a_f)
    metrics = {"plateau_analytic": plateau, "beta2_k1": float(r["beta2"][0]),
               "beta2_kmax": float(r["beta2"][-1]),
               "Sk_min": float(r["S_k"].min()), "Sk_max": float(r["S_k"].max()),
               "sakharov_contrast": float(r["S_k"].max() - r["S_k"].min())}
    ver = {"wronskian_max_dev": r["wronskian_max_dev"],
           "wronskian_ok": r["wronskian_max_dev"] < 1e-8,
           "plateau_match": abs(r["beta2"][0] - plateau) / plateau < 0.05}
    summary = (f"|beta_k=1|^2={r['beta2'][0]:.3f} vs plateau {plateau:.3f}; "
               f"S_k oscillates {r['S_k'].min():.2f}-{r['S_k'].max():.2f} (Sakharov)")
    return dict(metrics=metrics, verification=ver,
                series={"ks": ks.tolist(), "beta2": r["beta2"].tolist(),
                        "S_k": r["S_k"].tolist()}, summary=summary)


@register
def rogue_ring(a0=1.0, g=8.0, q_seed=1, Nth=256, dt=2e-4, T=5.0):
    """Focusing belly-ring MI -> Akhmediev/Peregrine breather; report the peak
    amplitude vs the Peregrine 3x bound."""
    ring = ring_grid(Nth, radius=1.0)
    step = RingStepper(ring, dt)
    u = fields.ring_seeded(ring, a0=a0, q_seed=q_seed, eps_seed=1e-3)
    m0 = float(np.mean(np.abs(u) ** 2))
    n = int(round(T / dt)); peak_amp = 0.0
    for i in range(n):
        u = step.step(u, g=g)
        peak_amp = max(peak_amp, float(np.max(np.abs(u))))
    mass_drift = abs(np.mean(np.abs(u) ** 2) - m0) / m0
    metrics = {"peak_over_background": peak_amp / a0, "peregrine_bound": 3.0}
    ver = {"mass_drift": float(mass_drift), "mass_conserved": mass_drift < 1e-3,
           "below_peregrine_bound": peak_amp / a0 <= 3.05}
    summary = f"rogue peak {peak_amp/a0:.3f}x background (Peregrine limit 3.0)"
    return dict(metrics=metrics, verification=ver, series={}, summary=summary)


@register
def solver_bench(amp=8.0, sigma5=0.0, Nx=64, Nth=64, dt=1e-3, nsteps=12, pmax=40):
    """Newton vs Picard for the implicit nonlinear solve on a concentrating hump.
    Reports iterations/step for each and their step agreement (both solve the same
    equation -> a converged step is method-independent). Newton wins near strong
    concentration, where Picard's linear convergence slows or stalls."""
    surf = build_surface(Nx, Nth)
    step = CNStepper(surf, dt, sigma5=sigma5, tol=1e-11, pmax=pmax)
    U0 = fields.localized_hump(surf, amp=amp)
    Wp = step.step(U0, -1.0, method="picard")
    Wn = step.step(U0, -1.0, method="newton")
    agree = float(np.linalg.norm(Wp - Wn))
    stats = {}
    for method in ("picard", "newton"):
        U = fields.localized_hump(surf, amp=amp); its = []
        for _ in range(nsteps):
            io = []; U = step.step(U, -1.0, method=method, n_iter_out=io); its.append(io[0])
        stats[method] = (int(max(its)), float(np.mean(its)))
    ratio = stats["picard"][1] / max(stats["newton"][1], 1e-9)
    metrics = {"picard_max_iters": stats["picard"][0], "picard_mean_iters": stats["picard"][1],
               "newton_max_iters": stats["newton"][0], "newton_mean_iters": stats["newton"][1],
               "iter_ratio": ratio}
    ver = {"methods_agree": agree < 1e-8, "step_diff": agree,
           "newton_converged": stats["newton"][0] < pmax,
           "picard_converged": stats["picard"][0] < pmax}
    summary = (f"Newton {stats['newton'][1]:.1f} vs Picard {stats['picard'][1]:.1f} iters/step "
               f"({ratio:.1f}x fewer); step agreement {agree:.1e}")
    return dict(metrics=metrics, verification=ver, series={}, summary=summary)


@register
def thouless_pump(J=1.0, delta=0.7, mass=0.9, Nk=400):
    """Adiabatic (Thouless) pump on a Rice-Mele sliding lump lattice: the filled band's
    Wannier centre winds by an integer per cycle (the Chern number). A topological loop
    pumps 1; a trivial loop pumps 0."""
    def h(k, s, enc):
        u = delta * np.cos(s)
        v = mass * np.sin(s) if enc else mass * (0.5 + 0.5 * np.cos(s))
        J1, J2 = J + u, J - u
        return np.array([[v, (J1 + J2 * np.cos(k)) - 1j * (J2 * np.sin(k))],
                         [(J1 + J2 * np.cos(k)) + 1j * (J2 * np.sin(k)), -v]])

    def wcentre(s, enc):
        ks = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
        low = [np.linalg.eigh(h(k, s, enc))[1][:, 0] for k in ks]
        prod = 1.0 + 0j
        for j in range(Nk):
            prod *= np.vdot(low[j], low[(j + 1) % Nk])
        return (-np.angle(prod) / (2 * np.pi)) % 1.0

    ss = np.linspace(0, 2 * np.pi, 240)
    topo = np.unwrap([2 * np.pi * wcentre(s, True) for s in ss]) / (2 * np.pi)
    triv = np.unwrap([2 * np.pi * wcentre(s, False) for s in ss]) / (2 * np.pi)
    pump_t = float(topo[-1] - topo[0]); pump_v = float(triv[-1] - triv[0])
    metrics = {"pumped_charge": pump_t, "pumped_charge_trivial": pump_v,
               "chern": int(round(pump_t))}
    ver = {"quantized": abs(pump_t - round(pump_t)) < 0.03 and abs(pump_v) < 0.03}
    summary = f"topological loop pumps {pump_t:.3f}/cycle; trivial loop {pump_v:.3f} (quantized)"
    return dict(metrics=metrics, verification=ver, series={}, summary=summary)


def _clean(obj):
    """Recursively convert numpy scalars/bools/arrays to native Python so the Result
    is JSON-serialisable (the agent tool speaks JSON)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@register
def kz_freeze(tau_Q=0.4, g_f=40.0, a0=1.0, radius=10.0, Nth=1024, dt=2e-4,
              nseed=5, sat=1.35):
    """One Kibble-Zurek quench point: ramp the ring coupling 0->g_f over tau_Q and
    read the frozen wavenumber k* (band-limited to the MI band, seed-averaged). Sweep
    tau_Q to get the KZ power law k* ~ tau_Q^{-b}."""
    ring = ring_grid(Nth, radius=radius)
    step = RingStepper(ring, dt)
    qband = 1.3 * np.sqrt(g_f)
    ks = []
    for s in range(nseed):
        u = fields.uniform_noise(ring, a0=a0, noise=3e-3, seed=s)
        n = int((tau_Q + 8.0) / dt)
        for i in range(n):
            g = g_f * min(i * dt / tau_Q, 1.0)
            u = step.step(u, g)
            if np.max(np.abs(u)) > sat * a0:
                ks.append(diag.ring_dominant_wavenumber(u, ring, band=qband))
                break
    ks = np.array([k for k in ks if np.isfinite(k) and k > 0])
    kbar = float(ks.mean()) if len(ks) else float("nan")
    metrics = {"tau_Q": tau_Q, "kstar_mean": kbar,
               "kstar_std": float(ks.std()) if len(ks) else float("nan")}
    ver = {"n_seeds": int(len(ks)), "seeds_ok": len(ks) >= max(3, nseed - 1),
           "kstar_in_band": bool(0 < kbar <= qband)}
    return dict(metrics=metrics, verification=ver, series={},
                summary=f"tau_Q={tau_Q}: k*={kbar:.2f} (band-limited, {len(ks)} seeds)")


def run(name, **params):
    if name not in REGISTRY:
        raise KeyError(f"unknown experiment '{name}'. Available: {sorted(REGISTRY)}")
    out = REGISTRY[name](**params)
    out.update(experiment=name, params=params,
               provenance={"solver": "nls_torus", "git_commit": _git_commit(),
                           "note": "reduced mean-field model; results need validation vs full physics"})
    return _clean(out)
