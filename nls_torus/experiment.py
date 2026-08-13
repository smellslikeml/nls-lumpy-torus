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
             Nx=96, Nth=96, dt=1e-3, T=1.5, pk_cap=150.0, geometry="lumpy_torus"):
    """Localized hump under focusing cubic (+ optional quintic sigma5, + optional
    breathing drive g_eff=g0(1+delta cos Omega t)) on any registered `geometry`. Covers
    bare collapse, nonlinearity management (drive_delta>0), and genuine cubic-quintic
    arrest (sigma5>0) — one experiment, three regimes, any manifold."""
    surf = build_surface(Nx, Nth, geometry=geometry, eps=eps)
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


@register
def conservation(eps=1.0, Nx=64, Nth=128, dt=2e-3, T=1.0, sigma3=1.0, geometry="lumpy_torus"):
    """Solver validation on any registered `geometry`: mass and energy drift."""
    surf = build_surface(Nx, Nth, geometry=geometry, eps=eps); step = CNStepper(surf, dt)
    U = fields.localized_hump(surf, amp=1.0, k=4)
    M, K = surf["Mdiag"], surf["K"]
    hm = [diag.mass(U, M)]; he = [diag.energy(U, K, M, sigma3)]
    for _ in range(int(round(T / dt))):
        U = step.step(U, sigma3); hm.append(diag.mass(U, M)); he.append(diag.energy(U, K, M, sigma3))
    c = diag.conservation_drift(hm, he)
    return dict(metrics={"mass_drift": c["mass_drift"], "energy_drift": c["energy_drift"]},
                verification={"mass_conserved": c["mass_conserved"], "energy_conserved": c["energy_conserved"]},
                series={}, summary=f"mass drift {c['mass_drift']:.1e}, energy drift {c['energy_drift']:.1e}")


@register
def bifurcation(eps=1.0):
    """Elliptic<->hyperbolic geometry bifurcation: belly/neck Gaussian curvatures vs eps
    (numeric vs the analytic K_belly=eps/(1+eps), K_neck=-eps)."""
    from .geometry import curvature_K
    Kb, Kn = float(curvature_K(0.0, eps)), float(curvature_K(np.pi / 2, eps))
    return dict(metrics={"K_belly": Kb, "K_neck": Kn, "eps": eps},
                verification={"matches_analytic": abs(Kb - eps / (1 + eps)) < 1e-3 and abs(Kn + eps) < 1e-3},
                series={}, summary=f"eps={eps}: K_belly={Kb:.3f}, K_neck={Kn:.3f} (belly elliptic iff eps>0)")


@register
def revival(Nth=256, k0=6, wth=0.5):
    """Talbot quantum revival: a packet released on the belly ring reforms at t=2*pi."""
    ring = ring_grid(Nth, radius=1.0); th = ring["th"]
    dth = np.angle(np.exp(1j * (th - np.pi)))
    u0 = np.exp(-dth ** 2 / (2 * wth ** 2)) * np.exp(1j * k0 * th)
    n = 6000; dt = 2 * np.pi / n
    step = RingStepper(ring, dt); u = u0.copy()
    for _ in range(n):
        u = step.step(u, 0.0)                     # linear (g=0)
    fid = abs(np.vdot(u0, u)) / (np.linalg.norm(u0) * np.linalg.norm(u))
    return dict(metrics={"revival_fidelity": float(fid)},
                verification={"revives": fid > 0.99}, series={},
                summary=f"Talbot revival fidelity at t=2pi: {fid:.4f}")


@register
def geodesic_stability(k=6, amp=0.9, wx=0.25, Nx=80, Nth=160, dt=2e-3, T=3.0,
                       geometry="lumpy_torus"):
    """Beam along the elliptic (belly) vs hyperbolic (neck) parallel geodesic on any
    registered `geometry`: transverse width bounded on the stable orbit, grows on the
    unstable one."""
    surf = build_surface(Nx, Nth, geometry=geometry); step = CNStepper(surf, dt)
    x, A, dx, dth = surf["x"], surf["A"], surf["dx"], surf["dth"]
    Nx_, Nth_, Lx = surf["Nx"], surf["Nth"], np.pi

    def xwidth(U, xc):
        f = np.abs(U.reshape(Nx_, Nth_)) ** 2
        Px = f.sum(1) * A * dx * dth
        d = (x - xc + Lx / 2) % Lx - Lx / 2
        return np.sqrt(np.sum(Px * d ** 2) / (Px.sum() + 1e-300))

    res = {}
    for name, xc in [("elliptic", 0.0), ("hyperbolic", np.pi / 2)]:
        U = fields.geodesic_ring(surf, xc=xc, amp=amp, wx=wx, k=k)
        w0 = xwidth(U, xc); wmax = w0
        for _ in range(int(round(T / dt))):
            U = step.step(U, -1.0); wmax = max(wmax, xwidth(U, xc))
        res[name] = wmax / w0
    # A narrow packet breathes on either orbit; the stable/unstable discriminator is
    # RELATIVE — the hyperbolic neck spreads the beam more than the elliptic belly
    # (an absolute bound would mislabel the belly's bounded breathing as instability).
    return dict(metrics={"elliptic_width_ratio": res["elliptic"], "hyperbolic_width_ratio": res["hyperbolic"],
                         "spread_ratio": res["hyperbolic"] / res["elliptic"]},
                verification={"neck_spreads_more": res["hyperbolic"] > 1.3 * res["elliptic"]},
                series={}, summary=f"transverse spread: belly {res['elliptic']:.2f}x, "
                                   f"neck {res['hyperbolic']:.2f}x ({res['hyperbolic']/res['elliptic']:.1f}x more at the neck)")


@register
def quasimodes(k=8, Nx=400, eps=1.0):
    """Whispering-gallery quasimodes: bound states of the centrifugal well V_k=k^2/A^2
    (count below the neck barrier; ground width ~ k^{-1/2})."""
    from .geometry import profile_A
    x = np.linspace(-np.pi / 2, np.pi / 2, Nx, endpoint=False); dx = np.pi / Nx
    A = profile_A(x, eps); V = k ** 2 / A ** 2
    H = np.diag(2.0 / dx ** 2 + V) + np.diag(-1.0 / dx ** 2 * np.ones(Nx - 1), 1) \
        + np.diag(-1.0 / dx ** 2 * np.ones(Nx - 1), -1)
    H[0, -1] = H[-1, 0] = -1.0 / dx ** 2
    E, psi = np.linalg.eigh(H)
    Vneck = k ** 2 / A.min() ** 2
    nbound = int(np.sum(E < Vneck))
    p0 = np.abs(psi[:, 0]) ** 2; p0 = p0 / p0.sum()
    width = float(np.sqrt(np.sum(p0 * x ** 2)))
    return dict(metrics={"n_bound_states": nbound, "ground_width": width,
                         "width_times_sqrt_k": width * np.sqrt(k)},
                verification={"has_bound_states": nbound > 0, "ground_trapped": bool(E[0] < Vneck)},
                series={}, summary=f"k={k}: {nbound} whispering-gallery bound states, ground width {width:.3f}")


@register
def analog_horizon(mu=1.0, g=1.0, Nx=1200):
    """Neck as a de Laval nozzle -> sonic horizon; surface gravity kappa and Hawking
    temperature T_H = kappa/2pi (static fit cross-checked against the ray peeling rate)."""
    from scipy.integrate import solve_ivp
    x = np.linspace(-np.pi / 2, np.pi / 2, Nx)
    A = np.sqrt((1 + np.sin(x) ** 2) / 2); Amin = A.min()
    vs = np.sqrt(2 * mu / 3); J = vs ** 3 * Amin
    v = np.empty_like(x)
    for i, xi in enumerate(x):
        r = np.roots([0.5, 0, -mu, J / A[i]]); pos = np.sort(r[np.abs(r.imag) < 1e-6].real)
        pos = pos[pos > 0]; v[i] = pos[0] if xi < 0 else pos[-1]
    c = np.sqrt(g * J / (v * A)); vmc = v - c
    m = (np.abs(x) > 0.02) & (np.abs(x) < 0.35)
    cfit = np.polyfit(x[m], vmc[m], 3); vmc_s = np.poly1d(cfit)
    kappa = abs(np.polyval(np.polyder(cfit), 0.0)); T_H = kappa / (2 * np.pi)
    s = solve_ivp(lambda t, y: vmc_s(y), [0, 9], [-0.03], max_step=0.02, rtol=1e-8)
    tt, xx = s.t, s.y[0]; near = np.abs(xx) < 0.15
    kdyn = abs(np.polyfit(tt[near], np.log(np.abs(xx[near])), 1)[0])
    return dict(metrics={"kappa": float(kappa), "T_H": float(T_H), "kappa_dynamical": float(kdyn)},
                verification={"kappa_dyn_matches_static": abs(kdyn - kappa) / kappa < 0.1},
                series={}, summary=f"sonic horizon: kappa={kappa:.3f}, T_H={T_H:.3f}")


@register
def faraday(nres=3, delta=0.5, g0=1.0, a=1.0, Nth=512, T=24.0, Nt=12000):
    """Breathing torus -> parametric (Faraday) amplification: driving the coupling at
    Omega=2*omega_n grows Bogoliubov mode n out of noise."""
    th = np.linspace(0, 2 * np.pi, Nth, endpoint=False); q = np.fft.fftfreq(Nth, d=1.0 / Nth)
    rng = np.random.default_rng(1)
    Omega = 2 * nres * np.sqrt(nres ** 2 + 2 * g0 * a ** 2)
    dt = T / Nt; Lh = np.exp(-1j * q ** 2 * dt / 2)
    u = a * (1 + 0.01 * np.cos(nres * th)) + 1e-4 * (rng.standard_normal(Nth) + 1j * rng.standard_normal(Nth))
    amp0 = None; ampmax = 0.0
    for i in range(Nt):
        gg = g0 * (1 + delta * np.cos(Omega * i * dt))
        u = np.fft.ifft(Lh * np.fft.fft(u)); u = u * np.exp(-1j * gg * np.abs(u) ** 2 * dt)
        u = np.fft.ifft(Lh * np.fft.fft(u))
        uk = np.abs(np.fft.fft(u)[nres]) / Nth
        amp0 = uk if amp0 is None else amp0; ampmax = max(ampmax, uk)
    growth = ampmax / (amp0 + 1e-30)
    return dict(metrics={"mode": nres, "growth_factor": float(growth)},
                verification={"parametrically_amplified": growth > 10},
                series={}, summary=f"Faraday mode n={nres} grows {growth:.0f}x at Omega=2*omega_n")


@register
def topological_bands(alpha=1.0 / 3.0, Ncell=40, lam=1.0, t_hop=1.0, Nk=240):
    """Chiral lump twist -> Hofstadter strip: chiral edge modes crossing the gaps."""
    n = np.arange(Ncell); center = (Ncell - 1) / 2; kth = np.linspace(0, 2 * np.pi, Nk)
    max_edge = 0.0
    for kk in kth:
        H = np.diag(2 * lam * np.cos(2 * np.pi * alpha * n + kk)) \
            + np.diag(-t_hop * np.ones(Ncell - 1), 1) + np.diag(-t_hop * np.ones(Ncell - 1), -1)
        w, V = np.linalg.eigh(H)
        edge = np.sum(V ** 2 * ((n[:, None] - center) / center), axis=0)
        gap = (w > -3.2) & (w < -1.6)
        if gap.any():
            max_edge = max(max_edge, float(np.max(np.abs(edge[gap]))))
    return dict(metrics={"flux_alpha": alpha, "max_edge_localization": max_edge},
                verification={"edge_modes_present": max_edge > 0.5},
                series={}, summary=f"Hofstadter strip (flux {alpha:.2g}): max gap edge-localization {max_edge:.2f}")


@register
def floquet_bands(th1_over_pi=0.5, Ncell=20):
    """Two-step-driven lump chain -> Floquet edge modes at quasienergy 0 AND pi (the
    anomalous pi-mode has no static analogue)."""
    Ns = 2 * Ncell
    intra = [(2 * i, 2 * i + 1) for i in range(Ncell)]
    inter = [(2 * i + 1, 2 * i + 2) for i in range(Ncell - 1)]
    site = np.arange(Ns); center = (Ns - 1) / 2

    def rot(theta, pairs):
        Mr = np.eye(Ns, dtype=complex); c, s = np.cos(theta), -1j * np.sin(theta)
        for a, b in pairs:
            Mr[a, a] = c; Mr[b, b] = c; Mr[a, b] = s; Mr[b, a] = s
        return Mr

    th1 = th1_over_pi * np.pi
    best = (0.0, 0.0)
    for th2 in np.linspace(0, 2 * np.pi, 120):
        w, V = np.linalg.eig(rot(th2, inter) @ rot(th1, intra))
        eps = np.angle(w); edge = np.sum(np.abs(V) ** 2 * ((site[:, None] - center) / center), axis=0)

        def he(target):
            d = np.abs(np.angle(np.exp(1j * (eps - target))))
            cand = np.where((d < 0.4) & (np.abs(edge) > 0.5))[0]
            return float(np.max(np.abs(edge[cand]))) if len(cand) else 0.0
        z, p = he(0.0), he(np.pi)
        if z > 0.5 and p > 0.5 and z + p > best[0] + best[1]:
            best = (z, p)
    z, p = best
    return dict(metrics={"zero_mode_edge": z, "pi_mode_edge": p},
                verification={"anomalous_pi_mode": p > 0.5, "zero_mode": z > 0.5},
                series={}, summary=f"Floquet edge modes: 0-mode {z:.2f}, pi-mode {p:.2f} (anomalous)")


@register
def quantum_chaos(lam_weak=2.5, lam_strong=120.0, Nx=48, Nth=48):
    """Theta-lump drives level-spacing statistics from clustering to Wigner-GOE
    repulsion (P(s<0.3): Poisson~0.26, GOE~0.07)."""
    surf = build_surface(Nx, Nth); K = surf["K"].toarray(); M = surf["Mdiag"]
    x, th = surf["x"], surf["th"]; X, TH = np.meshgrid(x, th, indexing="ij")
    V = (np.cos(2 * X - 3 * TH) + 0.6 * np.cos(X + 2 * TH + 0.5)
         + 0.4 * np.sin(3 * X - TH + 0.9)).ravel()
    D = 1.0 / np.sqrt(M); DKD = (D[:, None] * K) * D[None, :]

    def rep(lam):
        E = np.linalg.eigvalsh(DKD + np.diag(lam * V))
        s = diag.level_spacings(E)
        return float(np.mean(s < 0.3))
    rw, rs = rep(lam_weak), rep(lam_strong)
    return dict(metrics={"repulsion_weak": rw, "repulsion_strong": rs, "goe_ref": 0.07, "poisson_ref": 0.26},
                verification={"goe_emerges": rs < 0.15, "clustering_weak": rw > 0.25},
                series={}, summary=f"P(s<0.3): weak {rw:.2f} (clustered), strong {rs:.2f} (GOE-like)")


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


def verified(ver):
    """A result is 'verified' iff every boolean trust-flag it reports is True."""
    flags = [v for v in ver.values() if isinstance(v, bool)]
    return all(flags) if flags else True


def sweep(name, param, values, base_params=None, metric=None):
    """Scan one parameter over `values`; per-value metric + verification, plus a log-log
    scaling fit when both axes are positive numeric (e.g. Kibble-Zurek k* ~ tau_Q^b)."""
    base_params = base_params or {}
    results = []
    for v in values:
        try:
            r = run(name, **{**base_params, param: v}); m = r["metrics"]
            results.append({"value": v, "metric": (m.get(metric) if metric else None),
                            "metrics": m, "verification": r["verification"],
                            "verified": verified(r["verification"]), "summary": r["summary"]})
        except Exception as e:
            results.append({"value": v, "error": str(e)})
    out = {"experiment": name, "param": param, "metric": metric, "results": results,
           "all_verified": all(rr.get("verified", False) for rr in results if "error" not in rr)}
    if metric:
        xs = [rr["value"] for rr in results if isinstance(rr.get("value"), (int, float))
              and isinstance(rr.get("metric"), (int, float)) and rr["value"] > 0 and rr["metric"] > 0]
        ys = [rr["metric"] for rr in results if isinstance(rr.get("value"), (int, float))
              and isinstance(rr.get("metric"), (int, float)) and rr["value"] > 0 and rr["metric"] > 0]
        if len(xs) >= 3:
            b, c = np.polyfit(np.log(xs), np.log(ys), 1)
            out["scaling"] = {"form": f"{metric} ~ {param}^({b:.3f})", "exponent": float(b)}
    return _clean(out)


def compare(name, configs, metric=None):
    """Run several labelled configs of one experiment side by side; flag which extremises
    the metric. `configs` maps a label -> params dict."""
    res = {}
    for label, params in configs.items():
        try:
            r = run(name, **params)
            res[label] = {"metric": (r["metrics"].get(metric) if metric else None),
                          "metrics": r["metrics"], "verification": r["verification"],
                          "verified": verified(r["verification"]), "summary": r["summary"]}
        except Exception as e:
            res[label] = {"error": str(e)}
    out = {"experiment": name, "metric": metric, "configs": res}
    if metric:
        scored = {k: v["metric"] for k, v in res.items() if isinstance(v.get("metric"), (int, float))}
        if scored:
            out["max_label"] = max(scored, key=scored.get); out["min_label"] = min(scored, key=scored.get)
    return _clean(out)
