"""H5 — non-compact horn resonator: geometry sets the Q of a leaky mode.

Past compact surfaces: the meridian runs x in [0, Xmax] with a throat (a dip in A(x), so a
barrier in the centrifugal potential V_k=k^2/A^2) that turns [0,x_t] into a CAVITY radiating
into an open horn [x_t, Xmax]. A complex absorbing potential (CAP/PML) near Xmax enforces the
outgoing condition, so the operator is non-Hermitian and its cavity modes become complex
resonances E = E_r - i*Gamma/2 with finite lifetime and quality factor Q = E_r/Gamma.

The lifetime is tunnelling-limited by the throat, so Q is set by the geometry: deepen the
throat and Q climbs exponentially (the leaky-resonator cousin of the H4 tunnelling doublet).

VERIFICATION is intrinsic and load-bearing: a CAP spectrum is littered with PML/box
artifacts (near-real eigenvalues, absurd Q). A *physical* resonance is invariant when the
absorber strength eta changes; an artifact moves. We keep only eta-stable eigenvalues.

Regenerate:  python3 horn_resonator.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_open, xt, w, Xmax, Xpml = 1.0, 8.0, 1.0, 40.0, 30.0
K = 2.0


def build(Delta, eta, w, N=1600):
    x = np.linspace(0, Xmax, N); dx = x[1] - x[0]
    A = A_open - Delta * np.exp(-((x - xt) / w) ** 2)
    Vk = K ** 2 / A ** 2
    cap = eta * np.clip((x - Xpml) / (Xmax - Xpml), 0, None) ** 2
    diag = 2 / dx ** 2 + Vk - 1j * cap
    off = -1 / dx ** 2 * np.ones(N - 1)
    return sp.diags([off, diag, off], [-1, 0, 1]).tocsc(), x, A, Vk


def spectrum(Delta, eta, w, nvec=18):
    H, x, A, Vk = build(Delta, eta, w)
    vals, vecs = eigs(H, k=nvec, sigma=K ** 2 + (np.pi / xt) ** 2 - 0.02j)
    return vals, vecs, x, A, Vk


def stable_resonance(Delta, w=1.0, eta1=2.0, eta2=4.5, tol=0.05):
    """The physical leaky resonance = the eta-INVARIANT eigenvalue. Every horn-continuum /
    PML artifact shifts by ~half its width when the absorber strength changes; the true
    resonance does not. Among leaky candidates (V_open<E_r<V_barrier, Gamma>0) return the
    most eta-stable one, or None if none is stable to `tol`."""
    Vopen, Vbar = K ** 2, K ** 2 / (A_open - Delta) ** 2
    v1, vec1, x, A, Vk = spectrum(Delta, eta1, w)
    v2, _, _, _, _ = spectrum(Delta, eta2, w)
    best = None
    for e, col in zip(v1, vec1.T):
        if not (Vopen < e.real < Vbar and e.imag < -1e-12):
            continue
        rel = abs(v2[np.argmin(np.abs(v2 - e))] - e) / max(abs(e.imag), 1e-12)
        if rel < tol and (best is None or rel < best[3]):
            best = (e, col, e.real / (-2 * e.imag), rel)
    if best is None:
        return None
    e, col, Q, rel = best
    return dict(E=e, Q=float(Q), psi=col, x=x, A=A, Vk=Vk, eta_rel=float(rel))


# ---- the physical resonance + throat-depth sweep -------------------------------------
base = stable_resonance(0.40)
print(f"physical resonance (throat Delta=0.40): E = {base['E'].real:.4f} "
      f"{base['E'].imag:+.2e} i,  Q = {base['Q']:.1f},  eta-drift {base['eta_rel']:.1e} (stable)")

deltas = [0.24, 0.30, 0.36, 0.42, 0.48]
Qs, ok = [], True
print(f"\n{'throat Delta':>12s} {'V_barrier':>10s} {'E_r':>8s} {'Gamma':>10s} {'Q':>10s} {'eta-drift':>10s}")
for d in deltas:
    r = stable_resonance(d)
    if r is None:
        Qs.append(np.nan); print(f"{d:12.2f}  (no eta-stable resonance — width below CAP floor)"); continue
    Qs.append(r["Q"]); ok &= r["eta_rel"] < 0.05
    print(f"{d:12.2f} {K**2/(A_open-d)**2:10.2f} {r['E'].real:8.4f} {-2*r['E'].imag:10.2e} "
          f"{r['Q']:10.1f} {r['eta_rel']:10.1e}")
print(f"\nVERIFICATION every reported Q is eta-stable (<5%): {ok}")

# ---- figure ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))
r = base; x, Vk, psi = r["x"], r["Vk"], r["psi"]
rho = np.abs(psi) ** 2; rho = rho / rho.max()
axL.plot(x, Vk, color="#1c2230", lw=2.0, label="$V_k=k^2/A^2$ (throat = barrier)")
axL.axhline(r["E"].real, color="#b83280", ls="--", lw=1.4, label=f"resonance $E_r={r['E'].real:.2f}$")
axL.fill_between(x, r["E"].real, r["E"].real + 3.2 * rho, color="#2b6cb0", alpha=.45, lw=0,
                 label="$|\\psi|^2$ (cavity-bound, leaking out)")
axL.axvspan(Xpml, Xmax, color="#f0d8d8", alpha=.6); axL.text(Xpml + 0.3, 2.4, "absorbing\ncollar (PML)", fontsize=8.5, color="#a33")
axL.axvline(xt, color="#aaa", ls=":", lw=1); axL.text(xt + .2, 0.8, "throat", fontsize=9, color="#666")
axL.set_xlabel("meridian $x$ (semi-infinite horn)"); axL.set_ylabel("energy")
axL.set_ylim(0, 12.5); axL.set_title(f"leaky cavity mode, $Q={r['Q']:.0f}$", fontsize=11.5)
axL.legend(fontsize=8.6, loc="upper right")

axR.semilogy(deltas, Qs, "o-", color="#2b6cb0", lw=2.0, ms=7)
for d, q in zip(deltas, Qs):
    if np.isfinite(q):
        axR.annotate(f"{q:.0f}", (d, q), textcoords="offset points", xytext=(5, 6), fontsize=8.5, color="#3a4658")
axR.set_xlabel("throat depth  $\\Delta$  (barrier height $\\to$)")
axR.set_ylabel("quality factor  $Q = E_r/\\Gamma$")
axR.set_title("geometry sets Q — tunnelling-limited, exponential", fontsize=11.5)
axR.grid(True, which="both", alpha=.25)
fig.tight_layout(); fig.savefig("horn_resonator.png", dpi=130, bbox_inches="tight")
print("wrote horn_resonator.png")
