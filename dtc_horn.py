"""Disorder-stabilized discrete time crystal — resurrecting the response H2 found dead.

Section 11 asked whether the clean driven ring is a discrete time crystal (DTC). It is not:
a sharp parametric resonance, frequency-pulled, heating with no rigidity. The diagnosis was
that a clean mean-field system has no localization mechanism to rigidify the subharmonic.
This test supplies exactly that missing ingredient — GEOMETRIC DISORDER (random on-site
energies, the Anderson-localizing cousin of MBL) — and re-runs the question.

Protocol (a driven nonlinear lattice): each Floquet period = an imperfect dimerized pi-swap
that flips a charge-density wave, then free evolution under disorder + hopping + interaction.
Order parameter O = amplitude of the period-2 (subharmonic) response. A DTC shows O large and
RIGID against the drive imperfection e; a non-DTC collapses with any e. The clean lattice
(W=0) is the built-in negative control — the H2 case.

Result: disorder resurrects a rigid period-2 response (O ~ 0.7, plateau to e ~ 0.1) that the
clean lattice cannot sustain (O ~ 0.02). The open-horn leakage is ~neutral: disorder, not
openness, is what the H2 gap required.

Regenerate:  python3 dtc_horn.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, JF, U, TF, DT, NPER = 64, 0.4, 1.0, 1.0, 0.02, 80


def deriv(p, eps, gmask):
    return -1j * (eps * p - JF * (np.roll(p, 1) + np.roll(p, -1)) + U * np.abs(p) ** 2 * p) - gmask * p


def free(psi, eps, gmask):
    for _ in range(int(round(TF / DT))):
        k1 = deriv(psi, eps, gmask); k2 = deriv(psi + 0.5*DT*k1, eps, gmask)
        k3 = deriv(psi + 0.5*DT*k2, eps, gmask); k4 = deriv(psi + DT*k3, eps, gmask)
        psi = psi + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    return psi


def trajectory(W, e, seed, leaky=False):
    rng = np.random.default_rng(seed); eps = rng.uniform(-W, W, L)
    psi = np.zeros(L, complex); psi[0::2] = 1.0; th = (np.pi/2) * (1 - e)
    gmask = np.zeros(L)
    if leaky:
        gmask[L//2-2:L//2+2] = 0.5
    Is = []
    for _ in range(NPER):
        a = psi[0::2].copy(); b = psi[1::2].copy()                 # imperfect dimerized pi-swap
        psi[0::2] = np.cos(th)*a - 1j*np.sin(th)*b
        psi[1::2] = -1j*np.sin(th)*a + np.cos(th)*b
        psi = free(psi, eps, gmask)
        tot = np.sum(np.abs(psi)**2) + 1e-30
        Is.append((np.sum(np.abs(psi[0::2])**2) - np.sum(np.abs(psi[1::2])**2)) / tot)
    return np.array(Is)


def order(W, e, leaky=False, nseed=8):
    Os = []
    for s in range(nseed):
        Is = trajectory(W, e, s, leaky); h = NPER//2
        Os.append(abs(np.mean(Is[h:] * (-1.0)**np.arange(h, NPER))))
    return float(np.mean(Os)), float(np.std(Os))


# ---- subharmonic-frequency verification ----------------------------------------------
Is_d = trajectory(20.0, 0.1, 0); Is_c = trajectory(0.0, 0.1, 0)
seg = Is_d[NPER//4:] - Is_d[NPER//4:].mean()
f = np.fft.rfftfreq(len(seg), d=1.0); P = np.abs(np.fft.rfft(seg))
f_peak = f[1 + np.argmax(P[1:])]
print(f"disordered response peak frequency = {f_peak:.3f} cycles/period (period-2 => 0.5)")
print(f"VERIFICATION subharmonic locked to Omega/2: {abs(f_peak-0.5) < 0.02}")

# ---- rigidity sweep ------------------------------------------------------------------
es = np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.26, 0.32])
curves = {"clean (W=0) — the H2 case": (0.0, False, "#c0803a"),
          "disordered (W=20)": (20.0, False, "#2b6cb0"),
          "disordered + leaky horn": (20.0, True, "#3aa76d")}
print(f"\n{'imperfection e':>14s} " + " ".join(f"{k.split('(')[0].strip()[:10]:>11s}" for k in curves))
data = {}
for lbl, (W, lk, _) in curves.items():
    data[lbl] = np.array([order(W, e, lk) for e in es])
for i, e in enumerate(es):
    print(f"{e:14.2f} " + " ".join(f"{data[k][i,0]:11.3f}" for k in curves))

# ---- figure ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.5))
p = np.arange(NPER)
sd = Is_d * (-1.0) ** p; sc = Is_c * (-1.0) ** p            # demodulate the period-2 response
axA.plot(p, sd, color="#2b6cb0", lw=1.7, label=f"disordered: rigid ($O={sd[NPER//2:].mean():.2f}$)")
axA.plot(p, sc, color="#c0803a", lw=1.4, alpha=.9, label="clean: incoherent ($O\\approx0$)")
axA.axhline(sd[NPER//2:].mean(), color="#2b6cb0", ls=":", lw=1.0)
axA.axhline(0, color="#999", lw=.8)
axA.set_xlabel("Floquet period"); axA.set_ylabel("subharmonic response  $I_p\\,(-1)^p$")
axA.set_title("the period-2 response: locked vs incoherent", fontsize=11.5)
axA.legend(fontsize=9, loc="upper right"); axA.set_ylim(-1.05, 1.05)

for lbl, (W, lk, c) in curves.items():
    m, sd = data[lbl][:, 0], data[lbl][:, 1]
    axB.plot(es, m, "o-", color=c, lw=2.0, ms=6, label=lbl)
    axB.fill_between(es, m - sd, m + sd, color=c, alpha=.15)
axB.set_xlabel("drive imperfection  $e$  (swap angle $= \\frac{\\pi}{2}(1-e)$)")
axB.set_ylabel("period-2 order parameter  $O$")
axB.set_title("rigidity: disorder resurrects the time crystal", fontsize=11.5)
axB.legend(fontsize=8.6, loc="upper right"); axB.grid(alpha=.25)
fig.tight_layout(); fig.savefig("dtc_horn.png", dpi=130, bbox_inches="tight")
print("wrote dtc_horn.png")
