"""H1 — is the collapse threshold set by geometry? The confound-controlled answer: NO.
The mass-critical (Townes) blow-up is a self-similar concentration at a point, so it sees
only the local (~flat) metric and the critical mass is a pure number. Measured correctly
(amplitude BISECTED to the true threshold), the critical mass is the Townes value
||Q||^2 ~ 11.7 for EVERY geometry — independent of local curvature AND of global structure
(one lobe, two lobes, a localized bump). The apparent "curvature lowers M_c" trend from a
coarse scan was entirely an artifact: mass = int A|u|^2 is A-weighted, so mass read at a
fixed amplitude drifts with geometry even though the true threshold does not.

Found + retracted by the nls_torus agent-toolkit under harness-owned verification (mass
conservation to ~1e-10, grid check at the threshold). Regenerate:
    python3 curvature_universality.py     (recomputes; a few minutes of bisection)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nls_torus import build_surface, CNStepper, fields
from nls_torus import diagnostics as d
from nls_torus.geometry import make_geometry

Nx, DT, T = 72, 1e-3, 1.0


def collapses(name, kw, amp):
    surf = build_surface(Nx, Nx, geometry=make_geometry(name, **kw))
    step = CNStepper(surf, DT); U = fields.localized_hump(surf, amp=amp, wx=0.35, wth=0.35)
    M = surf["Mdiag"]; m0 = d.mass(U, M); pk = d.peak(U)
    for i in range(1, int(T / DT)):
        U = step.step(U, sigma3=-1.0); pk = max(pk, d.peak(U))
        if pk > 150:
            return True, m0
    return False, m0


def mc_bisect(name, kw):                      # true critical mass at the bisected threshold
    lo, hi = 4.0, 8.0
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        col, _ = collapses(name, kw, mid)
        (hi := mid) if col else (lo := mid)
    return collapses(name, kw, hi)[1]


def coarse_proxy(name, kw, amp=6.0):          # the confounded read: mass at a fixed amplitude
    return collapses(name, kw, amp)[1]


def K(name, **kw):
    return float(make_geometry(name, **kw).curvature(0.0))


# ---- left: matched belly-K=0.5, four different global structures -----------------------
matched = [("flat_cylinder", {}, "flat\n(K=0)"),
           ("lumpy_torus", {"eps": 1.0}, "one lobe"),
           ("double_lump", {"eps": 0.143}, "two lobes"),
           ("gaussian_bump", {"height": 0.218, "width": 0.6}, "localized\nbump")]
mc_matched = [mc_bisect(n, kw) for n, kw, _ in matched]
townes = float(np.mean(mc_matched))

# ---- right: a curvature sweep — confounded proxy vs the true threshold ------------------
sweep = [("flat_cylinder", {}), ("lumpy_torus", {"eps": 0.3}), ("lumpy_torus", {"eps": 1.0}),
         ("lumpy_torus", {"eps": 3.0}), ("double_lump", {"eps": 1.0})]
Ks = [K(n, **kw) for n, kw in sweep]
mc_true = [mc_bisect(n, kw) for n, kw in sweep]
mc_conf = [coarse_proxy(n, kw) for n, kw in sweep]
order = np.argsort(Ks)
Ks = np.array(Ks)[order]; mc_true = np.array(mc_true)[order]; mc_conf = np.array(mc_conf)[order]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.5))

xpos = np.arange(len(matched))
axL.axhline(townes, color="#b83280", ls="--", lw=1.6, zorder=1,
            label=f"Townes $\\|Q\\|^2 \\approx {townes:.1f}$")
axL.scatter(xpos, mc_matched, s=120, color="#2b6cb0", zorder=3, edgecolor="white", lw=1.2)
for xp, m in zip(xpos, mc_matched):
    axL.annotate(f"{m:.2f}", (xp, m), textcoords="offset points", xytext=(0, 11),
                 ha="center", fontsize=9.5, color="#1c2230")
axL.set_xticks(xpos); axL.set_xticklabels([lab for *_, lab in matched], fontsize=9.5)
axL.set_ylim(townes - 1.2, townes + 1.2)
axL.set_ylabel("critical mass  $M_c$"); axL.legend(fontsize=9.5, loc="upper right")
axL.set_title("matched belly-K = 0.5 : $M_c$ is geometry-blind", fontsize=11.5)

axR.plot(Ks, mc_conf, "s--", color="#c0803a", lw=1.8, ms=7, label="confounded read\n(mass at fixed amplitude)")
axR.plot(Ks, mc_true, "o-", color="#2b6cb0", lw=2.0, ms=7, label="true threshold\n(amplitude bisected)")
axR.axhline(townes, color="#b83280", ls="--", lw=1.2, alpha=.8)
axR.text(0.02, townes + 0.15, "Townes", transform=axR.get_yaxis_transform(),
         color="#b83280", fontsize=9)
axR.set_xlabel("belly curvature  $K$"); axR.set_ylabel("critical mass  $M_c$")
axR.set_title("the 'curvature trend' was the A-weighting confound", fontsize=11.5)
axR.legend(fontsize=9, loc="center right")

fig.tight_layout()
fig.savefig("curvature_universality.png", dpi=130, bbox_inches="tight")
print("matched-K M_c:", [f"{m:.2f}" for m in mc_matched], "-> Townes ~", f"{townes:.2f}")
print("sweep K:", [f"{k:.2f}" for k in Ks])
print("  true M_c :", [f"{m:.2f}" for m in mc_true])
print("  confound :", [f"{m:.2f}" for m in mc_conf])
