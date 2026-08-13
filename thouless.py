"""
Frontier prototype 9: a Thouless (adiabatic) pump from a sliding lump lattice.

Slide + breathe the lump chain through one cycle and it pumps a QUANTIZED amount of
"charge" across the strip per period -- robust, geometry-driven transport that does
not depend on the details of the drive, only on the topology of the loop it traces.

Model: the whispering-gallery lump chain is a Rice-Mele dimer chain; sliding the
lumps modulates the two bond strengths out of phase (dimerization u(s)) while
breathing them modulates the on-site energy (stagger v(s)). Cycling (u, v) around a
loop that ENCLOSES the gap-closing point pumps exactly one unit cell per cycle; a
loop that does not enclose it pumps zero. We show both, two ways:

  (1) the filled lower band's Wannier centre x-bar(s) winds by the Chern number
      (Berry-phase / Wilson-loop over the Brillouin zone), quantized to +1 vs 0;
  (2) on an open strip, a single edge state is dragged across the gap from one edge
      to the other -- the real-space face of that quantized transport.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/thorax/nls_lumpy_torus"
PINK, BLUE, GOLD = "#b83280", "#2b6cb0", "#d69e2e"
J = 1.0                      # mean hopping
DELTA = 0.7                  # dimerization amplitude (sliding the lumps)
MASS = 0.9                   # on-site stagger amplitude (breathing the lumps)


def h_bloch(k, s, enclosing=True):
    """Rice-Mele Bloch Hamiltonian. s in [0,2pi) is the pump phase."""
    u = DELTA * np.cos(s)                      # bond dimerization
    v = MASS * np.sin(s) if enclosing else MASS * (0.5 + 0.5 * np.cos(s))  # on-site stagger
    J1, J2 = J + u, J - u
    hx = J1 + J2 * np.cos(k)
    hy = J2 * np.sin(k)
    hz = v
    return np.array([[hz, hx - 1j * hy], [hx + 1j * hy, -hz]])


def wannier_center(s, enclosing=True, Nk=400):
    """Lower-band Wannier centre via the Wilson loop over the BZ (in [0,1))."""
    ks = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    lower = []
    for k in ks:
        w, V = np.linalg.eigh(h_bloch(k, s, enclosing))
        lower.append(V[:, 0])
    prod = 1.0 + 0j
    for j in range(Nk):
        prod *= np.vdot(lower[j], lower[(j + 1) % Nk])
    return (-np.angle(prod) / (2 * np.pi)) % 1.0


def open_chain_spectrum(s, Ncell=25):
    """Real-space Rice-Mele chain (open) -> energies + edge localization."""
    u, v = DELTA * np.cos(s), MASS * np.sin(s)
    J1, J2 = J + u, J - u
    N = 2 * Ncell
    H = np.zeros((N, N))
    for n in range(Ncell):
        H[2 * n, 2 * n] = v
        H[2 * n + 1, 2 * n + 1] = -v
        H[2 * n, 2 * n + 1] = H[2 * n + 1, 2 * n] = J1          # intracell
        if n < Ncell - 1:
            H[2 * n + 1, 2 * n + 2] = H[2 * n + 2, 2 * n + 1] = J2   # intercell
    w, V = np.linalg.eigh(H)
    site = np.arange(N); center = (N - 1) / 2
    edge = np.sum(V ** 2 * ((site[:, None] - center) / center), axis=0)
    return w, edge


ss = np.linspace(0, 2 * np.pi, 240)

# (1) Wannier-centre winding, topological vs trivial loop (unwrap to see net pump)
xb_topo = np.unwrap([2 * np.pi * wannier_center(s, True) for s in ss]) / (2 * np.pi)
xb_triv = np.unwrap([2 * np.pi * wannier_center(s, False) for s in ss]) / (2 * np.pi)
pump_topo = xb_topo[-1] - xb_topo[0]
pump_triv = xb_triv[-1] - xb_triv[0]
print(f"pumped charge: topological loop = {pump_topo:+.3f}  trivial loop = {pump_triv:+.3f}", flush=True)

# (2) open-chain spectrum vs s
E = np.array([open_chain_spectrum(s)[0] for s in ss])
EDG = np.array([open_chain_spectrum(s)[1] for s in ss])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.9))

axA = ax[0]
axA.plot(ss / np.pi, xb_topo - xb_topo[0], color=PINK, lw=2.2,
         label=f"sliding + breathing loop  →  {pump_topo:+.2f} per cycle")
axA.plot(ss / np.pi, xb_triv - xb_triv[0], color=BLUE, lw=2.0, ls="--",
         label=f"trivial loop  →  {pump_triv:+.2f}")
axA.axhline(1, color="0.6", lw=0.8, ls=":")
axA.set_xlabel(r"pump phase  $s/\pi$"); axA.set_ylabel("Wannier centre  (unit cells)")
axA.set_title("filled band pumps a QUANTIZED charge per cycle")
axA.legend(loc="upper left", fontsize=9, framealpha=.92)

axB = ax[1]
sc = axB.scatter(np.repeat(ss / np.pi, E.shape[1]), E.ravel(), c=EDG.ravel(),
                 cmap="coolwarm", s=4, vmin=-1, vmax=1)
axB.set_xlabel(r"pump phase  $s/\pi$"); axB.set_ylabel("energy")
axB.set_title("open strip: an edge state is dragged across the gap")
cb = fig.colorbar(sc, ax=axB); cb.set_label("edge localization  (−1 … +1)")

fig.suptitle("Thouless pump: sliding the lump lattice moves a quantized charge across the strip per cycle",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/thouless.png", dpi=120)
print("wrote thouless.png", flush=True)
