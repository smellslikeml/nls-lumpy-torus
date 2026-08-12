"""
Defocusing regime (sigma=+1): the opposite of self-trapping. A high, broad hump
under repulsive nonlinearity expands and its edges steepen into oscillatory
DISPERSIVE SHOCK WAVES -- the canonical defocusing-NLS phenomenology, in
contrast to the focusing runs that concentrate/collapse.
"""
import numpy as np
from nls_lumpy_torus import build_operators, initial_condition, run
from render import animate_chart, animate_torus

OUT = "/home/thorax/nls_lumpy_torus"
grid = build_operators(96, 192)
U0 = initial_condition(grid, amp=2.5, xc=0.0, thc=np.pi, wx=0.55, wth=0.7, k=0)
U, hist, snaps, stats = run(grid, U0, dt=1e-3, T=1.2, sigma=+1.0, p=2,
                            n_snapshots=80, verbose=True)
print(f"  mass drift {(hist['mass'][-1]-hist['mass'][0])/hist['mass'][0]:+.1e}, "
      f"max picard {max(stats['picard_iters'])}", flush=True)
animate_chart(grid, snaps, f"{OUT}/nls_defocusing_chart.gif", fps=18)
animate_torus(grid, snaps, f"{OUT}/nls_defocusing_torus.gif", fps=18)
print("ALL DONE", flush=True)
