"""Namesake hero render: a localized focusing wavepacket on the lumpy torus,
plus the conservation diagnostic and the (x,theta) chart. Rendered on the
crescent immersion (via render.torus_embedding)."""
import numpy as np
from nls_lumpy_torus import build_operators, initial_condition, run
from render import plot_conservation, animate_chart, animate_torus

OUT = "/home/thorax/nls_lumpy_torus"
grid = build_operators(80, 160)
U0 = initial_condition(grid, amp=1.2, xc=0.0, thc=np.pi, wx=0.35, wth=0.5, k=6)
U, hist, snaps, stats = run(grid, U0, dt=2e-3, T=2.0, sigma=-1.0, p=2,
                            n_snapshots=80, verbose=True)
plot_conservation(hist, f"{OUT}/conservation.png")
animate_chart(grid, snaps, f"{OUT}/nls_chart.gif")
animate_torus(grid, snaps, f"{OUT}/nls_lumpy_torus.gif")
print("ALL DONE", flush=True)
