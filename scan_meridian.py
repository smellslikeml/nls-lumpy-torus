import numpy as np
from nls_lumpy_torus import build_operators, beam_along_meridian, run

grid = build_operators(Nx=80, Nth=160)
x = grid["x"]; th = grid["th"]
i_belly = int(np.argmin(np.abs(x - 0.0)))
i_neck = 0  # x=-pi/2 exact node


def wth_at(U, i0, thc=0.0):
    P = np.abs(U.reshape(grid["Nx"], grid["Nth"])[i0, :]) ** 2
    d = np.angle(np.exp(1j * (th - thc)))
    return np.sqrt(np.sum(P * d ** 2) / (P.sum() + 1e-300))


cases = [
    dict(wth=1.0, amp=1.0, sigma=0.0),    # wide, linear (pure geometry)
    dict(wth=0.8, amp=1.2, sigma=-1.0),   # wide, mild focusing
    dict(wth=0.6, amp=1.8, sigma=-1.0),   # moderate, stronger focusing
    dict(wth=0.5, amp=2.2, sigma=-1.0),   # narrow, soliton-strength
]
for c in cases:
    U0 = beam_along_meridian(grid, thc=0.0, amp=c["amp"], wth=c["wth"], q=4)
    U = U0.copy()
    from nls_lumpy_torus import make_stepper
    step, stats = make_stepper(grid, dt=2e-3, sigma=c["sigma"], p=2)
    print(f"\nwth={c['wth']} amp={c['amp']} sigma={c['sigma']:+.0f}")
    print(f"  {'t':>5} {'w_belly':>8} {'w_neck':>8} {'ratio':>6} {'peak':>7}")
    nsteps = int(round(1.5 / 2e-3))
    for n in range(nsteps + 1):
        if n % 250 == 0:
            wb = wth_at(U, i_belly); wn = wth_at(U, i_neck)
            pk = float(np.max(np.abs(U) ** 2))
            print(f"  {n*2e-3:5.2f} {wb:8.3f} {wn:8.3f} {wn/wb:6.2f} {pk:7.3f}")
        if n < nsteps:
            U = step(U)
    print(f"  max picard = {max(stats['picard_iters'])}")
