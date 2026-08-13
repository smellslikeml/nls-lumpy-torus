"""Initial conditions on the surface and the ring."""
import numpy as np


def localized_hump(surface, amp=1.0, xc=0.0, thc=np.pi, wx=0.35, wth=0.35, k=0):
    """A 2-D Gaussian blob (Townes-type collapse / single-hump experiments)."""
    x, th = surface["x"], surface["th"]
    X, TH = np.meshgrid(x, th, indexing="ij")
    dth = np.angle(np.exp(1j * (TH - thc)))
    env = np.exp(-((X - xc) ** 2) / (2 * wx ** 2) - dth ** 2 / (2 * wth ** 2))
    return (amp * env * np.exp(1j * k * TH)).ravel()


def geodesic_ring(surface, xc=0.0, amp=1.0, wx=0.30, k=6, Lx=np.pi):
    """A ring concentrated transverse to the parallel geodesic x=xc (self-trapping,
    necklace collapse)."""
    x, th = surface["x"], surface["th"]
    X, TH = np.meshgrid(x, th, indexing="ij")
    dxc = (X - xc + Lx / 2.0) % Lx - Lx / 2.0
    return (amp * np.exp(-dxc ** 2 / (2 * wx ** 2)) * np.exp(1j * k * TH)).ravel()


def uniform_noise(ring, a0=1.0, noise=3e-3, seed=0):
    """Uniform background + complex noise on the ring (MI / quench experiments)."""
    rng = np.random.default_rng(seed)
    n = ring["Nth"]
    return a0 + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def ring_seeded(ring, a0=1.0, q_seed=1, eps_seed=1e-3):
    """Uniform background + a single seeded mode (Akhmediev/Peregrine rogue waves)."""
    th = ring["th"]
    return a0 * (1.0 + eps_seed * np.cos(q_seed * th / ring["radius"]))
