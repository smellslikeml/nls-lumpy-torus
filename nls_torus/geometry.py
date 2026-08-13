"""Geometry of the surface of revolution: the generating radius A(x) and its
curvature. A(x) is the single control field — static, tunable (eps), or (for the
dynamical experiments) modulated in time by a separate protocol.

    ds^2 = dx^2 + A(x)^2 dtheta^2 ,   sqrt|g| = A ,   K = -A''/A  (Gaussian curvature).

Tunable family  A(x;eps) = sqrt((1 + eps cos^2 x)/(1 + eps))  has, exactly,
    K_belly = eps/(1+eps)  (x=0, elliptic for eps>0),
    K_neck  = -eps         (x=+-pi/2, hyperbolic for eps>0),
so eps=0 is the flat cylinder (the belly<->neck stability bifurcation) and eps=1 is
the study's torus.
"""
import numpy as np


def profile_A(x, eps=1.0):
    """Generating radius A(x;eps): A(belly)=1, A(neck)=1/sqrt(1+eps)."""
    return np.sqrt((1.0 + eps * np.cos(x) ** 2) / (1.0 + eps))


def curvature_K(x, eps=1.0):
    """Gaussian curvature K = -A''/A for the tunable family (analytic)."""
    A = profile_A(x, eps)
    # A^2 = (1 + eps cos^2 x)/(1+eps);  (A^2)'' gives A'' via A'' = ((A^2)'' - 2 A'^2)/(2A)
    s2 = np.sin(2 * x)
    A2p = -eps * s2 / (1.0 + eps)                 # (A^2)'
    A2pp = -2 * eps * np.cos(2 * x) / (1.0 + eps)  # (A^2)''
    Ap = A2p / (2 * A)
    App = (A2pp - 2 * Ap ** 2) / (2 * A)
    return -App / A


def geometry(kind="lumpy_torus", eps=1.0):
    """Return a small spec dict describing the geometry (for provenance)."""
    return {"kind": kind, "eps": float(eps),
            "K_belly": float(eps / (1 + eps)), "K_neck": float(-eps)}
