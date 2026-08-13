"""Geometry of a surface of revolution  ds^2 = dx^2 + A(x)^2 dtheta^2.

A(x) is the single control field. This module keeps it GENERAL: any positive periodic
profile A(x) is a valid (torus-topology) manifold, so new geometries are one
`@register_geometry` away — the operators and the agent only need A(x), the x-domain,
and a topology label. The lumpy torus is just the default instance.

    sqrt|g| = A ,   K = -A''/A  (Gaussian curvature).

Extending to other topologies (a sphere has A->0 at two poles; higher genus needs a
different chart) means registering a Geometry with the right `topology` and teaching
`operators.build_surface` how to assemble that case — the abstraction leaves room for
it without pretending to support it yet.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


# ------------------------------------------------------------------ profiles
def profile_A(x, eps=1.0):
    """Tunable lump family: A(belly)=1, A(neck)=1/sqrt(1+eps); eps=1 is the study torus,
    eps=0 the flat cylinder (the belly<->neck bifurcation point)."""
    return np.sqrt((1.0 + eps * np.cos(x) ** 2) / (1.0 + eps))


def curvature_K(x, eps=1.0):
    """Gaussian curvature K=-A''/A for the tunable family (analytic)."""
    A = profile_A(x, eps)
    A2p = -eps * np.sin(2 * x) / (1.0 + eps)
    A2pp = -2 * eps * np.cos(2 * x) / (1.0 + eps)
    Ap = A2p / (2 * A)
    App = (A2pp - 2 * Ap ** 2) / (2 * A)
    return -App / A


# ------------------------------------------------------------------ Geometry
@dataclass
class Geometry:
    """A surface of revolution, described by everything the solver/agent need."""
    name: str
    A: Callable                       # A(x) -> ndarray, must be > 0 and Lx-periodic
    x0: float = -np.pi / 2.0
    Lx: float = np.pi
    topology: str = "torus"
    note: str = ""

    def A_of(self, x):
        return self.A(x)

    def curvature(self, x, h=1e-4):
        A = np.asarray(self.A(x), float)
        return -(self.A(x + h) - 2 * A + self.A(x - h)) / h ** 2 / A


GEOMETRIES = {}


def register_geometry(name):
    def deco(fn):
        GEOMETRIES[name] = fn
        return fn
    return deco


@register_geometry("lumpy_torus")
def _lumpy_torus(eps=1.0, **_):
    return Geometry("lumpy_torus", lambda x: profile_A(x, eps),
                    note=f"tunable lump, eps={eps}; K_belly={eps/(1+eps):.3f}, K_neck={-eps:.3f}")


@register_geometry("flat_cylinder")
def _flat_cylinder(**_):
    return Geometry("flat_cylinder", lambda x: np.ones_like(np.asarray(x, float)),
                    note="A=1 everywhere (eps=0): flat, zero curvature")


@register_geometry("double_lump")
def _double_lump(eps=1.0, **_):
    # two belly/neck pairs per period -> a different lump structure (peanut)
    return Geometry("double_lump", lambda x: np.sqrt((1.0 + eps * np.cos(2 * x) ** 2) / (1.0 + eps)),
                    note=f"two lumps per period, eps={eps} (four critical parallels)")


@register_geometry("gaussian_bump")
def _gaussian_bump(height=0.6, width=0.5, **_):
    # a single smooth (periodic) bump on an otherwise round tube
    def A(x):
        x = np.asarray(x, float)
        d = (x + np.pi / 2) % np.pi - np.pi / 2
        return 1.0 + height * np.exp(-d ** 2 / (2 * width ** 2)) - height * np.exp(-(np.pi / 2) ** 2 / (2 * width ** 2))
    return Geometry("gaussian_bump", A, note=f"single Gaussian lump h={height}, w={width}")


def make_geometry(spec="lumpy_torus", **kw):
    """Resolve a geometry: a name (see GEOMETRIES) with optional params, or a Geometry."""
    if isinstance(spec, Geometry):
        return spec
    if spec not in GEOMETRIES:
        raise KeyError(f"unknown geometry '{spec}'. Available: {sorted(GEOMETRIES)}")
    return GEOMETRIES[spec](**kw)
