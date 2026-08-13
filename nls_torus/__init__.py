"""nls_torus — a modular solver + experiment toolkit for the nonlinear Schrodinger /
Gross-Pitaevskii equation on a lumpy torus (geometry as a control field).

Core pieces:
  geometry   — A(x;eps), curvature
  operators  — 2-D Laplace-Beltrami (build_surface), 1-D ring (ring_grid), WG chain
  steppers   — CNStepper (2-D, pluggable cubic+quintic+time-dependent coupling),
               RingStepper (1-D split-step)
  fields     — initial conditions
  diagnostics— observables + VERIFICATION signals (conservation, Wronskian, ...)
  bogoliubov — linear particle-production mode integrator
  experiment — declarative Experiment -> Result{metrics, provenance, verification}
"""
from . import geometry, operators, steppers, fields, diagnostics, bogoliubov, experiment
from .operators import build_surface, ring_grid, wg_chain
from .steppers import CNStepper, RingStepper

__all__ = ["geometry", "operators", "steppers", "fields", "diagnostics",
           "bogoliubov", "experiment", "build_surface", "ring_grid", "wg_chain",
           "CNStepper", "RingStepper"]
