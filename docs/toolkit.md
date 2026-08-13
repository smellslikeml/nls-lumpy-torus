---
type: Toolkit
title: nls_torus — modular solver + agent tool
description: The scripts behind the gallery, refactored into a reusable library and a self-describing tool for numerically-grounded inference.
resource: https://github.com/smellslikeml/nls-lumpy-torus/blob/main/docs/toolkit.md
tags: [nls, solver, library, agent, numerically-grounded]
---

# nls_torus — a modular solver + an agent tool

The gallery grew as a pile of scripts that each re-derived the same solver (and each
grew its own copy-paste bugs — a Picard guard that mislabels stiff-but-stable steps as
collapse, a surface-gravity gradient corrupted by a sonic-point kink, a defect counter
fooled by broadband noise). `nls_torus/` factors that into one tested engine, and
`experiment_tool.py` exposes it as a typed tool an agent can drive.

## Library

```
nls_torus/
  geometry.py   A(x;eps), curvature K = -A''/A   (eps=0 flat cylinder … eps=1 the torus)
  operators.py  build_surface (2-D Laplace–Beltrami K, M), ring_grid (1-D), wg_chain (tight-binding)
  steppers.py   CNStepper  — 2-D Crank–Nicolson, ONE pluggable nonlinearity
                             N(u,t) = sigma3(t)|u|^2 u + sigma5|u|^4 u  (cubic, quintic, time-dependent),
                             solved by Picard OR Newton (real-split Jacobian; ~2.5x fewer
                             iterations and robust where Picard stalls near concentration)
                RingStepper — 1-D split-step Fourier
  fields.py     localized_hump, geodesic_ring, uniform_noise, ring_seeded
  diagnostics.py mass, energy, peak, spectrum, level_spacings + VERIFICATION
                 (conservation_drift, wronskian_residual)
  bogoliubov.py linear particle-production mode integrator (Wronskian-checked)
  mesh.py       cotangent (FEM) Laplace-Beltrami on ARBITRARY triangle meshes — past
                surfaces of revolution. Validated on the sphere (eig = l(l+1)); genus-2
                generator (marching cubes, needs scikit-image) + non-compact horns
  experiment.py declarative experiments -> Result{metrics, verification, provenance, series, summary}
```

Beyond the genus-1 metric `ds^2=dx^2+A(x)^2 dtheta^2`, three registered experiments run on
`mesh.py`/CAP: `sta_transport` (shortcut-to-adiabaticity soliton transport), `genus2_tunneling`
(which-handle doublet splitting on a true genus-2 surface — H4 on higher genus), and
`horn_resonator` (geometry-set Q of a leaky mode in an open horn, gated by absorber-independence).
And `microdisk_Q` takes it to a real device: the radiation-limited Q of a silica-microdisk
whispering-gallery mode (as radial tunnelling) + the Kerr-comb threshold `P_th ~ 1/Q^2`, showing
geometry is an exponential lever on Q but a null on the Townes-universal nonlinear threshold.

```python
import nls_torus as nt
from nls_torus import fields, diagnostics as d
surf = nt.build_surface(96, 96, eps=1.0)
step = nt.CNStepper(surf, dt=1e-3, sigma5=0.4)          # focusing cubic + defocusing quintic
U = fields.localized_hump(surf, amp=6.0)
for _ in range(1500): U = step.step(U, sigma3=-1.0)
print(d.peak(U))                                        # bounded -> arrested
```

## Agent tool — `run_experiment`

Self-describing, deterministic, JSON in/out. The point is the **verification block**:
it lets the caller separate a trustworthy result from a plausible-looking artifact.

```
python3 experiment_tool.py --list                              # capabilities (JSON)
python3 experiment_tool.py run collapse '{"amp":6,"sigma5":0.4}'
python3 experiment_tool.py run expanding_cosmos --figure out.png
```

A result:

```json
{ "experiment": "expanding_cosmos",
  "metrics":      {"beta2_k1": 0.329, "plateau_analytic": 0.333, "sakharov_contrast": 2.34},
  "verification": {"wronskian_max_dev": 2e-13, "wronskian_ok": true, "plateau_match": true},
  "provenance":   {"solver": "nls_torus", "git_commit": "…",
                   "note": "reduced mean-field model; results need validation vs full physics"} }
```

## MCP server — native agent tools (`mcp_server.py`)

The same toolkit is exposed over the Model Context Protocol, so Claude Code calls it as
**native tools** (no shelling out). Register once:

```
claude mcp add nls-torus -- python3 /path/to/mcp_server.py
```

Verbs:

| tool | what it does |
|---|---|
| `list_experiments()` | capabilities — names, params, defaults |
| `run_experiment(name, params)` | one run → `{metrics, verification, provenance, summary}` |
| `sweep(name, param, values, base_params, metric)` | scan one parameter; returns per-value metric + verification **and a log-log scaling fit** |
| `compare(name, configs, metric)` | run labelled configs side by side; reports which extremises the metric |
| `compose(code, resolutions)` | run a NEW experiment (`run(Nx)` you write) with **harness-owned** verification — the server computes conservation + grid-convergence itself; you can't self-certify |
| `search_arxiv(query)` | cross-check a finding against the literature (no key) |
| `probe(question, geometry)` | hand the whole design→run→verify loop to the generative `ManifoldExperimenter` (NOOA/Gemini sub-agent, py3.12 subprocess) |

The last three expose the `ManifoldExperimenter` itself: `compose` lets the calling agent
write the experiment and have the harness verify it; `probe` delegates the full loop to
the autonomous sub-agent.

`sweep` turns a scaling question into one call. Example — Kibble–Zurek:

```
sweep("kz_freeze", "tau_Q", [0.1, 0.4, 1.6, 6.4], metric="kstar_mean")
  → k* = 5.65, 4.28, 3.02, 2.30 ;  scaling: kstar_mean ~ tau_Q^(-0.219) ;  all_verified: true
```

— the tool recovers the KZ exponent and confirms every point passed its trust checks.
`compare` does the three collapse regimes at once (bare collapses, cubic-quintic
arrests, managed hastens), each tagged with its verification flag.

## Generative agent — `ManifoldExperimenter` (NOOA)

Where the MCP verbs *run the experiments we designed*, the agent *composes new ones on
demand* for questions no registered experiment covers. It's a NOOA (`nooa.Agent`) built
like VQASynth's `SpatialAnnotator`: the toolkit functions are the tool surface (NOOA
derives schemas from signatures + docstrings), and `probe(question)` is a CodeAct
generation method — the LLM writes Python composing the tools in a sandboxed REPL
(AST validation, deny-lists, iteration budget, tracing).

- **Geometry-general.** Parameterized by any registered manifold — `lumpy_torus`,
  `flat_cylinder`, `double_lump`, `gaussian_bump` — and new ones are one
  `@register_geometry` away (the solver only needs `A(x)`). Experiments thread a
  `geometry=` param straight through `build_surface`.
- **Verify-gated.** `probe()` returns a typed `ExperimentResult{metrics, verification,
  provenance, trustworthy}`; the agent is instructed to call `is_trustworthy(...)` and
  refine rather than trust a result whose flags fail (the metadata_resolver discipline).

```python
from nls_torus.agent import ManifoldExperimenter          # needs py3.12 + nooa + GOOGLE_API_KEY
agent = ManifoldExperimenter(geometry="lumpy_torus", llm="gemini/gemini-2.5-pro")
res = await agent.probe("Does a defocusing quintic prevent the collapse pure cubic suffers?")
# -> "Yes … cubic collapsed at t=0.073; cubic+quintic stable (peak→6.2). trustworthy=True"
#    with per-run metrics + verification, computed live — not asserted from priors.
```

The deterministic tool functions (`tool_run`, `tool_sweep`, `tool_compare`) run without
nooa or an LLM, so the tool surface is unit-testable on its own.

## Numerically-grounded inference

The loop this project ran by hand — *pose a phenomenon → map to control knobs → run →
verify against a conserved quantity / analytic bound / cross-check → iterate → honestly
scope* — is an agent loop. `run_experiment` makes it mechanical: the agent reasons over
returned numbers, not its priors, and the verification signals (mass drift, Wronskian,
`kappa_dyn` vs `kappa_static`, plateau match) are what caught the real bugs here. The
pattern is domain-agnostic: **reduced-model sandbox + typed runner + verification metrics
+ an honesty rubric = an agent that does numerically-grounded science.**
