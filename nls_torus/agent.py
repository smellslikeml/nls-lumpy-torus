"""ManifoldExperimenter — a NOOA agent that composes nls_torus primitives into NEW
numerical experiments on demand, to answer questions no registered experiment covers.

Mirrors VQASynth's SpatialAnnotator: the toolkit functions are the tool surface (NOOA
derives their schemas from signatures + docstrings), and the `probe(question)`
generation method has the LLM write Python that composes them in a sandboxed CodeAct
REPL (AST validation, deny-lists, iteration budget, tracing — all from NOOA). The agent
is geometry-parameterized (any registered manifold) and VERIFY-GATED: every answer must
carry a verification block, and the agent is instructed to reject a result whose
trust-flags fail and refine instead (the metadata_resolver discipline).

Runs need Python 3.12 + nooa + numpy/scipy (+ an LLM key for a live probe()). The
deterministic tool functions below run anywhere — no nooa, no LLM — so the tool surface
is unit-testable on its own.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from . import experiment as X
from . import geometry as G

try:
    from nooa import Agent, strategy
    from nooa.strategies import CodeActStrategy
    from nooa.config import CodeActConfig
    _NOOA_IMPORT_ERROR = None
except ImportError as _e:                       # nooa needs Python 3.12
    Agent = object; strategy = None; CodeActStrategy = None; CodeActConfig = None
    _NOOA_IMPORT_ERROR = _e

DEFAULT_MAX_ITERATIONS = 16


@dataclass
class ExperimentResult:
    """Typed output of probe(): what was found, and whether to trust it."""
    question: str
    summary: str
    metrics: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    trustworthy: bool = False


# ------------------- deterministic tool surface (no nooa / no LLM needed) -------------------
def _accepts_geometry(name):
    return "geometry" in inspect.signature(X.REGISTRY[name]).parameters


def tool_available_experiments() -> dict:
    """The registered experiments, their one-line docs, and their parameters/defaults."""
    return {n: {"doc": (f.__doc__ or "").strip().split("\n")[0],
                "params": {k: (None if p.default is inspect._empty else p.default)
                           for k, p in inspect.signature(f).parameters.items()}}
            for n, f in sorted(X.REGISTRY.items())}


def tool_available_geometries() -> list:
    """The registered manifolds an experiment can run on."""
    return sorted(G.GEOMETRIES)


def tool_run(name, params=None, geometry="lumpy_torus") -> dict:
    """Run a registered experiment (on `geometry` if it supports one)."""
    p = dict(params or {})
    if _accepts_geometry(name):
        p.setdefault("geometry", geometry)
    return X.run(name, **p)


def tool_sweep(name, param, values, base_params=None, metric=None, geometry="lumpy_torus") -> dict:
    bp = dict(base_params or {})
    if _accepts_geometry(name):
        bp.setdefault("geometry", geometry)
    return X.sweep(name, param, values, base_params=bp, metric=metric)


def tool_compare(name, configs, metric=None) -> dict:
    return X.compare(name, configs, metric=metric)


def _make_experimenter_class(max_iterations=DEFAULT_MAX_ITERATIONS, **class_kwargs):
    if _NOOA_IMPORT_ERROR is not None:
        raise ImportError(f"nooa is required for the agent (Python 3.12): {_NOOA_IMPORT_ERROR}")
    probe_strategy = CodeActStrategy(config=CodeActConfig(max_iterations=max_iterations))

    class _ManifoldExperimenter(Agent, **class_kwargs):
        """Designs and runs NLS/Gross-Pitaevskii experiments on a lumpy-torus-family
        manifold to answer open-ended questions, grounding every claim in a verified run."""
        geometry: str = "lumpy_torus"

        # --- tools (NOOA derives schemas from these signatures + docstrings) ---
        def available_experiments(self) -> dict:
            """List the registered experiments and their parameters/defaults."""
            return tool_available_experiments()

        def available_geometries(self) -> list:
            """List the manifolds experiments can run on (this agent uses self.geometry)."""
            return tool_available_geometries()

        def run_experiment(self, name: str, params: dict = None) -> dict:
            """Run one registered experiment on the current geometry -> {metrics,
            verification, provenance, summary}."""
            return tool_run(name, params, self.geometry)

        def sweep(self, name: str, param: str, values: list, base_params: dict = None,
                  metric: str = None) -> dict:
            """Scan a parameter -> per-value metric + verification + a log-log scaling fit."""
            return tool_sweep(name, param, values, base_params, metric, self.geometry)

        def compare(self, name: str, configs: dict, metric: str = None) -> dict:
            """Run labelled configs of one experiment side by side."""
            return tool_compare(name, configs, metric)

        def is_trustworthy(self, verification: dict) -> bool:
            """True iff every boolean trust-flag in a verification block is True. Call this
            before believing ANY result; if False, refine the run rather than trusting it."""
            return X.verified(verification)

        # --- the on-the-fly experiment generator (NOOA implements the body at runtime) ---
        @strategy(probe_strategy)
        async def probe(self, question: str) -> ExperimentResult:
            """Answer a physics question by DESIGNING and RUNNING a numerical experiment.

            Compose the tools above — or, for a genuinely new experiment, the nls_torus
            primitives (build_surface, CNStepper/RingStepper, fields, diagnostics,
            bogoliubov) — into the smallest run that decides the question. ALWAYS read the
            returned verification block and call is_trustworthy(...) before concluding; if
            it fails, refine (finer grid / smaller dt / more seeds), do not trust the
            number. Return an ExperimentResult whose trustworthy = is_trustworthy(the
            verification you relied on). Never assert physics you did not compute here."""
            ...
    return _ManifoldExperimenter


def ManifoldExperimenter(geometry: str = "lumpy_torus",
                         max_iterations: int = DEFAULT_MAX_ITERATIONS,
                         llm=None, **class_kwargs):
    """Construct a geometry-parameterized experimenter. `llm` may be a NOOA client, a
    model-name string (e.g. 'gemini/gemini-2.5-pro'), or None (-> that default); it is
    resolved to a client here (a live probe() needs GOOGLE_API_KEY). The deterministic
    tool functions run without any LLM."""
    if _NOOA_IMPORT_ERROR is not None:
        raise ImportError(f"nooa is required for the agent (Python 3.12): {_NOOA_IMPORT_ERROR}")
    if isinstance(llm, str) or llm is None:
        from nooa.unifiedllm import get_llm_client
        llm = get_llm_client(llm or "gemini/gemini-2.5-pro")
    cls = _make_experimenter_class(max_iterations=max_iterations, llm=llm, **class_kwargs)
    agent = cls()
    agent.geometry = geometry
    return agent
