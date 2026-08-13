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
    from nooa import Agent, strategy, InvariantError
    from nooa.strategies import CodeActStrategy
    from nooa.config import CodeActConfig
    _NOOA_IMPORT_ERROR = None
except ImportError as _e:                       # nooa needs Python 3.12
    Agent = object; strategy = None; CodeActStrategy = None; CodeActConfig = None
    InvariantError = Exception
    _NOOA_IMPORT_ERROR = _e


def _has_real_content(m):
    """True iff `m` holds at least one non-empty value (rejects {} and {'x': []} etc.)."""
    if not isinstance(m, dict) or not m:
        return False
    return any(v not in (None, [], {}, (), "") for v in m.values())


def _require_verification(agent, result, call):
    """NOOA postcondition on probe(): reject a result that is empty or self-certifies
    without real content — routed back for correction. Closes the 'empty verification
    passes vacuously' hole AND the 'vacuous flag over zero results' hole a live run
    exposed. (A structural gate can still be gamed; the durable fix is harness-COMPUTED
    verification — which the registered experiments have and compose() should adopt.)"""
    ver = getattr(result, "verification", None) or {}
    met = getattr(result, "metrics", None) or {}
    flags = [v for v in ver.values() if isinstance(v, bool)]
    if not flags:
        raise InvariantError(
            "ExperimentResult.verification is EMPTY. Populate it with >=1 boolean trust-flag "
            "you actually computed (mass_conserved from diagnostics.conservation_drift, and "
            "grid_converged from a two-resolution run), set trustworthy=is_trustworthy(...).")
    if not _has_real_content(met):
        raise InvariantError(
            "metrics is empty or holds only empty values — your run produced NO numbers. "
            "Check compose()'s return for an 'error' key, fix the code, and re-run so metrics "
            "and verification reflect actual computed results (do not self-certify on no data).")
    if getattr(result, "trustworthy", False) and not all(flags):
        raise InvariantError(
            "trustworthy=True but a verification flag is False. Set trustworthy=False, or "
            "refine the run (finer grid / smaller dt / more seeds) until the checks pass.")

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


def _rel_drift(series):
    if not series:
        return None
    s0 = series[0]
    return max(abs(v - s0) / (abs(s0) + 1e-300) for v in series)


def _observables_agree(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 0.1 * max(abs(a), abs(b), 1e-9)
    return a == b


def tool_compose(code, resolutions=(64, 96), mass_tol=1e-6, energy_tol=1e-2, timeout=180.0) -> dict:
    """HARNESS-OWNED verification. The caller's `code` must define a function

        run(Nx) -> {"observable": <number|bool|label answering the question>,
                    "mass_series":   [diagnostics.mass(U,M) at each step],
                    "energy_series": [diagnostics.energy(...) at each step],  # optional
                    "metrics": {...}}                                          # optional extras

    The harness runs it in a subprocess sandbox at two resolutions and COMPUTES the trust
    flags itself — mass_conserved / energy_conserved from the returned series, and
    grid_converged from whether `observable` agrees across resolutions. The caller cannot
    set these flags (that closes the self-certification hole). Returns
    {metrics, verification, trustworthy, provenance} with trustworthy = all flags True.

    Trust boundary: runs agent-authored physics code — the subprocess gives crash/time
    isolation, not a security sandbox; deploy the agent in a contained environment."""
    import os
    import subprocess
    import sys
    import json
    import textwrap
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = [int(r) for r in resolutions]
    driver = (
        "import json\n"
        "import numpy as np\n"
        "import nls_torus as nt\n"
        "from nls_torus import build_surface, ring_grid, wg_chain, CNStepper, RingStepper\n"
        "from nls_torus import fields, diagnostics, bogoliubov, geometry\n"
        + textwrap.dedent(code) + "\n"
        "assert callable(run), 'define run(Nx) -> dict'\n"
        "out = {}\n"
        f"for _Nx in {res!r}:\n"
        "    _r = run(_Nx)\n"
        "    out[str(_Nx)] = {'observable': _r.get('observable'),\n"
        "                     'mass_series': [float(v) for v in _r.get('mass_series', [])],\n"
        "                     'energy_series': [float(v) for v in _r.get('energy_series', [])],\n"
        "                     'metrics': nt.experiment._clean(_r.get('metrics', {}))}\n"
        "print('__RUNS__' + json.dumps(out))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run([sys.executable, "-c", driver], capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=repo)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    runs = None
    for line in p.stdout.splitlines():
        if line.startswith("__RUNS__"):
            runs = json.loads(line[len("__RUNS__"):])
            break
    if runs is None:
        return {"error": (p.stderr or "run(Nx) did not execute or produced no output")[-2000:]}

    lo, hi = str(res[0]), str(res[-1])
    ver = {}
    md = _rel_drift(runs[hi]["mass_series"])
    if md is not None:
        ver["mass_conserved"] = md < mass_tol
    ed = _rel_drift(runs[hi]["energy_series"])
    if ed is not None:
        ver["energy_conserved"] = ed < energy_tol
    ver["grid_converged"] = _observables_agree(runs[lo]["observable"], runs[hi]["observable"])
    metrics = dict(runs[hi]["metrics"])
    metrics.setdefault("observable", runs[hi]["observable"])
    metrics["observable_by_resolution"] = {lo: runs[lo]["observable"], hi: runs[hi]["observable"]}
    if md is not None:
        metrics["mass_drift"] = md
    return {"metrics": metrics, "verification": ver,
            "trustworthy": bool(ver) and all(ver.values()),
            "provenance": {"resolutions": res, "harness_computed_verification": True}}


def tool_search_arxiv(query, max_results=5) -> dict:
    """Search arXiv (title/abstract/authors) to cross-check a numerical finding against
    the literature or locate the relevant paper. Deterministic, no API key."""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results})
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read()
    except Exception as e:
        return {"error": str(e), "query": query}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(data).findall("a:entry", ns):
        aid = e.findtext("a:id", "", ns)
        out.append({"id": aid.rsplit("/", 1)[-1],
                    "title": " ".join(e.findtext("a:title", "", ns).split()),
                    "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:6],
                    "summary": " ".join(e.findtext("a:summary", "", ns).split())[:300],
                    "url": aid})
    return {"query": query, "results": out}


def _make_experimenter_class(max_iterations=DEFAULT_MAX_ITERATIONS, **class_kwargs):
    if _NOOA_IMPORT_ERROR is not None:
        raise ImportError(f"nooa is required for the agent (Python 3.12): {_NOOA_IMPORT_ERROR}")
    probe_strategy = CodeActStrategy(config=CodeActConfig(
        max_iterations=max_iterations, postconditions=[_require_verification]))

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

        def compose(self, code: str, resolutions: tuple = (64, 96),
                    timeout: float = 180.0) -> dict:
            """Run a NEW experiment for questions the registered ones don't express, with
            HARNESS-OWNED verification. Your `code` defines ONE function run(Nx) that sets
            up + evolves the experiment at grid resolution Nx and RETURNS a dict:

                {"observable": <the answer: a number, bool, or label>,
                 "mass_series": [diagnostics.mass(U, surf["Mdiag"]) at each step],
                 "energy_series": [diagnostics.energy(U, K, M, sigma3) at each step],  # optional
                 "metrics": {...}}                                                     # optional

            The HARNESS runs it at two resolutions and COMPUTES the trust flags itself
            (mass/energy conservation from your series, grid_converged from whether
            `observable` agrees across resolutions) — you do NOT set them, so you cannot
            self-certify. It returns {metrics, verification, trustworthy}; copy that
            verification + trustworthy VERBATIM into your ExperimentResult.

            The library is preloaded in the sandbox: build_surface, ring_grid, wg_chain,
            CNStepper, RingStepper, fields, diagnostics, bogoliubov, geometry.

            Example:
                def run(Nx):
                    surf = build_surface(Nx, Nx)
                    step = CNStepper(surf, dt=1e-3)
                    U = fields.localized_hump(surf, amp=6.0, wx=0.35, wth=0.35)
                    M, K = surf["Mdiag"], surf["K"]
                    ms=[diagnostics.mass(U,M)]; es=[diagnostics.energy(U,K,M,-1.0)]
                    pk=diagnostics.peak(U); tc=None
                    for i in range(1, 1500):
                        U = step.step(U, sigma3=-1.0); pk=max(pk, diagnostics.peak(U))
                        ms.append(diagnostics.mass(U,M)); es.append(diagnostics.energy(U,K,M,-1.0))
                        if pk > 150: tc = i*1e-3; break
                    return {"observable": tc is not None, "mass_series": ms,
                            "energy_series": es, "metrics": {"collapse_time": tc, "peak_max": pk}}

            Prefer run_experiment/sweep/compare when they fit; use compose only for
            genuinely new setups."""
            return tool_compose(code, resolutions=resolutions, timeout=timeout)

        def search_arxiv(self, query: str, max_results: int = 5) -> dict:
            """Search arXiv (title/abstract/authors) to check a numerical finding against
            the literature or find the relevant paper — e.g. the Townes critical mass, a
            Kibble-Zurek exponent, the analog-Hawking temperature. Returns {id, title,
            authors, summary, url} per hit. No API key. Use it to sanity-check an expected
            value, but the RUN is the evidence — cite the number you computed."""
            return tool_search_arxiv(query, max_results)

        def is_trustworthy(self, verification: dict) -> bool:
            """True iff every boolean trust-flag in a verification block is True. Call this
            before believing ANY result; if False, refine the run rather than trusting it."""
            return X.verified(verification)

        # --- the on-the-fly experiment generator (NOOA implements the body at runtime) ---
        @strategy(probe_strategy)
        async def probe(self, question: str) -> ExperimentResult:
            """Answer a physics question by DESIGNING and RUNNING a numerical experiment.

            Use run_experiment/sweep/compare when a registered experiment fits; for a
            genuinely NEW setup, write it against the library and run it via compose(code).
            Pick the smallest run that decides the question. ALWAYS read the
            returned verification block and call is_trustworthy(...) before concluding; if
            it fails, refine (finer grid / smaller dt / more seeds), do not trust the
            number. You may search_arxiv(...) to check an expected value or universality
            class against the literature — but the RUN is the evidence.

            The returned ExperimentResult MUST populate `metrics` and `verification`.
            When you use compose(), it returns harness-COMPUTED {metrics, verification,
            trustworthy}: copy those three fields VERBATIM into your ExperimentResult — do
            not recompute, second-guess, or override them (the harness owns the trust
            flags). When you use run_experiment/sweep, set trustworthy =
            is_trustworthy(the verification it returned). An empty verification is
            UNVERIFIED. Never assert physics you did not compute and verify here."""
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
