#!/usr/bin/env python3
"""nls-torus MCP server — numerically-grounded physics experiments as agent tools.

Exposes the nls_torus experiment toolkit over the Model Context Protocol so an agent
(Claude Code) can run deterministic numerical experiments and reason over the returned
metrics AND verification signals — grounding inference in computation, not priors.

Tools:
  list_experiments()                          -> capabilities (names, params, defaults)
  run_experiment(name, params)                -> {metrics, verification, provenance, summary}
  sweep(name, param, values, base_params, metric)
        -> per-value {metric, verification} + a log-log scaling fit when applicable
  compare(name, configs, metric)              -> side-by-side labelled configs + verdict

Register with Claude Code:
  claude mcp add nls-torus -- python3 /home/thorax/nls_lumpy_torus/mcp_server.py
"""
import inspect
import os
import sys
import numpy as np
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find nls_torus regardless of cwd
from nls_torus import experiment as X
from nls_torus.agent import tool_compose, tool_search_arxiv   # deterministic (no nooa/LLM)

mcp = FastMCP("nls-torus")


def _verified(ver: dict) -> bool:
    """A result is 'verified' if every boolean trust-flag it reports is True."""
    flags = [v for v in ver.values() if isinstance(v, bool)]
    return all(flags) if flags else True


def _capabilities() -> dict:
    caps = {}
    for name, fn in sorted(X.REGISTRY.items()):
        sig = inspect.signature(fn)
        caps[name] = {
            "doc": (fn.__doc__ or "").strip().split("\n")[0],
            "params": {k: (None if p.default is inspect._empty else p.default)
                       for k, p in sig.parameters.items()},
        }
    return caps


@mcp.tool()
def list_experiments() -> dict:
    """List the available numerical experiments, their parameters and defaults."""
    return {"tool": "nls-torus", "experiments": _capabilities()}


@mcp.tool()
def run_experiment(name: str, params: dict = {}) -> dict:
    """Run one experiment. Returns {metrics, verification, provenance, summary}.
    The verification block (conservation drift, Wronskian residual, plateau match, ...)
    tells you whether to trust the numbers."""
    try:
        r = X.run(name, **params)
    except Exception as e:
        return {"error": str(e), "experiment": name, "params": params,
                "available": sorted(X.REGISTRY)}
    r.pop("series", None)
    return r


@mcp.tool()
def sweep(name: str, param: str, values: list, base_params: dict = {},
          metric: str = "") -> dict:
    """Scan one parameter over `values`, holding `base_params` fixed. Returns the chosen
    `metric` (and full metrics + verification) per value, plus a log-log scaling fit
    (exponent) when both axes are positive numeric — e.g. Kibble-Zurek k* ~ tau_Q^b."""
    results = []
    for v in values:
        try:
            r = X.run(name, **{**base_params, param: v})
            m = r["metrics"]
            results.append({"value": v, "metric": (m.get(metric) if metric else None),
                            "metrics": m, "verification": r["verification"],
                            "verified": _verified(r["verification"]), "summary": r["summary"]})
        except Exception as e:
            results.append({"value": v, "error": str(e)})
    out = {"experiment": name, "param": param, "metric": metric or None,
           "results": results,
           "all_verified": all(rr.get("verified", False) for rr in results if "error" not in rr)}
    if metric:
        xs, ys = [], []
        for rr in results:
            v, mv = rr.get("value"), rr.get("metric")
            if isinstance(v, (int, float)) and isinstance(mv, (int, float)) and v > 0 and mv > 0:
                xs.append(v); ys.append(mv)
        if len(xs) >= 3:
            b, c = np.polyfit(np.log(xs), np.log(ys), 1)
            out["scaling"] = {"form": f"{metric} ~ {param}^({b:.3f})",
                              "exponent": float(b), "prefactor": float(np.exp(c))}
    return out


@mcp.tool()
def compare(name: str, configs: dict, metric: str = "") -> dict:
    """Run several labelled configs of the same experiment side by side. `configs` maps
    a label to a params dict. Returns each config's metric + verification + summary, and
    (for a numeric metric) which label extremises it."""
    res = {}
    for label, params in configs.items():
        try:
            r = X.run(name, **params)
            res[label] = {"metric": (r["metrics"].get(metric) if metric else None),
                          "metrics": r["metrics"], "verification": r["verification"],
                          "verified": _verified(r["verification"]), "summary": r["summary"]}
        except Exception as e:
            res[label] = {"error": str(e)}
    out = {"experiment": name, "metric": metric or None, "configs": res}
    if metric:
        scored = {k: v["metric"] for k, v in res.items()
                  if isinstance(v.get("metric"), (int, float))}
        if scored:
            out["max_label"] = max(scored, key=scored.get)
            out["min_label"] = min(scored, key=scored.get)
    return out


@mcp.tool()
def compose(code: str, resolutions: list = [64, 96]) -> dict:
    """Design and run a NEW experiment the registered ones don't express, with
    HARNESS-OWNED verification. `code` must define run(Nx) -> {"observable": <the answer>,
    "mass_series": [diagnostics.mass(U, surf["Mdiag"]) per step], "energy_series": [...],
    "metrics": {...}}. The server runs it at two resolutions in a subprocess sandbox and
    COMPUTES the trust flags itself (mass/energy conservation from the series,
    grid_converged from whether `observable` agrees across resolutions) — you cannot set
    them. Returns {metrics, verification, trustworthy}. The library is preloaded:
    build_surface, ring_grid, wg_chain, CNStepper, RingStepper, fields, diagnostics,
    bogoliubov, geometry.

    Example:
        def run(Nx):
            surf = build_surface(Nx, Nx); step = CNStepper(surf, dt=1e-3)
            U = fields.localized_hump(surf, amp=6.0)
            M, K = surf["Mdiag"], surf["K"]
            ms=[diagnostics.mass(U,M)]; es=[diagnostics.energy(U,K,M,-1.0)]; pk=diagnostics.peak(U); tc=None
            for i in range(1,1500):
                U=step.step(U,sigma3=-1.0); pk=max(pk,diagnostics.peak(U))
                ms.append(diagnostics.mass(U,M)); es.append(diagnostics.energy(U,K,M,-1.0))
                if pk>150: tc=i*1e-3; break
            return {"observable": tc is not None, "mass_series": ms, "energy_series": es,
                    "metrics": {"collapse_time": tc, "peak_max": pk}}
    """
    return tool_compose(code, resolutions=tuple(resolutions))


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> dict:
    """Search arXiv (title/abstract/authors) to cross-check a numerical finding against
    the literature or find the relevant paper — the Townes critical mass, a Kibble-Zurek
    exponent, the analog-Hawking temperature. Returns {id, title, authors, summary, url}
    per hit. No API key. The RUN is the evidence; use this only to sanity-check."""
    return tool_search_arxiv(query, max_results)


def _run_probe_subprocess(question, geometry, max_iterations):
    """Delegate a probe to the generative ManifoldExperimenter in a py3.12+nooa
    subprocess (the server itself may be py3.10). Interpreter via NLS_AGENT_PYTHON."""
    import subprocess
    import json
    import re
    import pathlib
    agent_py = os.environ.get("NLS_AGENT_PYTHON",
                              str(pathlib.Path.home() / "metadata_resolver/.venv/bin/python"))
    if not os.path.exists(agent_py):
        return {"error": f"probe needs a py3.12+nooa interpreter; set NLS_AGENT_PYTHON (tried {agent_py})"}
    repo = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env.update(PROBE_Q=question, PROBE_GEO=geometry, PROBE_ITERS=str(int(max_iterations)))
    if "GOOGLE_API_KEY" not in env:                      # fall back to ~/.bashrc
        try:
            m = re.search(r'^(?:export )?GOOGLE_API_KEY=["\']?([^"\'\n]+)',
                          (pathlib.Path.home() / ".bashrc").read_text(), re.M)
            if m:
                env["GOOGLE_API_KEY"] = m.group(1)
        except Exception:
            pass
    if "GOOGLE_API_KEY" not in env:
        return {"error": "probe needs GOOGLE_API_KEY in the server env"}
    driver = (
        "import asyncio, json, os, dataclasses\n"
        "import nls_torus.agent as A\n"
        "async def main():\n"
        "    ag = A.ManifoldExperimenter(geometry=os.environ['PROBE_GEO'],\n"
        "                                max_iterations=int(os.environ['PROBE_ITERS']))\n"
        "    res = await ag.probe(os.environ['PROBE_Q'])\n"
        "    print('__PROBE__' + json.dumps(dataclasses.asdict(res)))\n"
        "asyncio.run(main())\n"
    )
    try:
        p = subprocess.run([agent_py, "-c", driver], capture_output=True, text=True,
                           timeout=600, env=env, cwd=repo)
    except subprocess.TimeoutExpired:
        return {"error": "probe timed out"}
    for line in p.stdout.splitlines():
        if line.startswith("__PROBE__"):
            return json.loads(line[len("__PROBE__"):])
    return {"error": (p.stderr or "probe produced no result")[-2000:]}


@mcp.tool()
def probe(question: str, geometry: str = "lumpy_torus", max_iterations: int = 16) -> dict:
    """Delegate to the generative ManifoldExperimenter (NOOA/Gemini sub-agent): it designs,
    runs, and HARNESS-verifies a NEW experiment to answer `question` on `geometry`, then
    returns {question, summary, metrics, verification, trustworthy}. Slow (LLM + compute)
    and needs a py3.12+nooa interpreter + GOOGLE_API_KEY (runs in a subprocess). For most
    cases prefer composing directly with the `compose` tool (you are already the reasoner);
    use `probe` to hand the whole design-run-verify loop to the autonomous agent."""
    return _run_probe_subprocess(question, geometry, max_iterations)


if __name__ == "__main__":
    mcp.run()   # stdio transport
