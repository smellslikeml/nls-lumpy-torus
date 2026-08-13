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


if __name__ == "__main__":
    mcp.run()   # stdio transport
