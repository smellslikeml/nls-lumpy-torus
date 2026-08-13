#!/usr/bin/env python3
"""run_experiment — a typed, self-describing tool for numerically-grounded inference.

An agent (or a human) discovers what can be run, runs it, and reasons over the
returned NUMBERS and VERIFICATION signals rather than its priors:

    python3 experiment_tool.py --list                      # capabilities (JSON)
    python3 experiment_tool.py run collapse '{"amp":6,"sigma5":0.4}'   # run (JSON)
    python3 experiment_tool.py run expanding_cosmos --figure out.png

Every result carries {metrics, verification, provenance, summary}. The `verification`
block (conservation drift, Wronskian residual, plateau match, ...) is the point: it
lets the caller distinguish a trustworthy result from a plausible-looking artifact,
and it is deterministic and reproducible (fixed params -> fixed numbers).
"""
import argparse
import inspect
import json
import sys

from nls_torus import experiment as X


def capabilities():
    caps = {}
    for name, fn in sorted(X.REGISTRY.items()):
        sig = inspect.signature(fn)
        params = {k: (None if p.default is inspect._empty else p.default)
                  for k, p in sig.parameters.items()}
        caps[name] = {"doc": (fn.__doc__ or "").strip().split("\n")[0], "params": params}
    return caps


def render_figure(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = result.get("series", {})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if "beta2" in s:
        ax.plot(s["ks"], s["beta2"], "-o", color="#b83280", ms=4, label=r"$|\beta_k|^2$")
        ax.plot(s["ks"], s["S_k"], "-s", color="#2b6cb0", ms=3, alpha=.7, label=r"$S_k(t_{obs})$")
        ax.set_xlabel("k"); ax.set_ylabel("produced pairs / structure factor")
        ax.legend()
    elif "mass_hist" in s:
        ax.plot(s["mass_hist"], color="#2b6cb0"); ax.set_xlabel("step"); ax.set_ylabel("mass")
    ax.set_title(result["experiment"] + " — " + result["summary"], fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="run_experiment tool")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--list", action="store_true", help="print capabilities as JSON")
    pr = sub.add_parser("run")
    pr.add_argument("name")
    pr.add_argument("params", nargs="?", default="{}", help="JSON dict of parameters")
    pr.add_argument("--figure", default=None, help="optional path to render a PNG")
    args = ap.parse_args(argv)

    if args.list or args.cmd is None:
        print(json.dumps({"tool": "run_experiment", "experiments": capabilities()}, indent=2))
        return 0

    params = json.loads(args.params)
    try:
        result = X.run(args.name, **params)
    except Exception as e:
        print(json.dumps({"error": str(e), "experiment": args.name, "params": params}))
        return 1
    if args.figure:
        result["figure_path"] = render_figure(result, args.figure)
    # drop bulky raw series from the printed JSON (keep a length note)
    result["series"] = {k: (f"[{len(v)} values]" if isinstance(v, list) else v)
                        for k, v in result.get("series", {}).items()}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
