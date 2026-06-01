#!/usr/bin/env python3
"""Plot thread-scalability curves from run_scalability.sh output.

Reads a TSV with columns: baseline, threads, build_seconds, qps
(qps may be "NA" for the serial-query baselines).

Produces two figures next to the input file:
  * build_scalability.png  — build time vs threads (left) and parallel
                             speedup S(T)=T1/T_T vs threads with the ideal
                             linear-speedup reference (right), all 5 baselines.
  * query_scalability.png  — query throughput (QPS) vs threads, only for the
                             baselines whose query path is parallel
                             (RangeFilteredANN, UNIFY).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARKERS = {"dsg": "o", "serf": "s", "irangegraph": "^",
           "rangefilteredann": "D", "unify": "v"}
LABELS = {"dsg": "DSG", "serf": "SeRF", "irangegraph": "iRangeGraph",
          "rangefilteredann": "RangeFilteredANN", "unify": "UNIFY"}


def _num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def load(path: Path):
    build = defaultdict(list)   # baseline -> [(threads, seconds)]
    query = defaultdict(list)   # baseline -> [(threads, qps)]
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            b = row["baseline"]
            t = _num(row["threads"])
            if t is None:
                continue
            bs = _num(row.get("build_seconds"))
            if bs is not None:
                build[b].append((t, bs))
            q = _num(row.get("qps"))
            if q is not None:
                query[b].append((t, q))
    for d in (build, query):
        for b in d:
            d[b].sort()
    return build, query


def plot_build(build, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for b in sorted(build):
        pts = build[b]
        if not pts:
            continue
        ts = [t for t, _ in pts]
        secs = [s for _, s in pts]
        m, lab = MARKERS.get(b, "o"), LABELS.get(b, b)
        ax1.plot(ts, secs, marker=m, label=lab)
        base = secs[0]            # time at the smallest thread count
        speedup = [base / s for s in secs]
        ax2.plot(ts, speedup, marker=m, label=lab)

    # ideal linear speedup reference, normalised to the smallest thread count
    all_t = sorted({t for pts in build.values() for t, _ in pts})
    if all_t:
        t0 = all_t[0]
        ax2.plot(all_t, [t / t0 for t in all_t], "k--", alpha=0.6,
                 label="ideal (linear)")

    ax1.set(xscale="log", yscale="log", xlabel="threads",
            ylabel="index build time (s)", title="Build time vs threads")
    ax1.set_xticks(all_t); ax1.set_xticklabels([str(t) for t in all_t])
    ax1.grid(True, which="both", ls=":", alpha=0.5); ax1.legend()

    ax2.set(xscale="log", yscale="log", xlabel="threads",
            ylabel="speedup  S(T)=T1 / Tn", title="Build speedup vs threads")
    ax2.set_xticks(all_t); ax2.set_xticklabels([str(t) for t in all_t])
    ax2.grid(True, which="both", ls=":", alpha=0.5); ax2.legend()

    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def plot_query(query, out: Path):
    parallel = {b: pts for b, pts in query.items() if pts}
    if not parallel:
        print("no parallel-query data (qps all NA) — skipping query figure")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    all_t = sorted({t for pts in parallel.values() for t, _ in pts})
    for b in sorted(parallel):
        ts = [t for t, _ in parallel[b]]
        qs = [q for _, q in parallel[b]]
        ax.plot(ts, qs, marker=MARKERS.get(b, "o"), label=LABELS.get(b, b))
    ax.set(xscale="log", xlabel="threads", ylabel="query throughput (QPS)",
           title="Query QPS vs threads")
    ax.set_xticks(all_t); ax.set_xticklabels([str(t) for t in all_t])
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    default_in = here.parents[1] / "results" / "scalability" / "scalability.tsv"
    ap.add_argument("--input", type=Path, default=default_in)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    outdir = args.outdir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    build, query = load(args.input)
    plot_build(build, outdir / "build_scalability.png")
    plot_query(query, outdir / "query_scalability.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
