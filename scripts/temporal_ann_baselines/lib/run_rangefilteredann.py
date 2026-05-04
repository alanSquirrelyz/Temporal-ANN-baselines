from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    build_queries,
    compute_recall,
    compress_scalars_and_ranges,
    directory_size_bytes,
    dtype_token,
    ensure_c_contiguous,
    ensure_float32,
    kpqs,
    load_timerange_artifacts,
    load_vectors,
    milliseconds_per_query,
    normalize_angular,
    process_rss_bytes,
    qps,
    write_tsv,
)

BASELINES_ROOT = Path(__file__).resolve().parents[3]


def load_wrapper(rangefiltered_repo: Path):
    experiments_dir = rangefiltered_repo / "experiments"
    sys.path.insert(0, str(experiments_dir))
    import wrapper as wp  # type: ignore

    return wp


def build_prefilter(wp, metric: str, dtype_name: str, base: np.ndarray, scalars: np.ndarray):
    ctor = wp.prefilter_index_constructor(metric, dtype_name)
    return ctor(base, scalars)


def build_postfilter(wp, metric: str, dtype_name: str, base: np.ndarray, scalars: np.ndarray, args):
    ctor = wp.postfilter_vamana_constructor(metric, dtype_name)
    build_params = wp.BuildParams(args.R, args.L, args.alpha, str(args.index_cache_dir / "postfilter") + "/")
    return ctor(base, scalars, build_params)


def build_tree(wp, metric: str, dtype_name: str, base: np.ndarray, scalars: np.ndarray, args):
    ctor = wp.vamana_range_filter_tree_constructor(metric, dtype_name)
    build_params = wp.BuildParams(args.R, args.L, args.alpha, str(args.index_cache_dir / "tree") + "/")
    return ctor(
        base,
        scalars,
        cutoff=args.cutoff,
        split_factor=args.split_factor,
        build_params=build_params,
    )


def build_super_postfilter(wp, metric: str, dtype_name: str, base: np.ndarray, scalars: np.ndarray, args):
    ctor = wp.super_optimized_postfilter_tree_constructor(metric, dtype_name)
    build_params = wp.BuildParams(
        args.R,
        args.L,
        args.alpha,
        str(args.index_cache_dir / "super-postfilter") + "/",
    )
    return ctor(
        base,
        scalars,
        args.cutoff,
        args.super_split_factor,
        args.super_shift_factor,
        build_params,
    )


def evaluate_batch(ids: np.ndarray, groundtruth: np.ndarray, k: int) -> float:
    return compute_recall(np.asarray(ids, dtype=np.int64), groundtruth, k)


def run_batch(batch_search, queries: np.ndarray, query_ranges: np.ndarray, empty_mask: np.ndarray, k: int):
    ids = np.full((queries.shape[0], k), -1, dtype=np.int64)
    nonempty = np.flatnonzero(~empty_mask)
    if nonempty.size == 0:
        return ids
    filters = [tuple(map(float, row)) for row in query_ranges[nonempty]]
    result_ids, _ = batch_search(queries[nonempty], filters, int(nonempty.size))
    ids[nonempty] = np.asarray(result_ids, dtype=np.int64)
    return ids


def build_index(builder, cache_path: Path | None = None, preprocess_seconds: float = 0.0):
    gc.collect()
    rss_before = process_rss_bytes()
    start = time.perf_counter()
    index = builder()
    index_build_seconds = time.perf_counter() - start
    build_seconds = preprocess_seconds + index_build_seconds
    gc.collect()
    rss_after = process_rss_bytes()
    rss_delta_bytes = max(0, rss_after - rss_before)
    cache_bytes = directory_size_bytes(cache_path) if cache_path is not None else 0
    if cache_bytes > 0:
        space_usage_bytes = cache_bytes
        space_usage_source = "cache_dir_bytes"
    else:
        space_usage_bytes = rss_delta_bytes
        space_usage_source = "rss_delta_bytes"
    return index, {
        "build_seconds": f"{build_seconds:.6f}",
        "index_build_seconds": f"{index_build_seconds:.6f}",
        "preprocess_seconds": f"{preprocess_seconds:.6f}",
        "space_usage_bytes": int(space_usage_bytes),
        "space_usage_source": space_usage_source,
        "rss_delta_bytes": int(rss_delta_bytes),
        "cache_bytes": int(cache_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RangeFilteredANN baselines on ANNlib timerange artifacts.")
    parser.add_argument("--rangefiltered-repo", type=Path, default=BASELINES_ROOT / "RangeFilteredANN")
    parser.add_argument("--dataset", required=True, help="ANNlib dataset spec, e.g. ./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory containing point_timestamps.bin, query_ids.bin, range*.query_ranges.bin, and range*.gt.ibin")
    parser.add_argument("--dist", choices=["L2", "angular"], required=True)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--ranges", type=int, nargs="*", help="Explicit range widths to evaluate")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["prefilter", "postfilter", "vamana_tree", "optimized_postfilter", "three_split", "super_opt_postfilter"],
        choices=["prefilter", "postfilter", "vamana_tree", "optimized_postfilter", "smart_combined", "three_split", "super_opt_postfilter"],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--final-beam-multiplies", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--L", type=int, default=80)
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for Parlay/OpenMP backends")
    parser.add_argument("--cutoff", type=int, default=1000)
    parser.add_argument("--split-factor", type=int, default=2)
    parser.add_argument("--super-split-factor", type=float, default=2.0)
    parser.add_argument("--super-shift-factor", type=float, default=0.5)
    parser.add_argument("--postfiltering-max-beam", type=int, default=10000)
    parser.add_argument("--min-query-to-bucket-ratio", type=float, default=0.05)
    parser.add_argument(
        "--timestamp-mode",
        choices=["raw", "rank"],
        default="raw",
        help="Use raw timestamp filters, or compress timestamps/ranges to dense ranks before indexing.",
    )
    parser.add_argument("--index-cache-dir", type=Path, default=BASELINES_ROOT / "cache" / "rangefilteredann")
    parser.add_argument("--output", type=Path, default=BASELINES_ROOT / "results" / "rangefilteredann_timerange.tsv")
    args = parser.parse_args()

    num_threads = args.threads if args.threads is not None else os.cpu_count() or 1
    num_threads = max(1, int(num_threads))
    os.environ["PARLAY_NUM_THREADS"] = str(num_threads)

    wp = load_wrapper(args.rangefiltered_repo)
    args.index_cache_dir.mkdir(parents=True, exist_ok=True)

    base = load_vectors(args.dataset, max_points=args.max_points if args.max_points > 0 else None)
    artifacts = load_timerange_artifacts(args.artifacts_dir, args.ranges)
    if artifacts.point_timestamps.shape[0] < base.shape[0]:
        raise ValueError("timestamp count is smaller than loaded base size")

    query_ids = artifacts.query_ids
    queries = build_queries(base, query_ids)

    metric = "Euclidian" if args.dist == "L2" else "mips"
    if args.dist == "angular":
        base = normalize_angular(base)
        queries = normalize_angular(queries)
    base = ensure_c_contiguous(base)
    queries = ensure_c_contiguous(queries)

    preprocess_start = time.perf_counter()
    if args.timestamp_mode == "rank":
        compressed = compress_scalars_and_ranges(
            artifacts.point_timestamps[: base.shape[0]],
            artifacts.query_ranges,
        )
        filter_values = compressed.scalars.astype(np.float32)
        query_ranges = compressed.query_ranges
        empty_masks = compressed.empty_masks
    else:
        filter_values = artifacts.point_timestamps[: base.shape[0]].astype(np.float32)
        query_ranges = {width: ranges.astype(np.float32) for width, ranges in artifacts.query_ranges.items()}
        empty_masks = {
            width: np.zeros(ranges.shape[0], dtype=bool)
            for width, ranges in artifacts.query_ranges.items()
        }
    preprocess_seconds = time.perf_counter() - preprocess_start

    dtype_name = dtype_token(base)
    rows: list[dict[str, object]] = []

    built_indexes: dict[str, object] = {}
    build_metrics: dict[str, dict[str, object]] = {}
    if any(m in args.methods for m in {"prefilter"}):
        built_indexes["prefilter"], build_metrics["prefilter"] = build_index(
            lambda: build_prefilter(wp, metric, dtype_name, base, filter_values),
            preprocess_seconds=preprocess_seconds,
        )
    if any(m in args.methods for m in {"postfilter"}):
        built_indexes["postfilter"], build_metrics["postfilter"] = build_index(
            lambda: build_postfilter(wp, metric, dtype_name, base, filter_values, args),
            args.index_cache_dir / "postfilter",
            preprocess_seconds=preprocess_seconds,
        )
    if any(m in args.methods for m in {"vamana_tree", "optimized_postfilter", "smart_combined", "three_split"}):
        built_indexes["tree"], build_metrics["tree"] = build_index(
            lambda: build_tree(wp, metric, dtype_name, base, filter_values, args),
            args.index_cache_dir / "tree",
            preprocess_seconds=preprocess_seconds,
        )
    if any(m in args.methods for m in {"super_opt_postfilter"}):
        built_indexes["super"], build_metrics["super"] = build_index(
            lambda: build_super_postfilter(wp, metric, dtype_name, base, filter_values, args),
            args.index_cache_dir / "super-postfilter",
            preprocess_seconds=preprocess_seconds,
        )

    for range_width in sorted(artifacts.query_ranges):
        gt = artifacts.groundtruth[range_width]
        mapped_ranges = query_ranges[range_width]
        empty_mask = empty_masks[range_width]
        valid_queries = int(np.count_nonzero(~empty_mask))

        if "prefilter" in args.methods:
            qp = wp.build_query_params(k=args.k, beam_size=0, verbose=False)
            start = time.perf_counter()
            ids = run_batch(
                lambda q, f, n: built_indexes["prefilter"].batch_search(q, f, n, qp),
                queries,
                mapped_ranges,
                empty_mask,
                args.k,
            )
            elapsed = time.perf_counter() - start
            query_qps = qps(elapsed, valid_queries)
            rows.append(
                        {
                            "baseline": "rangefilteredann_prefilter",
                            "range": range_width,
                            "threads": num_threads,
                            "beam_size": 0,
                            "final_beam_multiply": 1,
                            **build_metrics["prefilter"],
                    "recall": f"{evaluate_batch(ids, gt, args.k):.6f}",
                    "avg_ms": f"{milliseconds_per_query(elapsed, valid_queries):.6f}",
                    "qps": f"{query_qps:.6f}",
                    "kpqs": f"{kpqs(elapsed, valid_queries):.6f}",
                    "valid_queries": valid_queries,
                }
            )

        if "postfilter" in args.methods:
            for beam_size in args.beam_sizes:
                for final_mult in args.final_beam_multiplies:
                    qp = wp.build_query_params(
                        k=args.k,
                        beam_size=beam_size,
                        final_beam_multiply=final_mult,
                        postfiltering_max_beam=args.postfiltering_max_beam,
                        verbose=False,
                    )
                    start = time.perf_counter()
                    ids = run_batch(
                        lambda q, f, n: built_indexes["postfilter"].batch_search(q, f, n, qp),
                        queries,
                        mapped_ranges,
                        empty_mask,
                        args.k,
                    )
                    elapsed = time.perf_counter() - start
                    query_qps = qps(elapsed, valid_queries)
                    rows.append(
                        {
                            "baseline": "rangefilteredann_postfilter",
                            "range": range_width,
                            "threads": num_threads,
                            "beam_size": beam_size,
                            "final_beam_multiply": final_mult,
                            **build_metrics["postfilter"],
                            "recall": f"{evaluate_batch(ids, gt, args.k):.6f}",
                            "avg_ms": f"{milliseconds_per_query(elapsed, valid_queries):.6f}",
                            "qps": f"{query_qps:.6f}",
                            "kpqs": f"{kpqs(elapsed, valid_queries):.6f}",
                            "valid_queries": valid_queries,
                        }
                    )

        for method_name, query_method in (
            ("vamana_tree", "fenwick"),
            ("optimized_postfilter", "optimized_postfilter"),
            ("smart_combined", "smart_combined"),
            ("three_split", "three_split"),
        ):
            if method_name not in args.methods:
                continue
            for beam_size in args.beam_sizes:
                final_mults = [1] if method_name == "vamana_tree" else args.final_beam_multiplies
                for final_mult in final_mults:
                    qp = wp.build_query_params(
                        k=args.k,
                        beam_size=beam_size,
                        final_beam_multiply=final_mult,
                        postfiltering_max_beam=args.postfiltering_max_beam,
                        min_query_to_bucket_ratio=args.min_query_to_bucket_ratio if method_name in {"smart_combined", "three_split"} else None,
                        verbose=False,
                    )
                    start = time.perf_counter()
                    ids = run_batch(
                        lambda q, f, n, qm=query_method, qp_=qp: built_indexes["tree"].batch_search(q, f, n, qm, qp_),
                        queries,
                        mapped_ranges,
                        empty_mask,
                        args.k,
                    )
                    elapsed = time.perf_counter() - start
                    query_qps = qps(elapsed, valid_queries)
                    rows.append(
                        {
                            "baseline": f"rangefilteredann_{method_name}",
                            "range": range_width,
                            "threads": num_threads,
                            "beam_size": beam_size,
                            "final_beam_multiply": final_mult,
                            **build_metrics["tree"],
                            "recall": f"{evaluate_batch(ids, gt, args.k):.6f}",
                            "avg_ms": f"{milliseconds_per_query(elapsed, valid_queries):.6f}",
                            "qps": f"{query_qps:.6f}",
                            "kpqs": f"{kpqs(elapsed, valid_queries):.6f}",
                            "valid_queries": valid_queries,
                        }
                    )

        if "super_opt_postfilter" in args.methods:
            for beam_size in args.beam_sizes:
                for final_mult in args.final_beam_multiplies:
                    qp = wp.build_query_params(
                        k=args.k,
                        beam_size=beam_size,
                        final_beam_multiply=final_mult,
                        postfiltering_max_beam=args.postfiltering_max_beam,
                        verbose=False,
                    )
                    start = time.perf_counter()
                    ids = run_batch(
                        lambda q, f, n: built_indexes["super"].batch_search(q, f, n, qp),
                        queries,
                        mapped_ranges,
                        empty_mask,
                        args.k,
                    )
                    elapsed = time.perf_counter() - start
                    query_qps = qps(elapsed, valid_queries)
                    rows.append(
                        {
                            "baseline": "rangefilteredann_super_opt_postfilter",
                            "range": range_width,
                            "threads": num_threads,
                            "beam_size": beam_size,
                            "final_beam_multiply": final_mult,
                            **build_metrics["super"],
                            "recall": f"{evaluate_batch(ids, gt, args.k):.6f}",
                            "avg_ms": f"{milliseconds_per_query(elapsed, valid_queries):.6f}",
                            "qps": f"{query_qps:.6f}",
                            "kpqs": f"{kpqs(elapsed, valid_queries):.6f}",
                            "valid_queries": valid_queries,
                        }
                    )

    write_tsv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
