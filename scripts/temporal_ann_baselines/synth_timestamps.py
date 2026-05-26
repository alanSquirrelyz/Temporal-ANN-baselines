#!/usr/bin/env python3
"""Synthesize timestamp artifacts for Range-Filtered ANN baselines.

Produces files compatible with scripts/temporal_ann_baselines/lib/common.py:
  point_timestamps.bin       [n:u32][ts:u32 * n]
  query_ids.bin              [m:u32][qid:u32 * m]
  range<W>.query_ranges.bin  [m:u32][(lo:u32, hi:u32) * m]
  range<W>.gt.ibin           [m:u32, k:u32][id:u32 * m * k]

Distributions
-------------
uniform        i.i.d. uniform over [0, N). Sanity baseline.

pareto_recent  Univariate heavy-tail recency. NOTE: wrappers rank-compress
               timestamps before feeding baselines (see README line 77), so
               independent univariate distributions collapse to a random
               permutation in rank space. This mode is kept as a control to
               demonstrate that fact. Ref: Pareto 1896; Mitzenmacher,
               "Brief History of Generative Models for Power Law" (2004).

lognormal_recent Univariate log-normal recency. Same rank-compression caveat
               as pareto_recent. Often a better empirical fit than Pareto for
               page age and video popularity decay. Ref: Limpert et al.,
               BioScience (2001); Cha & Pingali, TON (2009).

topic_drift    Vector-correlated. K-means -> per-cluster Gaussian center.
               Standard "Topics over Time" generative model.
               Ref: Wang & McCallum, "Topics over Time" (KDD 2006);
                    Blei & Lafferty, "Dynamic Topic Models" (ICML 2006).

topic_burst    Per-cluster Hawkes self-exciting bursts (viral topics).
               Ref: Hawkes, Biometrika 1971; Ogata, JASA 1988 (thinning);
                    Zhao et al., "SEISMIC" (KDD 2015).

topic_seasonal Per-cluster non-homogeneous Poisson with diurnal/weekly
               oscillation. Sampled via Lewis-Shedler thinning.
               Ref: Lewis & Shedler, NRL 1979;
                    Karagiannis et al., INFOCOM 2004.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "temporal_ann_baselines"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.common import load_vectors  # noqa: E402


# ---------------------------------------------------------------------------
# I/O writers (must match common.py readers)
# ---------------------------------------------------------------------------

def write_u32_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(values, dtype=np.uint32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D array, got shape {arr.shape}")
    with path.open("wb") as fh:
        np.asarray([arr.size], dtype=np.uint32).tofile(fh)
        arr.tofile(fh)


def write_query_ranges(path: Path, ranges: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(ranges, dtype=np.uint32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (m, 2), got {arr.shape}")
    with path.open("wb") as fh:
        np.asarray([arr.shape[0]], dtype=np.uint32).tofile(fh)
        arr.tofile(fh)


def write_gt(path: Path, gt: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(gt, dtype=np.uint32)
    if arr.ndim != 2:
        raise ValueError(f"expected (m, k), got {arr.shape}")
    with path.open("wb") as fh:
        np.asarray([arr.shape[0], arr.shape[1]], dtype=np.uint32).tofile(fh)
        arr.tofile(fh)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def gen_uniform(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, T, size=n, dtype=np.uint64).astype(np.uint32)


def gen_pareto_recent(
    n: int, T: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """t = T - 1 - normalize(Pareto(alpha)). Recent times have higher density.

    Refs: Pareto (1896); Mitzenmacher (2004) "Brief History of Generative Models
    for Power-Law and Lognormal"; Adamic & Huberman (2002).
    """
    x = rng.pareto(alpha, size=n) + 1.0
    x = np.clip(x, 1.0, np.quantile(x, 0.999))
    norm = (x - 1.0) / max(x.max() - 1.0, 1e-9)
    ts = (T - 1) - (norm * (T - 1)).astype(np.int64)
    return np.clip(ts, 0, T - 1).astype(np.uint32)


def gen_lognormal_recent(
    n: int, T: int, mu_log: float, sigma_log: float, rng: np.random.Generator
) -> np.ndarray:
    """t = T - 1 - normalize(LogNormal(mu_log, sigma_log^2)). Recent times denser.

    Log-normal often fits real-world age distributions (web page age, video
    popularity decay) better than Pareto.

    Refs: Limpert, Stahel & Abbt (2001) "Log-normal distributions across the
    sciences"; Cha & Pingali (2009) "Analyzing the video popularity
    characteristics of large-scale UGC systems"; Mitzenmacher (2004).
    """
    x = rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)
    cap = float(np.quantile(x, 0.999))
    x = np.clip(x, 0.0, cap)
    norm = x / max(cap, 1e-9)
    ts = (T - 1) - (norm * (T - 1)).astype(np.int64)
    return np.clip(ts, 0, T - 1).astype(np.uint32)


def _kmeans_labels(
    base: np.ndarray, num_clusters: int, rng: np.random.Generator,
    sample_size: int = 80_000,
) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        raise RuntimeError(
            "topic_* distributions require scikit-learn: pip install scikit-learn"
        ) from e
    n = base.shape[0]
    fit_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    seed = int(rng.integers(0, 2**31 - 1))
    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=3,
        batch_size=4096,
        max_iter=100,
    )
    km.fit(base[fit_idx].astype(np.float32, copy=False))
    return km.predict(base.astype(np.float32, copy=False)).astype(np.int32)


def gen_topic_drift(
    base: np.ndarray, T: int, num_clusters: int, sigma_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-cluster Gaussian center. Wang & McCallum (KDD 2006) — Gaussian variant."""
    labels = _kmeans_labels(base, num_clusters, rng)
    mu = rng.uniform(0, T, size=num_clusters)
    sigma = np.abs(rng.normal(0.0, sigma_frac * T, size=num_clusters)) + 1.0
    ts = rng.normal(mu[labels], sigma[labels])
    return np.clip(ts.astype(np.int64), 0, T - 1).astype(np.uint32)


def gen_topic_burst(
    base: np.ndarray, T: int, num_clusters: int, mu_rate: float,
    alpha_kernel: float, beta_kernel: float, rng: np.random.Generator,
) -> np.ndarray:
    """Per-cluster Hawkes process. Stability requires alpha_kernel < beta_kernel.

    For each cluster, simulate a 1D Hawkes process over [0, T] via Ogata's
    thinning, then assign timestamps to the cluster's points (with wraparound /
    uniform fill if the simulation produced fewer events than needed).
    """
    if alpha_kernel >= beta_kernel:
        raise ValueError("Hawkes stability requires alpha_kernel < beta_kernel")
    labels = _kmeans_labels(base, num_clusters, rng)
    out = np.empty(base.shape[0], dtype=np.uint32)
    for c in range(num_clusters):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        events = _simulate_hawkes(
            T=T, mu=mu_rate, alpha=alpha_kernel, beta=beta_kernel,
            target=idx.size, rng=rng,
        )
        if events.size < idx.size:
            pad = rng.uniform(0, T, size=idx.size - events.size)
            events = np.concatenate([events, pad])
        elif events.size > idx.size:
            events = rng.choice(events, size=idx.size, replace=False)
        rng.shuffle(events)
        out[idx] = np.clip(events.astype(np.int64), 0, T - 1).astype(np.uint32)
    return out


def _simulate_hawkes(
    T: int, mu: float, alpha: float, beta: float, target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ogata's thinning with O(1) intensity update.

    Maintain S = sum_{t_i <= t_prev} exp(-beta * (t_prev - t_i)), then decay to
    a new time t' via S_new = S * exp(-beta * (t' - t_prev)) and bump by 1
    after accepting an event. Avoids the O(N) sum over history.
    """
    events: list[float] = []
    t_prev = 0.0
    S = 0.0
    lam_bar = mu
    cap = max(2 * target, 64)
    while t_prev < T and len(events) < cap:
        u = rng.uniform()
        if u <= 0.0:
            continue
        w = -math.log(u) / max(lam_bar, 1e-12)
        t = t_prev + w
        if t >= T:
            break
        S *= math.exp(-beta * (t - t_prev))
        lam_t = mu + alpha * S
        if rng.uniform() * lam_bar <= lam_t:
            events.append(t)
            S += 1.0
            lam_bar = mu + alpha * S
        else:
            lam_bar = max(lam_t, mu)
        t_prev = t
    return np.asarray(events, dtype=np.float64)


def gen_topic_seasonal(
    base: np.ndarray, T: int, num_clusters: int, period: float,
    amp: float, rng: np.random.Generator,
) -> np.ndarray:
    """Per-cluster NHPP with sinusoidal seasonality. Each cluster has its own
    phase, so different topics peak at different points in the cycle.
    Lewis-Shedler thinning, NRL 1979.
    """
    labels = _kmeans_labels(base, num_clusters, rng)
    phase = rng.uniform(0, 2 * math.pi, size=num_clusters)
    out = np.empty(base.shape[0], dtype=np.uint32)
    for c in range(num_clusters):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        events = _simulate_nhpp_seasonal(
            T=T, n_target=idx.size, period=period, amp=amp,
            phase=phase[c], rng=rng,
        )
        rng.shuffle(events)
        out[idx] = np.clip(events.astype(np.int64), 0, T - 1).astype(np.uint32)
    return out


def _simulate_nhpp_seasonal(
    T: int, n_target: int, period: float, amp: float, phase: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lewis-Shedler thinning. lambda(t) = base * (1 + amp * sin(2*pi*t/P + phase))."""
    base_rate = n_target / T
    peak_rate = base_rate * (1.0 + amp)
    # Generate candidates at peak rate, then thin.
    expected = int(peak_rate * T * 1.4) + n_target
    inter = rng.exponential(1.0 / max(peak_rate, 1e-12), size=expected)
    t_cands = np.cumsum(inter)
    t_cands = t_cands[t_cands < T]
    if t_cands.size == 0:
        return rng.uniform(0, T, size=n_target)
    rate_at = base_rate * (1.0 + amp * np.sin(2.0 * math.pi * t_cands / period + phase))
    keep_prob = np.clip(rate_at / peak_rate, 0.0, 1.0)
    accept = rng.uniform(size=t_cands.size) < keep_prob
    out = t_cands[accept]
    if out.size < n_target:
        out = np.concatenate([out, rng.uniform(0, T, size=n_target - out.size)])
    elif out.size > n_target:
        out = rng.choice(out, size=n_target, replace=False)
    return out


# ---------------------------------------------------------------------------
# Query / range generation
# ---------------------------------------------------------------------------

def pick_queries(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if m > n:
        raise ValueError(f"num_queries={m} > base size n={n}")
    return rng.choice(n, size=m, replace=False).astype(np.uint32)


def make_ranges_for_width(
    timestamps: np.ndarray, m: int, width: int, center_bias: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate m intervals [lo, hi] that target ~`width` points inside.

    Works in rank space then maps back to original timestamp values, so the
    window indeed contains ~`width` points regardless of the timestamp
    distribution shape.
    """
    n = timestamps.size
    sorted_ts = np.sort(timestamps)
    if center_bias == "uniform":
        center_rank = rng.integers(0, n, size=m)
    elif center_bias == "recent":
        # Beta(5,1) biased to 1.0 -> high ranks -> recent times after sort.
        u = rng.beta(5.0, 1.0, size=m)
        center_rank = np.clip((u * n).astype(np.int64), 0, n - 1)
    else:
        raise ValueError(f"unknown center_bias: {center_bias}")
    half = max(width // 2, 1)
    lo_rank = np.clip(center_rank - half, 0, n - 1)
    hi_rank = np.clip(center_rank + half, 0, n - 1)
    lo = sorted_ts[lo_rank]
    hi = sorted_ts[hi_rank]
    return np.stack([lo, hi], axis=1).astype(np.uint32)


# ---------------------------------------------------------------------------
# Ground truth (brute force, chunked over queries)
# ---------------------------------------------------------------------------

def compute_gt_for_widths(
    base: np.ndarray, queries: np.ndarray, timestamps: np.ndarray,
    ranges_by_width: dict[int, np.ndarray], k: int, dist: str,
    query_chunk: int = 256,
) -> dict[int, np.ndarray]:
    """Brute-force top-k within each query's [lo, hi] window, chunked."""
    if dist == "angular":
        b = _l2_normalize(base)
        q = _l2_normalize(queries)
    elif dist == "L2":
        b = base.astype(np.float32, copy=False)
        q = queries.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown dist: {dist}")

    m = q.shape[0]
    n = b.shape[0]
    invalid = np.iinfo(np.uint32).max
    gt_by_width = {w: np.full((m, k), invalid, dtype=np.uint32)
                   for w in ranges_by_width}

    if dist == "L2":
        b_norm_sq = (b * b).sum(axis=1, dtype=np.float32)

    t0 = time.time()
    for chunk_start in range(0, m, query_chunk):
        chunk_end = min(chunk_start + query_chunk, m)
        qc = q[chunk_start:chunk_end]
        # (chunk, n) distance matrix; smaller = nearer for both dist modes.
        if dist == "L2":
            qc_norm_sq = (qc * qc).sum(axis=1, dtype=np.float32, keepdims=True)
            d_chunk = qc_norm_sq + b_norm_sq[None, :] - 2.0 * (qc @ b.T)
        else:
            d_chunk = -(qc @ b.T)  # rank-preserving for normalized cosine
        for j_local, j in enumerate(range(chunk_start, chunk_end)):
            for w, ranges in ranges_by_width.items():
                lo, hi = int(ranges[j, 0]), int(ranges[j, 1])
                mask = (timestamps >= lo) & (timestamps <= hi)
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                d = d_chunk[j_local, idx]
                kk = min(k, idx.size)
                top = np.argpartition(d, kk - 1)[:kk]
                top = top[np.argsort(d[top])]
                gt_by_width[w][j, :kk] = idx[top]
        elapsed = time.time() - t0
        print(f"  gt chunk {chunk_end}/{m}  elapsed {elapsed:.1f}s", flush=True)
    return gt_by_width


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_ranges(s: str) -> list[int]:
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return [int(p) for p in parts]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="<path>:<format>, e.g. /data/zshen055/ANN/Yandex-DEEP/base.1B.fbin:fbin")
    p.add_argument("--out-dir", required=True, help="Where to write the 4 artifact files")
    p.add_argument("--max-points", type=int, default=1_000_000)
    p.add_argument("--num-queries", type=int, default=1000)
    p.add_argument("--query-k", type=int, default=10)
    p.add_argument("--ranges", type=str, default="100 1000 10000 100000",
                   help="Space- or comma-separated target window widths in # of points")
    p.add_argument("--dist", choices=["L2", "angular"], required=True)
    p.add_argument("--distribution", required=True,
                   choices=["uniform", "pareto_recent", "lognormal_recent",
                            "topic_drift", "topic_burst", "topic_seasonal"])
    p.add_argument("--center-bias", choices=["uniform", "recent"], default="uniform")
    p.add_argument("--seed", type=int, default=42)

    # Distribution-specific knobs
    p.add_argument("--pareto-alpha", type=float, default=1.2,
                   help="pareto_recent: tail index. Smaller=heavier tail.")
    p.add_argument("--lognormal-mu", type=float, default=0.0,
                   help="lognormal_recent: mean of underlying normal (log scale)")
    p.add_argument("--lognormal-sigma", type=float, default=1.0,
                   help="lognormal_recent: std of underlying normal")
    p.add_argument("--num-clusters", type=int, default=128,
                   help="topic_*: k-means cluster count")
    p.add_argument("--sigma-frac", type=float, default=0.05,
                   help="topic_drift: cluster temporal std as fraction of T")
    p.add_argument("--hawkes-mu", type=float, default=0.5)
    p.add_argument("--hawkes-alpha", type=float, default=0.6)
    p.add_argument("--hawkes-beta", type=float, default=1.0,
                   help="must be > hawkes-alpha for stability")
    p.add_argument("--seasonal-period-frac", type=float, default=1.0 / 7.0,
                   help="topic_seasonal: period as fraction of T (default ~weekly over T)")
    p.add_argument("--seasonal-amp", type=float, default=0.6,
                   help="topic_seasonal: amplitude in [0,1)")

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    print(f"[load] {args.dataset}  max_points={args.max_points}", flush=True)
    base = load_vectors(args.dataset, max_points=args.max_points)
    n, d = base.shape
    print(f"[load] n={n} d={d} dtype={base.dtype}", flush=True)

    # We use T = n so timestamps live in [0, n) and rank compression is a no-op
    # for uniform; other distributions still vary how points are *assigned* to
    # timestamps (which matters when they're correlated with vectors).
    T = n

    print(f"[gen]  distribution={args.distribution}", flush=True)
    t0 = time.time()
    if args.distribution == "uniform":
        ts = gen_uniform(n, T, rng)
    elif args.distribution == "pareto_recent":
        ts = gen_pareto_recent(n, T, alpha=args.pareto_alpha, rng=rng)
    elif args.distribution == "lognormal_recent":
        ts = gen_lognormal_recent(n, T, mu_log=args.lognormal_mu,
                                  sigma_log=args.lognormal_sigma, rng=rng)
    elif args.distribution == "topic_drift":
        ts = gen_topic_drift(base, T, num_clusters=args.num_clusters,
                             sigma_frac=args.sigma_frac, rng=rng)
    elif args.distribution == "topic_burst":
        ts = gen_topic_burst(base, T, num_clusters=args.num_clusters,
                             mu_rate=args.hawkes_mu,
                             alpha_kernel=args.hawkes_alpha,
                             beta_kernel=args.hawkes_beta, rng=rng)
    elif args.distribution == "topic_seasonal":
        ts = gen_topic_seasonal(base, T, num_clusters=args.num_clusters,
                                period=args.seasonal_period_frac * T,
                                amp=args.seasonal_amp, rng=rng)
    else:
        raise AssertionError(args.distribution)
    print(f"[gen]  done in {time.time() - t0:.1f}s", flush=True)

    print("[query] sampling query ids", flush=True)
    qids = pick_queries(n, args.num_queries, rng)
    query_vecs = base[qids]

    widths = parse_ranges(args.ranges)
    print(f"[range] widths={widths} center_bias={args.center_bias}", flush=True)
    ranges_by_width = {
        w: make_ranges_for_width(ts, args.num_queries, w, args.center_bias, rng)
        for w in widths
    }

    print(f"[gt]   brute force k={args.query_k} dist={args.dist}", flush=True)
    gt_by_width = compute_gt_for_widths(
        base=base, queries=query_vecs, timestamps=ts,
        ranges_by_width=ranges_by_width, k=args.query_k, dist=args.dist,
    )

    print(f"[write] {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_u32_array(out_dir / "point_timestamps.bin", ts)
    write_u32_array(out_dir / "query_ids.bin", qids)
    for w in widths:
        write_query_ranges(out_dir / f"range{w}.query_ranges.bin", ranges_by_width[w])
        write_gt(out_dir / f"range{w}.gt.ibin", gt_by_width[w])

    # Manifest for reproducibility.
    manifest = out_dir / "manifest.txt"
    with manifest.open("w") as fh:
        for k, v in vars(args).items():
            fh.write(f"{k}={v}\n")
    print(f"[done] wrote artifacts to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
