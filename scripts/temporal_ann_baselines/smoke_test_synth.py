#!/usr/bin/env python3
"""End-to-end smoke test for synth_timestamps.py.

Creates a tiny in-memory dataset, invokes synth_timestamps.py as a subprocess
on each available distribution, reads the artifacts back via lib/common.py,
and validates shapes / dtypes / range-membership / GT correctness.

Usage:
    python3 scripts/temporal_ann_baselines/smoke_test_synth.py

Exit code 0 = all distributions PASS; nonzero = at least one FAIL.
Runs in a few seconds with no external data. Requires only numpy (sklearn is
optional and only needed for the topic_* distributions).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "temporal_ann_baselines"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.common import load_timerange_artifacts  # noqa: E402

SYNTH = SCRIPTS_DIR / "synth_timestamps.py"

N = 2000
DIM = 32
NUM_QUERIES = 20
QUERY_K = 5
RANGES = "10 50 200"
RANGE_WIDTHS = [10, 50, 200]


def make_synth_fbin(path: Path, n: int, dim: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    with path.open("wb") as fh:
        np.asarray([n, dim], dtype=np.uint32).tofile(fh)
        vectors.tofile(fh)


def run_synth(dataset_spec: str, out_dir: Path, distribution: str) -> tuple[bool, str]:
    cmd = [
        sys.executable, str(SYNTH),
        "--dataset", dataset_spec,
        "--out-dir", str(out_dir),
        "--max-points", str(N),
        "--num-queries", str(NUM_QUERIES),
        "--query-k", str(QUERY_K),
        "--ranges", RANGES,
        "--dist", "L2",
        "--distribution", distribution,
        "--seed", "7",
        "--num-clusters", "8",
        "--sigma-frac", "0.05",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"synth_timestamps.py exited {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    return True, proc.stdout


def validate_artifacts(artifacts_dir: Path, base: np.ndarray) -> list[str]:
    """Return list of failure strings; empty list = pass."""
    failures = []
    arts = load_timerange_artifacts(artifacts_dir, ranges=RANGE_WIDTHS)

    if arts.point_timestamps.shape != (N,):
        failures.append(f"point_timestamps shape {arts.point_timestamps.shape} != ({N},)")
    if arts.point_timestamps.dtype != np.uint32:
        failures.append(f"point_timestamps dtype {arts.point_timestamps.dtype} != uint32")
    if arts.query_ids.shape != (NUM_QUERIES,):
        failures.append(f"query_ids shape {arts.query_ids.shape} != ({NUM_QUERIES},)")

    INVALID = np.iinfo(np.uint32).max

    for w in RANGE_WIDTHS:
        ranges = arts.query_ranges[w]
        gt = arts.groundtruth[w]
        if ranges.shape != (NUM_QUERIES, 2):
            failures.append(f"range{w} ranges shape {ranges.shape}")
            continue
        if gt.shape != (NUM_QUERIES, QUERY_K):
            failures.append(f"range{w} gt shape {gt.shape}")
            continue
        # Check: every non-sentinel GT id is inside its window AND truly
        # nearest among all points in that window.
        ts = arts.point_timestamps
        queries = base[arts.query_ids.astype(np.int64)]
        for qi in range(NUM_QUERIES):
            lo, hi = int(ranges[qi, 0]), int(ranges[qi, 1])
            window = np.flatnonzero((ts >= lo) & (ts <= hi))
            gt_ids = gt[qi]
            valid_gt = gt_ids[gt_ids != INVALID]
            if valid_gt.size == 0:
                # Empty windows are legal (degenerate cases).
                continue
            # Membership
            if not np.all(np.isin(valid_gt, window)):
                bad = valid_gt[~np.isin(valid_gt, window)]
                failures.append(
                    f"range{w} q{qi}: gt id {bad[0]} not in window [{lo},{hi}]"
                )
                break
            # Correctness: brute-force top-k inside window must equal GT for the
            # first min(k, |window|) positions (modulo ties).
            wnd_d = np.sum((base[window] - queries[qi]) ** 2, axis=1)
            kk = min(QUERY_K, window.size)
            ideal = window[np.argsort(wnd_d)[:kk]]
            actual = valid_gt[:kk]
            # Compare as sets (tie-tolerant up to distance equality).
            if set(int(x) for x in ideal) != set(int(x) for x in actual):
                # Tie-tolerant compare: same distance vector?
                ideal_d = np.sort(wnd_d[np.isin(window, ideal)])
                actual_d = np.sort(
                    np.array([wnd_d[np.flatnonzero(window == int(a))[0]] for a in actual])
                )
                if not np.allclose(ideal_d, actual_d, atol=1e-5):
                    failures.append(
                        f"range{w} q{qi}: GT mismatch. ideal={ideal.tolist()} "
                        f"actual={actual.tolist()} d_ideal={ideal_d.tolist()} "
                        f"d_actual={actual_d.tolist()}"
                    )
                    break
    return failures


def main() -> int:
    if not SYNTH.exists():
        print(f"FAIL: {SYNTH} does not exist", file=sys.stderr)
        return 2
    print(f"using interpreter: {sys.executable}")
    print(f"python version: {sys.version.split()[0]}")
    has_sklearn = True
    try:
        import sklearn  # noqa: F401
    except ImportError:
        has_sklearn = False
        print("(sklearn not installed — topic_* distributions will be skipped)")

    distributions = ["uniform", "pareto_recent", "lognormal_recent"]
    if has_sklearn:
        distributions += ["topic_drift", "topic_burst", "topic_seasonal"]

    workdir = Path(tempfile.mkdtemp(prefix="synth_smoke_"))
    print(f"workdir: {workdir}")
    try:
        # Build a synthetic .fbin and load it back as base.
        base_path = workdir / "base.fbin"
        make_synth_fbin(base_path, N, DIM)
        with base_path.open("rb") as fh:
            np.fromfile(fh, dtype=np.uint32, count=2)
            base = np.fromfile(fh, dtype=np.float32, count=N * DIM).reshape(N, DIM)
        dataset_spec = f"{base_path}:fbin"

        any_fail = False
        for dist in distributions:
            print(f"\n--- distribution={dist} ---")
            out_dir = workdir / f"out_{dist}"
            ok, log = run_synth(dataset_spec, out_dir, dist)
            if not ok:
                print(f"FAIL [{dist}] synth_timestamps.py failed:\n{log}")
                any_fail = True
                continue
            failures = validate_artifacts(out_dir, base)
            if failures:
                print(f"FAIL [{dist}] {len(failures)} issue(s):")
                for f in failures[:5]:
                    print(f"  - {f}")
                any_fail = True
            else:
                print(f"PASS [{dist}]")
        print()
        if any_fail:
            print("SMOKE TEST: FAIL (workdir kept for inspection: %s)" % workdir)
            return 1
        print("SMOKE TEST: PASS")
        return 0
    finally:
        # Only clean up on success to allow debugging on failure.
        pass


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        # Clean up tempdir only on full success.
        for p in Path(tempfile.gettempdir()).glob("synth_smoke_*"):
            try:
                shutil.rmtree(p)
            except OSError:
                pass
    raise SystemExit(rc)
