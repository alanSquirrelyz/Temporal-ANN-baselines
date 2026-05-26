# config.sh — Paths to external data and outputs. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# annlib_test_dir  Path to ANNlib/test (contains prepare_timerange_dataset.sh).
# artifact_root    Where ANNlib artifacts are stored (can be on a separate disk).
# cache_root       Where generated index/cache files are stored.
# result_root      Where benchmark result TSVs are written.

python_bin="${PYTHON_BIN:-/home/yliu908/.venv312/bin/python}"

anndataset_root="${ANNDATASET_ROOT:-/data/zshen055/ANN}"
annlib_test_dir="${ANNLIB_TEST_DIR:-/home/yliu908/Dev/Temporal-ANN-baselines/ANNlib/test}"
artifact_root="${ARTIFACT_ROOT:-/data/yliu908/temporal-ANN/point_timerange_ann}"
cache_root="${CACHE_ROOT:-/data/yliu908/temporal-ANN/baseline_cache}"
result_root="${RESULT_ROOT:-${repo_root}/results}"
