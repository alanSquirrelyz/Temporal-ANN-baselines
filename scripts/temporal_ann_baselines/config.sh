# config.sh — Paths to external data and outputs. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# annlib_test_dir  Path to ANNlib/test (contains prepare_timerange_dataset.sh).
# artifact_root    Where ANNlib artifacts are stored (can be on a separate disk).
# cache_root       Where generated index/cache files are stored.
# result_root      Where benchmark result TSVs are written.

# python_bin="${PYTHON_BIN:-/home/zwan018/autoresearch/.venv312/bin/python}"
python_bin="python3"

anndataset_root="${ANNDATASET_ROOT:-/home/zwan018/ANNlib/test/ANNdataset}"
annlib_test_dir="${ANNLIB_TEST_DIR:-/home/zwan018/ANNlib/test}"
artifact_root="${ARTIFACT_ROOT:-/data/zwan018/TimeStampANN/point_timerange_ann}"
cache_root="${CACHE_ROOT:-/data/zwan018/TimeStampANN/baseline_cache}"
result_root="${RESULT_ROOT:-${repo_root}/results}"
