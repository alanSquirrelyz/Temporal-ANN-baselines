# config.sh — Paths to external data and tools. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# annlib_dir       ANNlib repo root; artifacts and test scripts live under it.
# artifact_root    Where generated artifacts are stored (can be on a separate disk).

anndataset_root="${ANNDATASET_ROOT:-/home/zwan/ANNdataset}"
annlib_dir="${ANNLIB_DIR:-/home/zwan/ANNlib}"
artifact_root="${ARTIFACT_ROOT:-${annlib_dir}/artifacts/point_timerange_ann}"
