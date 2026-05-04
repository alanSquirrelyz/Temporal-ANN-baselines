# config.sh — Paths to external data and tools. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# annlib_test_dir  ANNlib test directory containing prepare_timerange_dataset.sh.
# annlib_dir       ANNlib root (used by scalability tests).

anndataset_root="${ANNDATASET_ROOT:-/home/zwan/ANNdataset}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/artifacts/point_timerange_ann}"
annlib_test_dir="${ANNLIB_TEST_DIR:-/home/zwan/ANNlib/test}"
annlib_dir="${ANNLIB_DIR:-/home/zwan/ANNlib}"
