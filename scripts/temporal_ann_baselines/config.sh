# config.sh — Paths to external data and tools. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# annlib_dir       ANNlib root; also the default home for generated artifacts.
# annlib_test_dir  ANNlib test directory containing prepare_timerange_dataset.sh.
# artifact_root    Where ANNlib artifacts are stored (can be on a separate disk).

anndataset_root="${ANNDATASET_ROOT:-/home/zwan/ANNdataset}"
annlib_dir="${ANNLIB_DIR:-/home/zwan/ANNlib}"
annlib_test_dir="${ANNLIB_TEST_DIR:-${annlib_dir}/test}"
artifact_root="${ARTIFACT_ROOT:-${annlib_dir}/artifacts/point_timerange_ann}"
