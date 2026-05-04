# config.sh — Paths to external data and outputs. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# artifact_root    Where ANNlib artifacts are stored (can be on a separate disk).
# result_root      Where benchmark result TSVs are written.

anndataset_root="${ANNDATASET_ROOT:-/home/zwan/ANNlib/test/ANNdataset}"
artifact_root="${ARTIFACT_ROOT:-/data/zwam018/TimeStampANN/point_timerange_ann}"
result_root="${RESULT_ROOT:-${repo_root}/results}"
