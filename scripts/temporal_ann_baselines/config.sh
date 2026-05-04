# config.sh — Paths to external data and outputs. Edit this file before running.
#
# anndataset_root  Where your raw ANN dataset files live (outside the repo).
# artifact_root    Where ANNlib artifacts are stored (can be on a separate disk).
# result_root      Where benchmark result TSVs are written.

anndataset_root="${ANNDATASET_ROOT:-/home/zwan/ANNdataset}"
artifact_root="${ARTIFACT_ROOT:-/home/zwan/artifacts/point_timerange_ann}"
result_root="${RESULT_ROOT:-${repo_root}/results/temporal_baselines}"
