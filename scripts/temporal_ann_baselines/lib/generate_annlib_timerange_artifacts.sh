#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
annlib_test_dir="${ANNLIB_TEST_DIR:?ANNLIB_TEST_DIR must be set}"
artifact_root="${ARTIFACT_ROOT:?ARTIFACT_ROOT must be set}"

if [[ $# -gt 0 && "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Generate point-timestamp, query-range, and ground-truth artifacts with ANNlib.

Environment variables:
  ANNLIB_TEST_DIR   ANNlib test directory (e.g. /home/zwan/ANNlib/test)
  ARTIFACT_ROOT     Output root for generated artifacts
  DATASET           ANNlib dataset spec, e.g. ./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin
  DATASET_TAG       Dataset tag used under ARTIFACT_ROOT
  TYPE              uint8 | int8 | float
  DIST              L2 | angular
  MAX_POINTS        Number of base points to use
  NUM_QUERIES       Number of query ids to generate
  K                 Ground-truth top-k
  RANGES_STR        Space-separated range widths, e.g. "100 1000 10000"
  FORCE_REGEN       1 to regenerate existing files

Example:
  DATASET=./ANNdataset/Yandex-DEEP/base.1B.fbin:fbin \
  DATASET_TAG=deep1M_yandex TYPE=float DIST=L2 MAX_POINTS=1000000 \
  NUM_QUERIES=1000 RANGES_STR="100 1000 10000 100000" \
  ./scripts/temporal_ann_baselines/generate_annlib_timerange_artifacts.sh
EOF
  exit 0
fi

mkdir -p "${artifact_root}"

(
  cd "${annlib_test_dir}"
  ARTIFACT_ROOT="${artifact_root}" \
  DATASET="${DATASET:?missing DATASET}" \
  DATASET_TAG="${DATASET_TAG:?missing DATASET_TAG}" \
  TYPE="${TYPE:?missing TYPE}" \
  DIST="${DIST:?missing DIST}" \
  MAX_POINTS="${MAX_POINTS:?missing MAX_POINTS}" \
  NUM_QUERIES="${NUM_QUERIES:-1000}" \
  K="${K:-10}" \
  RANGES_STR="${RANGES_STR:-100 1000 10000 100000 1000000}" \
  FORCE_REGEN="${FORCE_REGEN:-0}" \
  ./prepare_timerange_dataset.sh
)
