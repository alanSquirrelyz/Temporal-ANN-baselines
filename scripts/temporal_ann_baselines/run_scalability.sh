#!/usr/bin/env bash
# Scalability tests: vary thread count for RangeFilteredANN and ANNlib on bigann1M.
# Results written to:
#   ${result_root}/scalability/rangefilteredann/bigann1M_threads{N}.tsv
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${script_dir}/datasets.sh"


thread_list="${THREAD_LIST:-1 2 4 8 16 28 56 112 224}"

# bigann1M dataset spec (first entry in DATASETS)
tag="bigann1M"
IFS='|' read -r _ dataset _ dist max_points <<< "${DATASETS[0]}"

artifacts_dir="${artifact_root}/${tag}"

# ---------------------------------------------------------------------------
# RangeFilteredANN scalability
# ---------------------------------------------------------------------------
rf_result_root="${result_root}/scalability/rangefilteredann"
mkdir -p "${rf_result_root}"

echo "=== RangeFilteredANN scalability ==="
for threads in ${thread_list}; do
  cache_dir="${cache_root}/rangefilteredann/scalability/threads${threads}"
  rm -rf "${cache_dir}"

  echo "  threads=${threads}"
  PARLAY_NUM_THREADS="${threads}" \
  OMP_NUM_THREADS="${threads}" \
  "${python_bin}" "${script_dir}/lib/run_rangefilteredann.py" \
    --dataset "${dataset}" \
    --artifacts-dir "${artifacts_dir}" \
    --dist "${dist}" \
    --max-points "${max_points}" \
    --ranges ${ranges_str} \
    --methods vamana_tree \
    --beam-sizes 160 \
    --final-beam-multiplies 1 \
    --alpha 1 --R 32 --L 80 \
    --threads "${threads}" \
    --index-cache-dir "${cache_dir}" \
    --output "${rf_result_root}/${tag}_threads${threads}.tsv"
done
echo "RangeFilteredANN scalability done."