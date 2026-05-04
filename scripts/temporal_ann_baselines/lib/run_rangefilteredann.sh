#!/usr/bin/env bash
# Run RangeFilteredANN (vamana_tree) on all datasets defined in datasets.sh.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/../datasets.sh"


mkdir -p "${result_root}" "${log_root}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue

  cache_dir="${cache_root}/rangefilteredann/${tag}"
  rm -rf "${cache_dir}"
  mkdir -p "${log_root}/${tag}"

  echo "[${tag}] rangefilteredann start"
  "${python_bin}" "${script_dir}/run_rangefilteredann.py" \
    --dataset "${dataset}" \
    --artifacts-dir "${artifact_root}/${tag}" \
    --dist "${dist}" \
    --max-points "${max_points}" \
    --ranges ${ranges_str} \
    --methods vamana_tree \
    --beam-sizes 160 \
    --final-beam-multiplies 1 \
    --alpha 1 --R 32 --L 80 \
    --index-cache-dir "${cache_dir}" \
    --output "${result_root}/${tag}_rangefilteredann.tsv" \
    &> "${log_root}/${tag}/rangefilteredann.log"
  echo "[${tag}] rangefilteredann done"
done
