#!/usr/bin/env bash
# Run UNIFY (hsig_hybrid, hsig_pre, hsig_post) on all datasets defined in datasets.sh.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/../datasets.sh"


mkdir -p "${result_root}" "${log_root}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue

  cache_dir="${cache_root}/unify/${tag}"
  rm -rf "${cache_dir}"
  mkdir -p "${log_root}/${tag}"

  echo "[${tag}] unify start"
  "${python_bin}" "${script_dir}/run_unify.py" \
    --dataset "${dataset}" \
    --artifacts-dir "${artifact_root}/${tag}" \
    --dist "${dist}" \
    --max-points "${max_points}" \
    --ranges ${ranges_str} \
    --methods hsig_hybrid hsig_pre hsig_post \
    --ef-list 160 \
    --al-list 16 \
    --index-cache-dir "${cache_dir}" \
    --output "${result_root}/${tag}_unify.tsv" \
    &> "${log_root}/${tag}/unify.log"
  echo "[${tag}] unify done"
done
