#!/usr/bin/env bash
# Run SeRF on all datasets defined in datasets.sh.
# Requires sorted inputs from prepare_sorted_timerange_inputs.py.
# SeRF builds its index in memory; there is no persistent index file.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/../datasets.sh"

mkdir -p "${result_root}" "${log_root}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue

  sorted_dir="${sorted_root}/${tag}"
  mkdir -p "${log_root}/${tag}"

  echo "[${tag}] serf start"
  "${repo_root}/SeRF/build/benchmark/serf_annlib" \
    --data-path "${sorted_dir}/base_sorted.bin" \
    --query-path "${sorted_dir}/queries.bin" \
    --artifacts-dir "${sorted_dir}" \
    --output "${result_root}/${tag}_serf.tsv" \
    --data-size "${max_points}" \
    --k "${k}" \
    --index-k-list 16 \
    --ef-con-list 80 \
    --ef-max-list 300 \
    --ef-search-list 160 \
    --ranges "${ranges_csv}" \
    &> "${log_root}/${tag}/serf.log"
  echo "[${tag}] serf done"
done
