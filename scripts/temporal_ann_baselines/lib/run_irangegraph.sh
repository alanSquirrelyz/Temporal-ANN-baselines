#!/usr/bin/env bash
# Run iRangeGraph on all datasets defined in datasets.sh.
# Requires sorted inputs from prepare_sorted_timerange_inputs.py.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${script_dir}/../datasets.sh"

mkdir -p "${result_root}" "${log_root}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue

  sorted_dir="${sorted_root}/${tag}"
  index_dir="${cache_root}/irangegraph/${tag}"
  index_path="${index_dir}/irangegraph.bin"
  time_file="${index_dir}/build_seconds.txt"
  mkdir -p "${index_dir}" "${log_root}/${tag}"

  echo "[${tag}] irangegraph build start"
  /usr/bin/time -f "%e" -o "${time_file}" \
    "${repo_root}/iRangeGraph/build/tests/buildindex" \
      --data_path "${sorted_dir}/base_sorted.bin" \
      --index_file "${index_path}" \
      --M 16 \
      --ef_construction 80 \
      --threads "$(nproc)" \
      &> "${log_root}/${tag}/irangegraph_build.log"

  build_seconds="$(<"${time_file}")"
  index_bytes="$(stat -c%s "${index_path}")"

  echo "[${tag}] irangegraph query start"
  "${repo_root}/iRangeGraph/build/tests/search_annlib" \
    --data-path "${sorted_dir}/base_sorted.bin" \
    --query-path "${sorted_dir}/queries.bin" \
    --artifacts-dir "${sorted_dir}" \
    --index-file "${index_path}" \
    --output "${result_root}/${tag}_irangegraph.tsv" \
    --M 16 \
    --k "${k}" \
    --build-seconds "${build_seconds}" \
    --index-bytes "${index_bytes}" \
    --space-usage-source "index_file_bytes" \
    --ef-list 160 \
    --ranges "${ranges_csv}" \
    &> "${log_root}/${tag}/irangegraph_query.log"
  echo "[${tag}] irangegraph done"
done
