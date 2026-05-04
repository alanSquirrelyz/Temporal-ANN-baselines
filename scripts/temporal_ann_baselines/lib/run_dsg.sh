#!/usr/bin/env bash
# Run DSG (Dynamic Segment Graph) on all datasets defined in datasets.sh.
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
  index_dir="${cache_root}/dsg/${tag}"
  index_path="${index_dir}/dsg.index"
  time_file="${index_dir}/build_seconds.txt"
  mkdir -p "${index_dir}" "${log_root}/${tag}"

  echo "[${tag}] dsg build start"
  /usr/bin/time -f "%e" -o "${time_file}" \
    "${repo_root}/Dynamic-Range-Filtering-ANNS/build/apps/build_static_index" \
      -dataset annlib \
      -N "${max_points}" \
      -dataset_path "${sorted_dir}/base_sorted.bin" \
      -query_path "${sorted_dir}/queries.bin" \
      -index_path "${index_path}" \
      -query_num "${num_queries}" \
      -query_k "${k}" \
      -k 16 \
      -ef_construction 80 \
      -ef_max 300 \
      -alpha 1.0 \
      &> "${log_root}/${tag}/dsg_build.log"

  build_seconds="$(<"${time_file}")"
  index_bytes="$(stat -c%s "${index_path}")"

  echo "[${tag}] dsg query start"
  "${repo_root}/Dynamic-Range-Filtering-ANNS/build/apps/query_annlib_index" \
    --data-path "${sorted_dir}/base_sorted.bin" \
    --query-path "${sorted_dir}/queries.bin" \
    --index-path "${index_path}" \
    --artifacts-dir "${sorted_dir}" \
    --output "${result_root}/${tag}_dsg.tsv" \
    --data-size "${max_points}" \
    --k "${k}" \
    --M 16 \
    --build-seconds "${build_seconds}" \
    --index-bytes "${index_bytes}" \
    --space-usage-source "index_file_bytes" \
    --search-ef-list 160 \
    --ranges "${ranges_csv}" \
    &> "${log_root}/${tag}/dsg_query.log"
  echo "[${tag}] dsg done"
done
