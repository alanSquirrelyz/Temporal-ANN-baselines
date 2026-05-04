#!/usr/bin/env bash
# Scalability tests: vary thread count for RangeFilteredANN and ANNlib on bigann1M.
# Results written to:
#   ${result_root}/scalability/rangefilteredann/bigann1M_threads{N}.tsv
#   ${annlib_dir}/test/results/point_timerange_parallel_merge_scalability/annlib_scalability_summary.tsv
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${script_dir}/datasets.sh"


annlib_dir="${ANNLIB_DIR:?set ANNLIB_DIR to your ANNlib repo root}"
thread_list="${THREAD_LIST:-1 2 4 8 16 32 48 96}"

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

# ---------------------------------------------------------------------------
# ANNlib scalability (test_timerange_parallel_merge)
# ---------------------------------------------------------------------------
annlib_binary="${annlib_dir}/test/test_timerange_parallel_merge"
range="100000"
annlib_cache_dir="${artifacts_dir}/cached_graphs/test_timerange_parallel_merge/tree_graph-simple_type-uint8_dist-L2_max-1000000_step-1000_m-32_efc-80_alpha-1_b-2_proportion-0p3"
annlib_result_root="${annlib_dir}/test/results/point_timerange_parallel_merge_scalability"
summary_tsv="${annlib_result_root}/annlib_scalability_summary.tsv"
mkdir -p "${annlib_result_root}"

echo "threads	build_s	qps	recall" > "${summary_tsv}"

echo "=== ANNlib scalability ==="
for threads in ${thread_list}; do
  log_dir="${annlib_result_root}/thread${threads}/${tag}/simple_step1000_proportion0p3"
  log_file="${log_dir}/simple-range${range}-step1000-proportion0p3.log"
  mkdir -p "${log_dir}"

  echo "  threads=${threads}"
  rm -rf "${annlib_cache_dir}"

  cmd="./test/test_timerange_parallel_merge \
    -step 1000 -max ${max_points} -type uint8 -dist L2 \
    -in ${dataset%%:*}:${dataset##*:} \
    -ts ${artifacts_dir}/point_timestamps.bin \
    -query-ids ${artifacts_dir}/query_ids.bin \
    -query-range ${artifacts_dir}/range${range}.query_ranges.bin \
    -gt ${artifacts_dir}/range${range}.gt.ibin \
    -nq ${num_queries} -k ${k} -ef 160 -threads ${threads} \
    -m 32 -efc 80 -alpha 1 -b 2 -proportion 0.3 -graph simple"

  echo "${cmd}" > "${log_file}"
  (cd "${annlib_dir}" && \
    PARLAY_NUM_THREADS=${threads} OMP_NUM_THREADS=${threads} \
    eval "${cmd}" 2>&1 | tee -a "${log_file}")

  build_s=$(grep -oP "Parallel merge (?:tree build|cache load) time:\s+\K[\d.]+" "${log_file}" | \
            awk '{s+=$1} END {print s+0}')
  kqps=$(grep -oP "Find neighbors.*?:\s+[\d.]+\s+s,\s+\K[\d.e+]+" "${log_file}" | tail -1)
  qps_val=$(python3 -c "print(float('${kqps:-0}') * 1000)" 2>/dev/null || echo "0")
  recall=$(grep -oP "query recall@\d+:\s+\K[\d.]+" "${log_file}" | awk '{s+=$1;n++} END {if(n>0) print s/n; else print 0}')

  echo "  build=${build_s}s  qps=${qps_val}  recall=${recall}"
  echo "${threads}	${build_s}	${qps_val}	${recall}" >> "${summary_tsv}"
done

echo "ANNlib scalability done."
echo "Summary: ${summary_tsv}"
column -t -s $'\t' "${summary_tsv}"
