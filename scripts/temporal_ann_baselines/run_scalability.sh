#!/usr/bin/env bash
# Thread-scalability sweep for the range-filtered ANN baselines.
#
# For one (dataset, distribution) combo at a subsampled size, sweep the build
# thread count over SCAL_THREADS and record, per baseline:
#   * build_seconds  (all 5 baselines — index construction is parallel)
#   * qps            (only RangeFilteredANN + UNIFY — their query path is the
#                     only one that is actually multi-threaded; DSG / SeRF /
#                     iRangeGraph run queries with a strictly serial loop, so a
#                     "QPS vs threads" curve for them would be a flat, misleading
#                     line and is intentionally left as NA.)
#
# Output: ${result_root}/scalability/scalability.tsv
#   columns: baseline   threads   build_seconds   qps
# Then plot with:
#   ${python_bin} scripts/temporal_ann_baselines/plot_scalability.py
#
# Everything is overridable via env vars (defaults in the block below).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${script_dir}/config.sh"

# --------------------------- knobs -----------------------------------------
: "${SCAL_BASE:=deep1M_yandex}"          # which base dataset (see map below)
: "${SCAL_DISTRIBUTION:=uniform}"        # timestamp distribution for the combo
: "${SCAL_N:=100000}"                    # subsample size (points)
: "${SCAL_THREADS:=1 2 4 8 16 32}"       # thread counts to sweep
: "${SCAL_RANGE:=10000}"                 # single range width used for queries
: "${SCAL_RANGES_GEN:=100 1000 10000 100000}"  # widths to generate in artifacts
: "${K:=10}"
: "${NUM_QUERIES:=1000}"
: "${SEED:=42}"

# Base dataset map (spec|type|dist), same as run_synth_timestamps.sh.
declare -A BASE_SPEC=(
  [bigann1M]="${anndataset_root}/BIGANN/base.1B.u8bin:u8bin|uint8|L2"
  [deep1M_yandex]="${anndataset_root}/Yandex-DEEP/base.1B.fbin:fbin|float|L2"
  [openai1M]="${anndataset_root}/openai/openai_large_5m/base.fbin:fbin|float|angular"
  [cohere1M]="${anndataset_root}/cohere/cohere_large_10m/base.fbin:fbin|float|angular"
)
entry="${BASE_SPEC[$SCAL_BASE]:-}"
[[ -n "${entry}" ]] || { echo "unknown SCAL_BASE=${SCAL_BASE}"; exit 1; }
IFS='|' read -r spec vtype dist <<< "${entry}"

tag="${SCAL_BASE}__${SCAL_DISTRIBUTION}_N${SCAL_N}"
art_dir="${artifact_root}/scalability/${tag}"
sorted_dir="${cache_root}/scalability/sorted/${tag}"
work_root="${cache_root}/scalability/work/${tag}"
out_dir="${result_root}/scalability"
out_tsv="${out_dir}/scalability.tsv"
mkdir -p "${art_dir}" "${sorted_dir}" "${work_root}" "${out_dir}"

echo "[scal] base=${SCAL_BASE} dist=${SCAL_DISTRIBUTION} N=${SCAL_N} dim_type=${vtype} metric=${dist}"
echo "[scal] threads: ${SCAL_THREADS}   query range width: ${SCAL_RANGE}"
echo "[scal] output: ${out_tsv}"

# --------------------------- one-time inputs --------------------------------
if [[ ! -f "${art_dir}/manifest.txt" ]]; then
  echo "[scal] generating ${SCAL_N}-point ${SCAL_DISTRIBUTION} artifacts"
  "${python_bin}" "${script_dir}/synth_timestamps.py" \
    --dataset "${spec}" --out-dir "${art_dir}" \
    --max-points "${SCAL_N}" --num-queries "${NUM_QUERIES}" --query-k "${K}" \
    --ranges "${SCAL_RANGES_GEN}" --dist "${dist}" \
    --distribution "${SCAL_DISTRIBUTION}" --center-bias uniform --seed "${SEED}"
else
  echo "[scal] artifacts present — skip generation"
fi

if [[ ! -f "${sorted_dir}/base_sorted.bin" ]]; then
  echo "[scal] preparing sorted inputs"
  "${python_bin}" "${script_dir}/lib/prepare_sorted_timerange_inputs.py" \
    --dataset "${spec}" --artifacts-dir "${art_dir}" \
    --output-dir "${sorted_dir}" --max-points "${SCAL_N}"
else
  echo "[scal] sorted inputs present — skip"
fi

# --------------------------- helpers ----------------------------------------
# Extract a column value (by header name) from the first data row of a TSV.
col_val() { # <tsv> <colname>
  awk -F'\t' -v col="$2" 'NR==1{for(i=1;i<=NF;i++) if($i==col) c=i; next}
                          c && $c!="" {print $c; exit}' "$1"
}
wall() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.6f", b-a}'; }

# Fresh output file with header.
printf "baseline\tthreads\tbuild_seconds\tqps\n" > "${out_tsv}"
emit() { printf "%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" >> "${out_tsv}"; }

base_sorted="${sorted_dir}/base_sorted.bin"
queries="${sorted_dir}/queries.bin"

dsg_build="${repo_root}/Dynamic-Range-Filtering-ANNS/build/apps/build_static_index"
serf_bin="${repo_root}/SeRF/build/benchmark/serf_annlib"
irg_build="${repo_root}/iRangeGraph/build/tests/buildindex"

# --------------------------- sweep ------------------------------------------
for T in ${SCAL_THREADS}; do
  echo "========================  threads=${T}  ========================"
  wdir="${work_root}/T${T}"; mkdir -p "${wdir}"

  # ---- DSG (build only; query loop is serial) -- wall-clock around build ----
  if [[ -x "${dsg_build}" ]]; then
    echo "[T=${T}] dsg build"
    t0=$(wall)
    OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T} "${dsg_build}" \
      -dataset annlib -N "${SCAL_N}" \
      -dataset_path "${base_sorted}" -query_path "${queries}" \
      -index_path "${wdir}/dsg.index" \
      -query_num "${NUM_QUERIES}" -query_k "${K}" \
      -k 16 -ef_construction 80 -ef_max 300 -alpha 1.0 \
      &> "${wdir}/dsg_build.log" && \
      emit dsg "${T}" "$(elapsed "${t0}" "$(wall)")" NA || \
      { echo "  dsg FAILED (see ${wdir}/dsg_build.log)"; emit dsg "${T}" NA NA; }
  else
    echo "[T=${T}] dsg binary missing — skip"
  fi

  # ---- SeRF (build+serial query; take self-reported build_seconds) ----------
  if [[ -x "${serf_bin}" ]]; then
    echo "[T=${T}] serf build"
    OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T} "${serf_bin}" \
      --data-path "${base_sorted}" --query-path "${queries}" \
      --artifacts-dir "${sorted_dir}" --output "${wdir}/serf.tsv" \
      --data-size "${SCAL_N}" --k "${K}" \
      --index-k-list 16 --ef-con-list 80 --ef-max-list 300 \
      --ef-search-list 160 --ranges "${SCAL_RANGE}" \
      &> "${wdir}/serf.log" && \
      emit serf "${T}" "$(col_val "${wdir}/serf.tsv" build_seconds)" NA || \
      { echo "  serf FAILED (see ${wdir}/serf.log)"; emit serf "${T}" NA NA; }
  else
    echo "[T=${T}] serf binary missing — skip"
  fi

  # ---- iRangeGraph (build only; query loop is serial) -- --threads flag ------
  if [[ -x "${irg_build}" ]]; then
    echo "[T=${T}] irangegraph build"
    t0=$(wall)
    "${irg_build}" \
      --data_path "${base_sorted}" --index_file "${wdir}/irg.bin" \
      --M 16 --ef_construction 80 --threads "${T}" \
      &> "${wdir}/irg_build.log" && \
      emit irangegraph "${T}" "$(elapsed "${t0}" "$(wall)")" NA || \
      { echo "  irangegraph FAILED (see ${wdir}/irg_build.log)"; emit irangegraph "${T}" NA NA; }
  else
    echo "[T=${T}] irangegraph binary missing — skip"
  fi

  # ---- RangeFilteredANN (parlay: build + batch query both at PARLAY=T) -------
  echo "[T=${T}] rangefilteredann build+query"
  rm -rf "${wdir}/rfa_cache"
  OPENBLAS_NUM_THREADS=${T} "${python_bin}" "${script_dir}/lib/run_rangefilteredann.py" \
    --dataset "${spec}" --artifacts-dir "${art_dir}" --dist "${dist}" \
    --max-points "${SCAL_N}" --ranges "${SCAL_RANGE}" \
    --methods vamana_tree --beam-sizes 160 --final-beam-multiplies 1 \
    --alpha 1 --R 32 --L 80 --threads "${T}" \
    --index-cache-dir "${wdir}/rfa_cache" \
    --output "${wdir}/rfa.tsv" \
    &> "${wdir}/rfa.log" && \
    emit rangefilteredann "${T}" "$(col_val "${wdir}/rfa.tsv" build_seconds)" "$(col_val "${wdir}/rfa.tsv" qps)" || \
    { echo "  rfa FAILED (see ${wdir}/rfa.log)"; emit rangefilteredann "${T}" NA NA; }

  # ---- UNIFY (OMP build threads + --num-workers query threads, both = T) -----
  echo "[T=${T}] unify build+query"
  rm -rf "${wdir}/unify_cache"
  OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T} "${python_bin}" "${script_dir}/lib/run_unify.py" \
    --dataset "${spec}" --artifacts-dir "${art_dir}" --dist "${dist}" \
    --max-points "${SCAL_N}" --ranges "${SCAL_RANGE}" \
    --methods hsig_hybrid --ef-list 160 --al-list 16 \
    --num-workers "${T}" \
    --index-cache-dir "${wdir}/unify_cache" \
    --output "${wdir}/unify.tsv" \
    &> "${wdir}/unify.log" && \
    emit unify "${T}" "$(col_val "${wdir}/unify.tsv" build_seconds)" "$(col_val "${wdir}/unify.tsv" qps)" || \
    { echo "  unify FAILED (see ${wdir}/unify.log)"; emit unify "${T}" NA NA; }
done

echo "[scal] done -> ${out_tsv}"
echo "[scal] plot with: ${python_bin} ${script_dir}/plot_scalability.py --input ${out_tsv}"
