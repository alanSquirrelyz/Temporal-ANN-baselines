#!/usr/bin/env bash
# Batch-run synth_timestamps.py over all configured datasets and distributions.
#
# Reads paths from config.sh. Override per-dataset / per-distribution behavior
# via the env vars at the top, then run:
#
#   bash scripts/temporal_ann_baselines/run_synth_timestamps.sh
#
# Outputs go to ${artifact_root}/<dataset_tag>__<distribution>/.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
# shellcheck source=/dev/null
source "$script_dir/config.sh"

: "${NUM_QUERIES:=1000}"
: "${QUERY_K:=10}"
: "${RANGES_STR:=100 1000 10000 100000}"
: "${CENTER_BIAS:=uniform}"
: "${SEED:=42}"
: "${DISTRIBUTIONS:=uniform pareto_recent lognormal_recent topic_drift topic_burst topic_seasonal}"
: "${SKIP_EXISTING:=1}"

DATASETS=(
  "bigann1M|${anndataset_root}/BIGANN/base.1B.u8bin:u8bin|uint8|L2|1000000"
  "deep1M_yandex|${anndataset_root}/Yandex-DEEP/base.1B.fbin:fbin|float|L2|1000000"
  "openai1M|${anndataset_root}/openai/openai_large_5m/base.fbin:fbin|float|angular|1000000"
  "cohere1M|${anndataset_root}/cohere/cohere_large_10m/base.fbin:fbin|float|angular|1000000"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag spec _type dist max_points <<<"$entry"
  for distribution in $DISTRIBUTIONS; do
    out_dir="${artifact_root}/${tag}__${distribution}"
    echo "=========================================================="
    echo "dataset=${tag}  distribution=${distribution}  dist=${dist}"
    echo "out=${out_dir}"
    echo "=========================================================="
    if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/manifest.txt" ]]; then
      echo "[skip] manifest.txt present — set SKIP_EXISTING=0 to overwrite"
      continue
    fi
    "$python_bin" "$script_dir/synth_timestamps.py" \
      --dataset "$spec" \
      --out-dir "$out_dir" \
      --max-points "$max_points" \
      --num-queries "$NUM_QUERIES" \
      --query-k "$QUERY_K" \
      --ranges "$RANGES_STR" \
      --dist "$dist" \
      --distribution "$distribution" \
      --center-bias "$CENTER_BIAS" \
      --seed "$SEED"
  done
done
