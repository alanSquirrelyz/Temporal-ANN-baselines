#!/usr/bin/env bash
# Generate artifacts and run all baselines for every dataset in datasets.sh.
# Calls: generate_annlib_timerange_artifacts.sh, prepare_sorted_timerange_inputs.py,
#        run_rangefilteredann.sh, run_unify.sh, run_dsg.sh, run_serf.sh, run_irangegraph.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${script_dir}/datasets.sh"

python_bin="${PYTHON_BIN:-python3}"

mkdir -p "${artifact_root}" "${sorted_root}" "${result_root}" "${log_root}"

# Step 1: generate ANNlib artifacts and sorted inputs for all datasets
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r tag dataset type dist max_points <<< "${entry}"
  [[ -n "${tag}" ]] || continue

  echo "[${tag}] generating artifacts"
  DATASET="${dataset}" \
  DATASET_TAG="${tag}" \
  TYPE="${type}" \
  DIST="${dist}" \
  MAX_POINTS="${max_points}" \
  NUM_QUERIES="${num_queries}" \
  K="${k}" \
  RANGES_STR="${ranges_str}" \
  ARTIFACT_ROOT="${artifact_root}" \
  ANNLIB_TEST_DIR="${annlib_dir}/test" \
  "${script_dir}/lib/generate_annlib_timerange_artifacts.sh"

  echo "[${tag}] preparing sorted inputs"
  "${python_bin}" "${script_dir}/lib/prepare_sorted_timerange_inputs.py" \
    --dataset "${dataset}" \
    --artifacts-dir "${artifact_root}/${tag}" \
    --output-dir "${sorted_root}/${tag}" \
    --max-points "${max_points}"
done

# Step 2: run each baseline
"${script_dir}/lib/run_rangefilteredann.sh"
"${script_dir}/lib/run_unify.sh"
"${script_dir}/lib/run_dsg.sh"
"${script_dir}/lib/run_serf.sh"
"${script_dir}/lib/run_irangegraph.sh"
