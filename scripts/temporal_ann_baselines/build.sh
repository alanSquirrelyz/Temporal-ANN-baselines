#!/usr/bin/env bash
# Build all baselines: Python bindings (RangeFilteredANN, UNIFY) and C++ binaries (DSG, SeRF, iRangeGraph).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
patch_dir="${repo_root}/scripts/temporal_ann_baselines/patches"
python_bin="${PYTHON_BIN:-python3}"
rangefiltered_repo="${RANGEFILTERED_REPO:-${repo_root}/RangeFilteredANN}"
unify_repo="${UNIFY_REPO:-${repo_root}/UNIFY}"

# --- Apply integration patches to submodules ---

apply_patch() {
  local repo_dir="$1"
  local patch="${patch_dir}/$(basename "${repo_dir}").patch"
  [[ -s "${patch}" ]] || return 0
  echo "Applying patch to $(basename "${repo_dir}")..."
  git -C "${repo_dir}" apply --check "${patch}" 2>/dev/null && \
    git -C "${repo_dir}" apply "${patch}" || \
    echo "  (patch already applied or not needed)"
}

for sub in RangeFilteredANN UNIFY Dynamic-Range-Filtering-ANNS SeRF iRangeGraph; do
  apply_patch "${repo_root}/${sub}"
done

pybind11_cmakedir="$("${python_bin}" -m pybind11 --cmakedir)"
python_ext_suffix="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")')"
site_packages_dir="$("${python_bin}" -c 'import site; print(site.getsitepackages()[0])')"

# --- Python bindings ---

echo "Building RangeFilteredANN Python bindings..."
build_rangefilteredann() {
  (
    cd "${rangefiltered_repo}"
    export pybind11_DIR="${pybind11_cmakedir}"
    export CMAKE_PREFIX_PATH="${pybind11_cmakedir}:${CMAKE_PREFIX_PATH:-}"
    "${python_bin}" -m pip install -e .
  )
}
if ! build_rangefilteredann; then
  built_module="$(find "${rangefiltered_repo}/build" -type f -name 'window_ann*.so' | sort | tail -n 1)"
  if [[ -z "${built_module}" ]]; then
    echo "RangeFilteredANN build failed and no built window_ann module found." >&2
    exit 1
  fi
  install -D "${built_module}" "${site_packages_dir}/window_ann${python_ext_suffix}"
  "${python_bin}" -c 'import window_ann'
fi

echo "Building UNIFY Python bindings..."
(
  cd "${unify_repo}/python_bindings"
  export pybind11_DIR="${pybind11_cmakedir}"
  export CMAKE_PREFIX_PATH="${pybind11_cmakedir}:${CMAKE_PREFIX_PATH:-}"
  "${python_bin}" -m pip install --no-build-isolation .
)

# --- C++ binaries ---

build_cpp() {
  local repo_dir="$1"
  echo "Building ${repo_dir##*/}..."
  mkdir -p "${repo_dir}/build"
  (
    cd "${repo_dir}/build"
    cmake ..
    make -j"$(nproc)"
  )
}

build_cpp "${repo_root}/Dynamic-Range-Filtering-ANNS"
build_cpp "${repo_root}/SeRF"
build_cpp "${repo_root}/iRangeGraph"

echo "All baselines built."
