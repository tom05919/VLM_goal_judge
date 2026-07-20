#!/usr/bin/env bash
# Shared helpers for install_*_env.sh. Source this file; do not run it.

_go2_resolve_conda_base() {
  if [[ -n "${GO2_CONDA_BASE:-}" ]]; then
    printf '%s\n' "${GO2_CONDA_BASE}"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local base
    base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${base}"
      return 0
    fi
  fi
  if [[ -n "${MAMBA_ROOT_PREFIX:-}" && -f "${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh" ]]; then
    printf '%s\n' "${MAMBA_ROOT_PREFIX}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_path candidate
    conda_path="$(readlink -f "$(command -v conda)")"
    candidate="$(cd "$(dirname "${conda_path}")/.." && pwd)"
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    conda info --base
    return 0
  fi
  echo "Could not find conda. Set GO2_CONDA_BASE to your miniforge/conda root." >&2
  return 1
}

_go2_clear_inherited_env() {
  # A system ROS or previously active conda env must not contaminate the
  # RoboStack transaction or its Python package installer.
  unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
  unset LD_LIBRARY_PATH ROS_DISTRO ROS_PACKAGE_PATH ROS_PYTHON_VERSION
  unset ROS_ROOT ROS_VERSION
}

_go2_init_conda() {
  local base
  base="$(_go2_resolve_conda_base)" || return 1
  if [[ ! -f "${base}/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found under GO2_CONDA_BASE=${base}" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "${base}/etc/profile.d/conda.sh"
  export GO2_CONDA_BASE="${base}"
  if command -v mamba >/dev/null 2>&1; then
    GO2_MAMBA=(mamba)
  else
    GO2_MAMBA=(conda)
  fi
  echo "[envs] conda base: ${GO2_CONDA_BASE}"
}

_go2_envs_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

_go2_stack_root() {
  cd "$(_go2_envs_dir)/.." && pwd
}

_go2_workspace_root() {
  cd "$(_go2_stack_root)/.." && pwd
}

_go2_env_exists() {
  local name="$1"
  "${GO2_MAMBA[@]}" env list 2>/dev/null \
    | awk -v expected="${name}" '$1 == expected { found = 1 } END { exit !found }'
}

_go2_require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${description}: ${path}" >&2
    echo "Follow envs/VERSIONS.md to clone pinned source repositories." >&2
    return 1
  fi
}
