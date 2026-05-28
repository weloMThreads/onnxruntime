#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) Moore Threads. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/workspace/miniconda3/envs/onnx/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

CONFIG="${CONFIG:-Release}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_musa_wheel_only_py311}"
MUSA_HOME_DIR="${MUSA_HOME:-/usr/local/musa}"
JOBS="${JOBS:-8}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/ci_build/build.py" \
  --build_dir "${BUILD_DIR}" \
  --config "${CONFIG}" \
  --update \
  --build \
  --skip_tests \
  --skip_submodule_sync \
  --allow_running_as_root \
  --compile_no_warning_as_error \
  --parallel "${JOBS}" \
  --build_wheel \
  --use_musa \
  --musa_home "${MUSA_HOME_DIR}" \
  --targets onnxruntime_pybind11_state onnxruntime_providers_musa \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
