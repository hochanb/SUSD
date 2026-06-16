#!/usr/bin/env bash
set -euo pipefail

for lib_dir in "$HOME/.mujoco/mujoco210/bin" "/usr/lib/nvidia"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$lib_dir"
  fi
done

METHODS="${METHODS:-susd metra csd lsd diayn}"
SEEDS="${SEEDS:-0}"
NUM_SKILLS="${NUM_SKILLS:-16}"
HORIZON="${HORIZON:-100}"
NUM_EVAL_ROLLOUTS="${NUM_EVAL_ROLLOUTS:-5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-final_models/ant}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ant_heading_counterfactual}"
NORMALIZE_OBS="${NORMALIZE_OBS:-preset}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE_ARG=()
else
  DEVICE_ARG=(--device "${DEVICE}")
fi

"${PYTHON_BIN}" src/evaluations/ant_heading_counterfactual.py \
  --methods ${METHODS} \
  --seeds ${SEEDS} \
  --num-skills "${NUM_SKILLS}" \
  --horizon "${HORIZON}" \
  --num-eval-rollouts "${NUM_EVAL_ROLLOUTS}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --normalize-obs "${NORMALIZE_OBS}" \
  "${DEVICE_ARG[@]}"
