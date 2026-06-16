#!/usr/bin/env bash
set -euo pipefail

for lib_dir in "$HOME/.mujoco/mujoco210/bin" "/usr/lib/nvidia"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$lib_dir"
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/home/hochan/miniconda3/envs/dsd/bin/python}"
ENVS="${ENVS:-ant half_cheetah kitchen}"
METHODS="${METHODS:-susd metra dads lsd diayn dads_poe}"
SEEDS="${SEEDS:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-final_models}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-latest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/paper_skill_figures}"
STAGES="${STAGES:-traces coverage downstream plots}"
NUM_SKILLS="${NUM_SKILLS:-16}"
TRACE_HORIZON="${TRACE_HORIZON:-200}"
COVERAGE_STEPS="${COVERAGE_STEPS:-100000}"
DOWNSTREAM_STEPS="${DOWNSTREAM_STEPS:-20000}"
SKILL_PERIOD="${SKILL_PERIOD:-200}"
DOWNSTREAM_HORIZON="${DOWNSTREAM_HORIZON:-200}"
COVERAGE_BIN_SIZE="${COVERAGE_BIN_SIZE:-0.25}"
KITCHEN_GOAL="${KITCHEN_GOAL:-kettle}"
DEVICE="${DEVICE:-auto}"
SKIP_MISSING="${SKIP_MISSING:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE_ARG=()
else
  DEVICE_ARG=(--device "${DEVICE}")
fi

SKIP_ARG=(--skip-missing)
if [[ "${SKIP_MISSING}" == "0" ]]; then
  SKIP_ARG=(--no-skip-missing)
fi

DET_ARG=(--deterministic)
if [[ "${DETERMINISTIC}" == "0" ]]; then
  DET_ARG=(--no-deterministic)
fi

"${PYTHON_BIN}" src/evaluations/paper_skill_figures.py \
  --envs ${ENVS} \
  --methods ${METHODS} \
  --seeds ${SEEDS} \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --checkpoint-epoch "${CHECKPOINT_EPOCH}" \
  --output-root "${OUTPUT_ROOT}" \
  --stages ${STAGES} \
  --num-skills "${NUM_SKILLS}" \
  --trace-horizon "${TRACE_HORIZON}" \
  --coverage-steps "${COVERAGE_STEPS}" \
  --downstream-steps "${DOWNSTREAM_STEPS}" \
  --skill-period "${SKILL_PERIOD}" \
  --downstream-horizon "${DOWNSTREAM_HORIZON}" \
  --coverage-bin-size "${COVERAGE_BIN_SIZE}" \
  --kitchen-goal "${KITCHEN_GOAL}" \
  "${SKIP_ARG[@]}" \
  "${DET_ARG[@]}" \
  "${DEVICE_ARG[@]}"
