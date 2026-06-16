#!/usr/bin/env bash
set -euo pipefail

for lib_dir in "$HOME/.mujoco/mujoco210/bin" "/usr/lib/nvidia"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$lib_dir"
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/home/hochan/miniconda3/envs/dsd/bin/python}"
MODE="${MODE:-all}"
METHODS="${METHODS:-susd metra dads lsd diayn dads_poe}"
SEEDS="${SEEDS:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-final_models/ant}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-latest}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ant_maze_meta_policy}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-5}"
SKILL_STEPS="${SKILL_STEPS:-10}"
MAZE_HORIZON="${MAZE_HORIZON:-400}"
GOAL_X="${GOAL_X:-6.0}"
GOAL_Y="${GOAL_Y:-6.0}"
GOAL_RADIUS="${GOAL_RADIUS:-0.75}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
MIN_BUFFER_SIZE="${MIN_BUFFER_SIZE:-1000}"
GRADIENT_STEPS="${GRADIENT_STEPS:-50}"
TRAIN_FREQ="${TRAIN_FREQ:-1}"
DISCOUNT="${DISCOUNT:-0.99}"
TAU="${TAU:-0.005}"
TARGET_COEF="${TARGET_COEF:-1.0}"
HIDDEN_SIZES="${HIDDEN_SIZES:-512 512}"
INIT_ALPHA="${INIT_ALPHA:-1.0}"
DEVICE="${DEVICE:-auto}"
SKIP_MISSING="${SKIP_MISSING:-1}"
UNIT_SKILL="${UNIT_SKILL:-1}"
STOCHASTIC_LOW_POLICY="${STOCHASTIC_LOW_POLICY:-0}"

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE_ARG=()
else
  DEVICE_ARG=(--device "${DEVICE}")
fi

SKIP_ARG=(--skip-missing)
if [[ "${SKIP_MISSING}" == "0" ]]; then
  SKIP_ARG=(--no-skip-missing)
fi

UNIT_ARG=(--unit-skill)
if [[ "${UNIT_SKILL}" == "0" ]]; then
  UNIT_ARG=(--no-unit-skill)
fi

STOCH_ARG=()
if [[ "${STOCHASTIC_LOW_POLICY}" == "1" ]]; then
  STOCH_ARG=(--stochastic-low-policy)
fi

"${PYTHON_BIN}" src/evaluations/ant_maze_meta_policy.py \
  --mode "${MODE}" \
  --methods ${METHODS} \
  --seeds ${SEEDS} \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --checkpoint-epoch "${CHECKPOINT_EPOCH}" \
  --output-dir "${OUTPUT_DIR}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --eval-freq "${EVAL_FREQ}" \
  --n-eval-episodes "${N_EVAL_EPISODES}" \
  --skill-steps "${SKILL_STEPS}" \
  --maze-horizon "${MAZE_HORIZON}" \
  --goal-x "${GOAL_X}" \
  --goal-y "${GOAL_Y}" \
  --goal-radius "${GOAL_RADIUS}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --buffer-size "${BUFFER_SIZE}" \
  --min-buffer-size "${MIN_BUFFER_SIZE}" \
  --gradient-steps "${GRADIENT_STEPS}" \
  --train-freq "${TRAIN_FREQ}" \
  --discount "${DISCOUNT}" \
  --tau "${TAU}" \
  --target-coef "${TARGET_COEF}" \
  --hidden-sizes ${HIDDEN_SIZES} \
  --init-alpha "${INIT_ALPHA}" \
  "${SKIP_ARG[@]}" \
  "${UNIT_ARG[@]}" \
  "${STOCH_ARG[@]}" \
  "${DEVICE_ARG[@]}"
