#!/usr/bin/env bash
set -euo pipefail

for lib_dir in "$HOME/.mujoco/mujoco210/bin" "/usr/lib/nvidia"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$lib_dir"
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/home/hochan/miniconda3/envs/dsd/bin/python}"
METHODS="${METHODS:-susd metra}"
SEEDS="${SEEDS:-0}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-16000000}"
MAX_PATH_LENGTH="${MAX_PATH_LENGTH:-200}"
FINAL_ROOT="${FINAL_ROOT:-final_models/ant}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-0}"
SKIP_COPY="${SKIP_COPY:-0}"
DRY_RUN="${DRY_RUN:-0}"

export SUSD_N_EPOCHS_PER_PT_SAVE="${SUSD_N_EPOCHS_PER_PT_SAVE:-1000}"
export SUSD_N_EPOCHS_PER_SAVE="${SUSD_N_EPOCHS_PER_SAVE:-1000}"
export SUSD_N_EPOCHS_PER_PKL_UPDATE="${SUSD_N_EPOCHS_PER_PKL_UPDATE:-1000}"
export SUSD_N_EPOCHS_PER_EVAL="${SUSD_N_EPOCHS_PER_EVAL:-1000}"
export SUSD_N_EPOCHS_PER_LOG="${SUSD_N_EPOCHS_PER_LOG:-100}"
export SUSD_EVAL_RECORD_VIDEO="${SUSD_EVAL_RECORD_VIDEO:-0}"
export SUSD_USE_WANDB="${SUSD_USE_WANDB:-0}"
export SUSD_SHOW_PROGRESS="${SUSD_SHOW_PROGRESS:-1}"
export SUSD_PROGRESS_PERIOD="${SUSD_PROGRESS_PERIOD:-1}"
export SUSD_N_PARALLEL="${SUSD_N_PARALLEL:-32}"
export SUSD_TRAJ_BATCH_SIZE="${SUSD_TRAJ_BATCH_SIZE:-${SUSD_N_PARALLEL}}"

steps_per_epoch=$((SUSD_TRAJ_BATCH_SIZE * MAX_PATH_LENGTH))
if [[ "${steps_per_epoch}" -le 0 ]]; then
  echo "[error] steps_per_epoch must be positive. Got traj_batch_size=${SUSD_TRAJ_BATCH_SIZE}, max_path_length=${MAX_PATH_LENGTH}" >&2
  exit 2
fi
computed_epochs=$(((TOTAL_ENV_STEPS + steps_per_epoch - 1) / steps_per_epoch))
export SUSD_N_EPOCHS="${SUSD_N_EPOCHS:-${computed_epochs}}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-${SUSD_N_EPOCHS}}"
method_count=0
for _method in $METHODS; do
  method_count=$((method_count + 1))
done
seed_count=0
for _seed in $SEEDS; do
  seed_count=$((seed_count + 1))
done
total_jobs=$((method_count * seed_count))
job_idx=0
train_suite_start=$(date +%s)

if [[ "${SUSD_TRAJ_BATCH_SIZE}" -lt "${SUSD_N_PARALLEL}" ]]; then
  echo "[warn] SUSD_TRAJ_BATCH_SIZE=${SUSD_TRAJ_BATCH_SIZE} < SUSD_N_PARALLEL=${SUSD_N_PARALLEL}; some workers will be idle." >&2
fi

echo "[suite] methods=${METHODS} seeds=${SEEDS} jobs=${total_jobs} total_env_steps=${TOTAL_ENV_STEPS} steps_per_epoch=${steps_per_epoch} epochs=${SUSD_N_EPOCHS} checkpoint_epoch=${CHECKPOINT_EPOCH} n_parallel=${SUSD_N_PARALLEL} traj_batch_size=${SUSD_TRAJ_BATCH_SIZE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

copy_checkpoint() {
  local method="$1"
  local seed="$2"
  local run_group="$3"
  local dst_dir="${FINAL_ROOT}/${method^^}"

  local run_dir
  run_dir="$(ls -td "exp/${run_group}"/sd"$(printf '%03d' "${seed}")"*"_ant_metra" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${run_dir}" ]]; then
    echo "[error] Could not find run dir for ${method} seed=${seed} under exp/${run_group}" >&2
    return 1
  fi

  local option_src="${run_dir}/option_policy${CHECKPOINT_EPOCH}.pt"
  local encoder_src="${run_dir}/traj_encoder${CHECKPOINT_EPOCH}.pt"
  if [[ ! -f "${option_src}" || ! -f "${encoder_src}" ]]; then
    option_src="$(ls -v "${run_dir}"/option_policy*.pt 2>/dev/null | tail -n 1 || true)"
    encoder_src="$(ls -v "${run_dir}"/traj_encoder*.pt 2>/dev/null | tail -n 1 || true)"
  fi
  if [[ -z "${option_src}" || -z "${encoder_src}" ]]; then
    echo "[error] No option/traj_encoder checkpoint found in ${run_dir}" >&2
    return 1
  fi

  mkdir -p "${dst_dir}" "${dst_dir}/seed_${seed}"
  cp "${option_src}" "${dst_dir}/seed_${seed}/option_policy${CHECKPOINT_EPOCH}.pt"
  cp "${encoder_src}" "${dst_dir}/seed_${seed}/traj_encoder${CHECKPOINT_EPOCH}.pt"
  cp "${option_src}" "${dst_dir}/option_policy${CHECKPOINT_EPOCH}.pt"
  cp "${encoder_src}" "${dst_dir}/traj_encoder${CHECKPOINT_EPOCH}.pt"
  echo "[saved] ${method} seed=${seed}"
  echo "        ${dst_dir}/option_policy${CHECKPOINT_EPOCH}.pt"
  echo "        ${dst_dir}/traj_encoder${CHECKPOINT_EPOCH}.pt"
}

for method in ${METHODS}; do
  case "${method}" in
    susd)
      run_group="${SUSD_RUN_GROUP_BASE:-SUSD_ANT_5}"
      ;;
    metra)
      run_group="${METRA_RUN_GROUP_BASE:-METRA_ANT}"
      ;;
    csd|lsd|diayn|dusdi)
      echo "[unsupported] ${method}: this checkout does not include an Ant trainer for ${method}." >&2
      echo "              Add its trainer or provide final_models/ant/${method^^}/option_policy${CHECKPOINT_EPOCH}.pt directly." >&2
      exit 2
      ;;
    *)
      echo "[unsupported] Unknown method: ${method}" >&2
      exit 2
      ;;
  esac

  for seed in ${SEEDS}; do
    job_idx=$((job_idx + 1))
    job_start=$(date +%s)
    suite_elapsed=$((job_start - train_suite_start))
    echo "[train] job=${job_idx}/${total_jobs} method=${method} seed=${seed} epochs=${SUSD_N_EPOCHS} suite_elapsed=${suite_elapsed}s"
    SUSD_TRAIN_METHOD="${method}" \
    SUSD_RUN_GROUP="${run_group}" \
    SUSD_SEED="${seed}" \
    "${PYTHON_BIN}" src/pretrain.py

    if [[ "${SKIP_COPY}" == "1" ]]; then
      echo "[skip-copy] method=${method} seed=${seed}; smoke run only"
    else
      copy_checkpoint "${method}" "${seed}" "${run_group}"
    fi
    job_end=$(date +%s)
    job_elapsed=$((job_end - job_start))
    echo "[train-done] job=${job_idx}/${total_jobs} method=${method} seed=${seed} elapsed=${job_elapsed}s"
  done
done

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  METHODS="${METHODS}" \
  CHECKPOINT_ROOT="${FINAL_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/run_ant_heading_counterfactual.sh
fi
