#!/usr/bin/env bash
set -euo pipefail

for lib_dir in "$HOME/.mujoco/mujoco210/bin" "/usr/lib/nvidia"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$lib_dir"
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/home/hochan/miniconda3/envs/dsd/bin/python}"
ENVS="${ENVS:-ant half_cheetah kitchen}"
METHODS="${METHODS:-susd metra dads dads_poe}"
SEEDS="${SEEDS:-0}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-16000000}"
FINAL_ROOT="${FINAL_ROOT:-final_models}"
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

requested_epochs="${SUSD_N_EPOCHS:-}"
requested_checkpoint_epoch="${CHECKPOINT_EPOCH:-}"

env_count=0
for _env in ${ENVS}; do
  env_count=$((env_count + 1))
done
method_count=0
for _method in ${METHODS}; do
  method_count=$((method_count + 1))
done
seed_count=0
for _seed in ${SEEDS}; do
  seed_count=$((seed_count + 1))
done
total_jobs=$((env_count * method_count * seed_count))
job_idx=0
train_suite_start=$(date +%s)

if [[ "${SUSD_TRAJ_BATCH_SIZE}" -lt "${SUSD_N_PARALLEL}" ]]; then
  echo "[warn] SUSD_TRAJ_BATCH_SIZE=${SUSD_TRAJ_BATCH_SIZE} < SUSD_N_PARALLEL=${SUSD_N_PARALLEL}; some workers will be idle." >&2
fi

canonical_env() {
  local env_name="$1"
  case "${env_name}" in
    ant)
      PRETRAIN_ENV="ant"
      FINAL_ENV="ant"
      ENV_TAG="ANT"
      DEFAULT_MAX_PATH_LENGTH="200"
      ;;
    half_cheetah|half-cheetah|halfcheetah)
      PRETRAIN_ENV="half_cheetah"
      FINAL_ENV="half_cheetah"
      ENV_TAG="HALF_CHEETAH"
      DEFAULT_MAX_PATH_LENGTH="200"
      ;;
    kitchen|kitchen_franka|franka_kitchen)
      PRETRAIN_ENV="kitchen_franka"
      FINAL_ENV="kitchen"
      ENV_TAG="KITCHEN"
      DEFAULT_MAX_PATH_LENGTH="50"
      ;;
    *)
      echo "[unsupported] Unknown env: ${env_name}. Supported: ant half_cheetah kitchen" >&2
      exit 2
      ;;
  esac
}

resolve_run_group() {
  local method="$1"
  local final_env="$2"
  local env_tag="$3"
  local default_group
  case "${method}" in
    susd)
      case "${final_env}" in
        ant) default_group="SUSD_ANT_5" ;;
        half_cheetah) default_group="SUSD_HALF_CHEETAH" ;;
        kitchen) default_group="SUSD_KITCHEN" ;;
        *) default_group="SUSD_${env_tag}" ;;
      esac
      ;;
    metra) default_group="METRA_${env_tag}" ;;
    dads) default_group="DADS_${env_tag}" ;;
    dads_poe) default_group="DADS_POE_${env_tag}" ;;
    csd|lsd|diayn|dusdi)
      echo "[unsupported] ${method}: this checkout does not include a pretrain.py trainer for ${method}." >&2
      echo "              Provide final_models/${final_env}/${method^^}/ checkpoints directly or add its trainer." >&2
      exit 2
      ;;
    *)
      echo "[unsupported] Unknown method: ${method}" >&2
      exit 2
      ;;
  esac

  local specific_var
  specific_var="$(echo "${method}_${final_env}_RUN_GROUP_BASE" | tr '[:lower:]' '[:upper:]')"
  local legacy_var
  legacy_var="$(echo "${method}_RUN_GROUP_BASE" | tr '[:lower:]' '[:upper:]')"
  if [[ -n "${!specific_var-}" ]]; then
    RUN_GROUP="${!specific_var}"
  elif [[ -n "${!legacy_var-}" ]]; then
    RUN_GROUP="${!legacy_var}"
  else
    RUN_GROUP="${default_group}"
  fi
}

copy_checkpoint() {
  local method="$1"
  local seed="$2"
  local run_group="$3"
  local pretrain_env="$4"
  local final_env="$5"
  local checkpoint_epoch="$6"
  local dst_dir="${FINAL_ROOT}/${final_env}/${method^^}"

  local algo_suffix
  case "${method}" in
    susd) algo_suffix="metra" ;;
    *) algo_suffix="${method}" ;;
  esac

  local run_dir
  run_dir="$(ls -td "exp/${run_group}"/sd"$(printf '%03d' "${seed}")"*"_${pretrain_env}_${algo_suffix}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${run_dir}" ]]; then
    echo "[error] Could not find run dir for env=${final_env} method=${method} seed=${seed} under exp/${run_group}" >&2
    return 1
  fi

  local option_src="${run_dir}/option_policy${checkpoint_epoch}.pt"
  local encoder_src="${run_dir}/traj_encoder${checkpoint_epoch}.pt"
  if [[ ! -f "${option_src}" || ! -f "${encoder_src}" ]]; then
    option_src="$(ls -v "${run_dir}"/option_policy*.pt 2>/dev/null | tail -n 1 || true)"
    encoder_src="$(ls -v "${run_dir}"/traj_encoder*.pt 2>/dev/null | tail -n 1 || true)"
  fi
  if [[ -z "${option_src}" || -z "${encoder_src}" ]]; then
    echo "[error] No option/traj_encoder checkpoint found in ${run_dir}" >&2
    return 1
  fi

  mkdir -p "${dst_dir}" "${dst_dir}/seed_${seed}"
  cp "${option_src}" "${dst_dir}/seed_${seed}/option_policy${checkpoint_epoch}.pt"
  cp "${encoder_src}" "${dst_dir}/seed_${seed}/traj_encoder${checkpoint_epoch}.pt"
  cp "${option_src}" "${dst_dir}/option_policy${checkpoint_epoch}.pt"
  cp "${encoder_src}" "${dst_dir}/traj_encoder${checkpoint_epoch}.pt"
  echo "[saved] env=${final_env} method=${method} seed=${seed}"
  echo "        ${dst_dir}/option_policy${checkpoint_epoch}.pt"
  echo "        ${dst_dir}/traj_encoder${checkpoint_epoch}.pt"
}

echo "[suite] envs=${ENVS} methods=${METHODS} seeds=${SEEDS} jobs=${total_jobs} total_env_steps=${TOTAL_ENV_STEPS} n_parallel=${SUSD_N_PARALLEL} traj_batch_size=${SUSD_TRAJ_BATCH_SIZE}"

for env_name in ${ENVS}; do
  canonical_env "${env_name}"
  max_path_length="${MAX_PATH_LENGTH:-${DEFAULT_MAX_PATH_LENGTH}}"
  steps_per_epoch=$((SUSD_TRAJ_BATCH_SIZE * max_path_length))
  if [[ "${steps_per_epoch}" -le 0 ]]; then
    echo "[error] steps_per_epoch must be positive. Got traj_batch_size=${SUSD_TRAJ_BATCH_SIZE}, max_path_length=${max_path_length}" >&2
    exit 2
  fi
  computed_epochs=$(((TOTAL_ENV_STEPS + steps_per_epoch - 1) / steps_per_epoch))
  env_epochs="${requested_epochs:-${computed_epochs}}"
  checkpoint_epoch="${requested_checkpoint_epoch:-${env_epochs}}"

  echo "[env] requested=${env_name} pretrain_env=${PRETRAIN_ENV} final_env=${FINAL_ENV} max_path_length=${max_path_length} steps_per_epoch=${steps_per_epoch} epochs=${env_epochs} checkpoint_epoch=${checkpoint_epoch}"

  for method in ${METHODS}; do
    resolve_run_group "${method}" "${FINAL_ENV}" "${ENV_TAG}"

    for seed in ${SEEDS}; do
      job_idx=$((job_idx + 1))
      job_start=$(date +%s)
      suite_elapsed=$((job_start - train_suite_start))
      echo "[train] job=${job_idx}/${total_jobs} env=${FINAL_ENV} method=${method} seed=${seed} run_group=${RUN_GROUP} epochs=${env_epochs} suite_elapsed=${suite_elapsed}s"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] SUSD_ENV=${PRETRAIN_ENV} SUSD_TRAIN_METHOD=${method} SUSD_RUN_GROUP=${RUN_GROUP} SUSD_SEED=${seed} SUSD_N_EPOCHS=${env_epochs} SUSD_MAX_PATH_LENGTH=${max_path_length} ${PYTHON_BIN} src/pretrain.py"
        continue
      fi

      SUSD_ENV="${PRETRAIN_ENV}" \
      SUSD_TRAIN_METHOD="${method}" \
      SUSD_RUN_GROUP="${RUN_GROUP}" \
      SUSD_SEED="${seed}" \
      SUSD_N_EPOCHS="${env_epochs}" \
      SUSD_MAX_PATH_LENGTH="${max_path_length}" \
      "${PYTHON_BIN}" src/pretrain.py

      if [[ "${SKIP_COPY}" == "1" ]]; then
        echo "[skip-copy] env=${FINAL_ENV} method=${method} seed=${seed}; smoke run only"
      else
        copy_checkpoint "${method}" "${seed}" "${RUN_GROUP}" "${PRETRAIN_ENV}" "${FINAL_ENV}" "${checkpoint_epoch}"
      fi
      job_end=$(date +%s)
      job_elapsed=$((job_end - job_start))
      echo "[train-done] job=${job_idx}/${total_jobs} env=${FINAL_ENV} method=${method} seed=${seed} elapsed=${job_elapsed}s"
    done
  done
done

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  if [[ " ${ENVS} " == *" ant "* ]]; then
    METHODS="${METHODS}" \
    CHECKPOINT_ROOT="${FINAL_ROOT}/ant" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/run_ant_heading_counterfactual.sh
  else
    echo "[skip-eval] RUN_EVAL_AFTER only supports ant heading counterfactual eval." >&2
  fi
fi
