#!/usr/bin/env bash
# Default matrix: A/B/C plus all eight support-conditioning architecture arms.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_ROOT="${1:?Usage: CIRCE_PARTITION=... CIRCE_TIME=... bash $0 ABS_RUN_ROOT}"
[[ "$RUN_ROOT" = /* ]] || { echo "Run root must be absolute" >&2; exit 2; }
: "${CIRCE_PARTITION:?}" "${CIRCE_TIME:?}"

# Reuse the existing A/B/C submission and its checked input manifest.
bash "$REPO/scripts/submit_transition_film.bash" "$RUN_ROOT"
prepare="$(sed -n 's/^prepare=//p' "$RUN_ROOT/job_ids.txt" | tail -n 1)"
aggregate_abc="$(sed -n 's/^aggregate=//p' "$RUN_ROOT/job_ids.txt" | tail -n 1)"
[[ -n "$prepare" && -n "$aggregate_abc" ]] || { echo "A/B/C jobs were not submitted" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
DATASET_DIR="${DATASET_DIR:-$REPO/data_full}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$REPO/checkpoints/pusht_visual_shift}"
TARGETS="${TARGETS:-$REPO/data/pushobj_eval/val_T/plan_targets.pkl}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/offline"
jobs_file="$RUN_ROOT/support_job_ids-$(date +%Y%m%d_%H%M%S).txt"
: > "$jobs_file"

gpu_options=(--parsable --nodes=1 --ntasks=1 --cpus-per-task="${CIRCE_CPUS:-8}"
  --mem="${CIRCE_MEM:-64G}" --gres="${CIRCE_GPU_GRES:-gpu:1}"
  --partition="$CIRCE_PARTITION" --time="$CIRCE_TIME" --chdir="$REPO" --requeue)
cpu_options=(--parsable --nodes=1 --ntasks=1 --cpus-per-task="${CIRCE_CPU_CPUS:-8}"
  --mem="${CIRCE_CPU_MEM:-64G}" --partition="${CIRCE_CPU_PARTITION:-$CIRCE_PARTITION}"
  --time="${CIRCE_CPU_TIME:-1-00:00:00}" --chdir="$REPO" --requeue)
if [[ -n "${CIRCE_QOS:-}" ]]; then gpu_options+=(--qos="$CIRCE_QOS"); cpu_options+=(--qos="$CIRCE_QOS"); fi
if [[ -n "${CIRCE_ACCOUNT:-}" ]]; then gpu_options+=(--account="$CIRCE_ACCOUNT"); cpu_options+=(--account="$CIRCE_ACCOUNT"); fi

submit_stage() {
  local kind="$1" stage="$2" dependency="${3:-}" id
  local -a options
  if [[ "$kind" == gpu ]]; then options=("${gpu_options[@]}"); else options=("${cpu_options[@]}"); fi
  options+=(--job-name="support-$stage" --output="$RUN_ROOT/logs/$stage-%j.out"
    --error="$RUN_ROOT/logs/$stage-%j.out")
  [[ -z "$dependency" ]] || options+=(--dependency="afterok:$dependency")
  id="$(sbatch "${options[@]}" --export=ALL,REPO_DIR="$REPO",PYTHON_BIN="$PYTHON_BIN",DATASET_DIR="$DATASET_DIR",SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR",TARGETS="$TARGETS",RUN_ROOT="$RUN_ROOT" \
    "$REPO/scripts/slurm_support_stage.bash" "$stage")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Bad sbatch job id: $id" >&2; exit 2; }
  printf '%s=%s\n' "$stage" "$id" >> "$jobs_file"
  printf '%s=%s\n' "$stage" "$id" >> "$RUN_ROOT/job_ids.txt"
  printf '%s\n' "$id"
}

heldout="$(submit_stage cpu heldout "$prepare")"
prior_train=""
offline_ids=()
mpc_ids=()
identity_train="$(submit_stage gpu train_support_identity_vanilla "$prepare")"
prior_train="$identity_train"
offline_ids+=("$(submit_stage gpu offline_support_identity_vanilla "$identity_train:$heldout")")
mpc_ids+=("$(submit_stage gpu mpc_support_identity_vanilla_k0 "$identity_train")")
for feature in delta residual; do
  for base in frozen trainable; do
    for affine in mlp linear; do
      variant="support_${feature}_${base}_${affine}"
      dependency="$prepare"
      [[ -z "$prior_train" ]] || dependency="$prepare:$prior_train"
      trained="$(submit_stage gpu "train_$variant" "$dependency")"
      prior_train="$trained"
      offline_ids+=("$(submit_stage gpu "offline_$variant" "$trained:$heldout")")
      for k in 0 2 4 8; do
        mpc_ids+=("$(submit_stage gpu "mpc_${variant}_k$k" "$trained")")
      done
    done
  done
done
all_dependencies="$aggregate_abc"
for id in "${offline_ids[@]}" "${mpc_ids[@]}"; do all_dependencies+=":$id"; done
submit_stage cpu aggregate_support "$all_dependencies" >/dev/null
echo "All A/B/C and support ablation jobs submitted; job IDs: $RUN_ROOT/job_ids.txt"
echo "Use squeue -u \$USER or inspect $RUN_ROOT/logs. SSH disconnect does not cancel sbatch jobs."
