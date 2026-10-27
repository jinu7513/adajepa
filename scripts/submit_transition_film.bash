#!/usr/bin/env bash
# Submit the complete A/B/C PushT comparison DAG from a CIRCE login node.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || {
  echo "Commit tracked code changes before submitting a reproducible Slurm run" >&2; exit 2;
}
RUN_ROOT="${1:?Usage: CIRCE_PARTITION=... CIRCE_TIME=... bash $0 ABS_RUN_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
DATASET_DIR="${DATASET_DIR:-$REPO/data_full}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$REPO/checkpoints/pusht_visual_shift}"
TARGETS="${TARGETS:-$REPO/data/pushobj_eval/val_T/plan_targets.pkl}"
: "${CIRCE_PARTITION:?Set CIRCE_PARTITION to your permitted GPU partition}"
: "${CIRCE_TIME:?Set CIRCE_TIME to a permitted GPU walltime}"

for path in "$RUN_ROOT" "$REPO" "$PYTHON_BIN" "$DATASET_DIR" "$SOURCE_MODEL_DIR" "$TARGETS"; do
  [[ "$path" = /* ]] || { echo "Absolute path required: $path" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable missing: $PYTHON_BIN" >&2; exit 2; }
for path in "$SOURCE_MODEL_DIR/hydra.yaml" "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  "$TARGETS"; do
  [[ -f "$path" ]] || { echo "Required file missing: $path" >&2; exit 2; }
done
for split in train val; do
  for filename in states.pth rel_actions.pth velocities.pth seq_lengths.pkl; do
    path="$DATASET_DIR/pusht_noise/$split/$filename"
    [[ -f "$path" ]] || { echo "Required file missing: $path" >&2; exit 2; }
  done
done
command -v sbatch >/dev/null || { echo "sbatch unavailable; run on CIRCE" >&2; exit 2; }
sinfo -h -p "$CIRCE_PARTITION" -o '%P' | grep -q . || {
  echo "Partition $CIRCE_PARTITION not visible in sinfo" >&2; exit 2;
}
"$PYTHON_BIN" - "$SOURCE_MODEL_DIR" <<'PY'
import sys
import torch
from omegaconf import OmegaConf
root = sys.argv[1]
cfg = OmegaConf.load(root + "/hydra.yaml")
if cfg.env.name != "pusht":
    raise SystemExit("Source checkpoint is not a PushT model")
if (int(cfg.num_hist), int(cfg.num_pred), int(cfg.frameskip), int(cfg.concat_dim)) != (3, 1, 5, 1):
    raise SystemExit("Released model temporal/action layout differs from this experiment")
ckpt = torch.load(root + "/checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
required = {"encoder", "predictor", "action_encoder", "proprio_encoder"}
if required - ckpt.keys():
    raise SystemExit(f"Source checkpoint lacks {sorted(required - ckpt.keys())}")
if len(ckpt["predictor"].transformer.layers) != 6:
    raise SystemExit("Released predictor depth differs from the six-block FiLM config")
print(f"Released source epoch={ckpt.get('epoch')}, encoder={type(ckpt['encoder']).__name__}")
PY

if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -d "$RUN_ROOT" ]] || { echo "RESUME=1 needs existing $RUN_ROOT" >&2; exit 2; }
  [[ -f "$RUN_ROOT/submission.txt" && -f "$RUN_ROOT/input_sha256.txt" ]] || {
    echo "RESUME=1 needs the original submission and input hashes" >&2; exit 2;
  }
  original_git="$(sed -n 's/^git=//p' "$RUN_ROOT/submission.txt" | head -n 1)"
  [[ "$original_git" == "$(git -C "$REPO" rev-parse HEAD)" ]] || {
    echo "Repository commit changed since first submission; use a new run root" >&2; exit 2;
  }
  for entry in "repo=$REPO" "source=$SOURCE_MODEL_DIR" "targets=$TARGETS" "dataset=$DATASET_DIR"; do
    grep -Fxq "$entry" "$RUN_ROOT/submission.txt" || {
      echo "Submission input changed: $entry" >&2; exit 2;
    }
  done
else
  [[ ! -e "$RUN_ROOT" ]] || { echo "Run root exists; use a new path or RESUME=1" >&2; exit 2; }
  mkdir -p "$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models" "$RUN_ROOT/eval"
if [[ "${RESUME:-0}" != 1 ]]; then : > "$RUN_ROOT/job_ids.txt"; fi
jobs_file="$RUN_ROOT/job_ids-$(date +%Y%m%d_%H%M%S).txt"
: > "$jobs_file"
export REPO_DIR="$REPO" PYTHON_BIN DATASET_DIR SOURCE_MODEL_DIR TARGETS RUN_ROOT
if [[ "${RESUME:-0}" != 1 ]]; then
  printf 'repo=%s\ngit=%s\nsource=%s\ntargets=%s\ndataset=%s\nsubmitted=%s\n' \
    "$REPO" "$(git -C "$REPO" rev-parse HEAD)" "$SOURCE_MODEL_DIR" "$TARGETS" \
    "$DATASET_DIR" "$(date -Is)" > "$RUN_ROOT/submission.txt"
fi
current_hashes="$(sha256sum "$SOURCE_MODEL_DIR/hydra.yaml" "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" "$TARGETS" \
  "$DATASET_DIR"/pusht_noise/{train,val}/{states.pth,rel_actions.pth,velocities.pth,seq_lengths.pkl})"
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ "$(<"$RUN_ROOT/input_sha256.txt")" == "$current_hashes" ]] || {
    echo "Source checkpoint or targets changed since the first submission" >&2; exit 2;
  }
else
  printf '%s\n' "$current_hashes" > "$RUN_ROOT/input_sha256.txt"
fi

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
  options+=(--job-name="film-$stage" --output="$RUN_ROOT/logs/$stage-%j.out"
    --error="$RUN_ROOT/logs/$stage-%j.out")
  if [[ -n "$dependency" ]]; then options+=(--dependency="afterok:$dependency"); fi
  id="$(sbatch "${options[@]}" --export=ALL,REPO_DIR="$REPO",PYTHON_BIN="$PYTHON_BIN",DATASET_DIR="$DATASET_DIR",SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR",TARGETS="$TARGETS",RUN_ROOT="$RUN_ROOT" \
    "$REPO/scripts/slurm_transition_film_stage.bash" "$stage")"
  id="${id%%;*}"
  printf '%s=%s\n' "$stage" "$id" >> "$jobs_file"
  printf '%s=%s\n' "$stage" "$id" >> "$RUN_ROOT/job_ids.txt"
  printf '%s\n' "$id"
}

prepare="$(submit_stage cpu prepare)"
train_b="$(submit_stage gpu train_B "$prepare")"
train_c="$(submit_stage gpu train_C "$prepare")"
eval_a="$(submit_stage gpu eval_A "$prepare")"
eval_b="$(submit_stage gpu eval_B "$train_b")"
eval_c="$(submit_stage gpu eval_C "$train_c")"
aggregate="$(submit_stage cpu aggregate "$eval_a:$eval_b:$eval_c")"
echo "All jobs submitted. This submission: $jobs_file"
echo "Check: squeue -u \$USER; logs: $RUN_ROOT/logs"
