#!/usr/bin/env bash
# Submit one independent Event predictor recipe. Run this on a CIRCE login node.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: CIRCE_PARTITION=... CIRCE_TIME=... bash $0 A|B|C|D ABS_RUN_DIR ABS_EVENT_CKPT [TARGET_EPOCHS]" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$1" in A|B|C|D) EXPERIMENT="$1" ;; *) echo "Experiment must be A, B, C, or D" >&2; exit 2 ;; esac
RUN_DIR="$2"
EVENT_CKPT="$3"
TARGET_EPOCHS="${4:-3}"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
DATASET_DIR="${DATASET_DIR:-$REPO/data_full}"
: "${CIRCE_PARTITION:?Set CIRCE_PARTITION from sinfo on CIRCE}"
: "${CIRCE_TIME:?Set CIRCE_TIME to a site-approved walltime, e.g. DD-HH:MM:SS}"
[[ "$RUN_DIR" = /* && "$EVENT_CKPT" = /* && "$DATASET_DIR" = /* && "$PYTHON_BIN" = /* ]] || {
  echo "Run, checkpoint, dataset, and Python paths must be absolute" >&2; exit 2;
}
[[ "$TARGET_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "TARGET_EPOCHS must be positive" >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$EVENT_CKPT" && -f "$DATASET_DIR/pusht_noise/train/rel_actions.pth" ]] || {
  echo "Missing Python, Event checkpoint, or full PushT training data" >&2; exit 2;
}
# D uses the checkpoint only to verify that its randomly initialized encoder
# has the same architecture as B; it never initializes weights from it.
command -v sbatch >/dev/null || { echo "sbatch is unavailable; connect to a CIRCE login node" >&2; exit 2; }
sinfo -h -p "$CIRCE_PARTITION" -o '%P' | grep -q . || {
  echo "Partition $CIRCE_PARTITION is not visible in sinfo" >&2; exit 2;
}

if [[ "${RESUME:-0}" = 1 ]]; then
  [[ -f "$RUN_DIR/checkpoints/model_latest.pth" && -f "$RUN_DIR/hydra.yaml" ]] || {
    echo "Resume requires an existing completed-epoch checkpoint and hydra.yaml" >&2; exit 2;
  }
else
  [[ ! -e "$RUN_DIR" ]] || { echo "Run directory exists; use a new path or RESUME=1" >&2; exit 2; }
  mkdir -p "$RUN_DIR"
fi

options=(--parsable --job-name="event-${EXPERIMENT}" --nodes=1 --ntasks=1
  --cpus-per-task="${CIRCE_CPUS:-8}" --mem="${CIRCE_MEM:-64G}"
  --gres="${CIRCE_GPU_GRES:-gpu:1}" --partition="$CIRCE_PARTITION"
  --time="$CIRCE_TIME" --chdir="$REPO"
  --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.out")
if [[ -n "${CIRCE_QOS:-}" ]]; then options+=(--qos="$CIRCE_QOS"); fi
if [[ -n "${CIRCE_ACCOUNT:-}" ]]; then options+=(--account="$CIRCE_ACCOUNT"); fi
job_id="$(sbatch "${options[@]}" "$REPO/scripts/slurm_event_predictor_job.bash" \
  "$EXPERIMENT" "$RUN_DIR" "$EVENT_CKPT" "$DATASET_DIR" "$PYTHON_BIN" "$TARGET_EPOCHS")"
printf 'Submitted job %s\nRun directory: %s\nLog: %s/slurm-%s.out\n' "$job_id" "$RUN_DIR" "$RUN_DIR" "$job_id"
