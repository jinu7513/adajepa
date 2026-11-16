#!/usr/bin/env bash
# Frozen encoder probe for one completed P/J checkpoint.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO_DIR:?}" "${RUN_ROOT:?}" "${DATASET_DIR:?}"
mode="${1:?Use P or J}"
case "$mode" in P|J) ;; *) echo "Mode must be P or J" >&2; exit 2 ;; esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
if [[ -n "${SUBMITTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$SUBMITTED_COMMIT" ]]; then
  echo "Checkout commit changed after submission" >&2
  exit 2
fi
checkpoint="$RUN_ROOT/models/$mode/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
test -f "$checkpoint"
mkdir -p "$RUN_ROOT/probe/$mode"
"$python_bin" scripts/probe_gradient_encoder.py \
  --checkpoint "$checkpoint" --data-root "$DATASET_DIR/pusht_noise" \
  --output "$RUN_ROOT/probe/$mode"
