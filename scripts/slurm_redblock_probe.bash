#!/usr/bin/env bash
# A small diagnostic GPU job. It reads the released checkpoint and train/val states.
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO_DIR:?}" "${PYTHON_BIN:?}" "${SOURCE_MODEL_DIR:?}" "${DATA_ROOT:?}" "${OUT_ROOT:?}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ "$(git rev-parse HEAD)" == "${SUBMITTED_COMMIT:?}" ]] || {
  echo "Code checkout changed after submission" >&2; exit 2
}
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU allocated"'
"$PYTHON_BIN" scripts/probe_redblock_alignment.py \
  --source-model-dir "$SOURCE_MODEL_DIR" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/analysis" \
  --train-samples 256 --val-samples 128 \
  --oracle-width 32 --oracle-steps 200 \
  --device cuda \
  --wandb-project adajepa_transition_film_diagnostics
