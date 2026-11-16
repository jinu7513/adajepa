#!/usr/bin/env bash
# CPU stage for paired-color data generation and provenance verification.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO_DIR:?}" "${DATASET_DIR:?}" "${RUN_ROOT:?}"
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
git diff --quiet && git diff --cached --quiet || {
  echo "Commit code changes before preparing the dataset" >&2
  exit 2
}
if [[ -n "${SUBMITTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$SUBMITTED_COMMIT" ]]; then
  echo "Checkout commit changed after submission" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
"$python_bin" scripts/generate_transition_colors.py --data-root "$DATASET_DIR/pusht_noise" --preview-only
if [[ "${PREVIEW_ONLY:-0}" == 1 ]]; then
  echo "Preview complete; inspect paired-color contact sheets before full generation"
  exit 0
fi
"$python_bin" scripts/generate_transition_colors.py --data-root "$DATASET_DIR/pusht_noise"
"$python_bin" scripts/generate_transition_colors.py --data-root "$DATASET_DIR/pusht_noise" --heldout-only
"$python_bin" scripts/audit_gradient_color_data.py \
  --data-root "$DATASET_DIR/pusht_noise" --output "$RUN_ROOT/data_audit.json"
printf 'complete\n' > "$RUN_ROOT/preparation_done.txt"
