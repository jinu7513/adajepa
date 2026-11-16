#!/usr/bin/env bash
# Paired default/purple/red observed-video rollout diagnostics.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${REPO_DIR:?}" "${OUT_ROOT:?}" "${DATASET_DIR:?}"
track="${1:?}"; arm="${2:?}"
case "$track/$arm" in
  cross_color_jepa/same_fixed|cross_color_jepa/same_window_random|cross_color_jepa/cross_fixed|cross_color_jepa/cross_window_random|mi_jepa/default_jepa_baseline|mi_jepa/default_jepa_plus_inverse) ;;
  *) echo "Unknown track/arm" >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
model_dir="$OUT_ROOT/$track/models/$arm"
test -f "$model_dir/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
"$python_bin" -m scripts.eval_two_track_latent \
  --model-dir "$model_dir" --data-root "$DATASET_DIR/pusht_noise" \
  --output "$OUT_ROOT/$track/latent/$arm" --epoch "${TRAIN_EPOCHS:-3}" \
  --source-default
