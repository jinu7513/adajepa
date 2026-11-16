#!/usr/bin/env bash
# Frozen encoder probe; source-default for B, three-color training for A.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${REPO_DIR:?}" "${OUT_ROOT:?}" "${DATASET_DIR:?}"
track="${1:?}"; arm="${2:?}"
case "$track/$arm" in
  cross_color_jepa/same_fixed|cross_color_jepa/same_window_random|cross_color_jepa/cross_fixed|cross_color_jepa/cross_window_random)
    train_colors=(default blue green); eval_colors=(default purpleBlock redBlock) ;;
  mi_jepa/default_jepa_baseline|mi_jepa/default_jepa_plus_inverse)
    train_colors=(source_default); eval_colors=(source_default purpleBlock redBlock) ;;
  *) echo "Unknown track/arm" >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
checkpoint="$OUT_ROOT/$track/models/$arm/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
test -f "$checkpoint"
"$python_bin" scripts/probe_gradient_encoder.py \
  --checkpoint "$checkpoint" --data-root "$DATASET_DIR/pusht_noise" \
  --output "$OUT_ROOT/$track/probe/$arm" \
  --train-colors "${train_colors[@]}" --eval-colors "${eval_colors[@]}" \
  --contact-probe
