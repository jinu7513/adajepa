#!/usr/bin/env bash
# Frozen GD MPC for an independent track arm; no online model adaptation.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${REPO_DIR:?}" "${OUT_ROOT:?}" "${TARGETS:?}"
track="${1:?}"; arm="${2:?}"
case "$track/$arm" in
  cross_color_jepa/same_fixed|cross_color_jepa/same_window_random|cross_color_jepa/cross_fixed|cross_color_jepa/cross_window_random|mi_jepa/default_jepa_baseline|mi_jepa/default_jepa_plus_inverse) ;;
  *) echo "Unknown track/arm" >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
model_dir="$OUT_ROOT/$track/models/$arm"
test -f "$model_dir/hydra.yaml"
test -f "$model_dir/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
test -f "$TARGETS"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "GPU allocation missing"'
for condition in default purpleBlock redBlock; do
  case_dir="$OUT_ROOT/$track/eval/$arm/$condition"
  if [[ -f "$case_dir/done.txt" ]]; then continue; fi
  attempt="$case_dir/attempt-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$attempt"
  color_arg="env_kwargs_override=null"
  case "$condition" in
    purpleBlock) color_arg="env_kwargs_override={color:Purple}" ;;
    redBlock) color_arg="env_kwargs_override={color:Red}" ;;
  esac
  "$python_bin" plan.py --config-name adajepa_plan_gd_pushobj.yaml \
    "model_name=$(basename "$model_dir")" "ckpt_base_path=$model_dir" \
    "model_epoch=${TRAIN_EPOCHS:-3}" "eval_data_path=$TARGETS" \
    seed=100 n_evals=50 planner.max_iter=30 ood_corruption=null \
    planner.adapt.lr=0 planner.adapt.steps=0 planner.adapt.finetune_encoder=false \
    "$color_arg" decode_for_viz=false \
    +wandb_logging=true "+wandb_project=adajepa_${track}_mpc" \
    "+wandb_run_name=$track-$arm-$condition-frozen-gd-seed100" \
    "hydra.run.dir=$attempt" 2>&1 | tee "$attempt/console.log"
  test -s "$attempt/per_iter_logs.json"
  printf '%s\n' "$attempt" > "$case_dir/done.txt"
done
