#!/usr/bin/env bash
# Frozen GD MPC on matched default, validation-purple, and test-red targets.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO_DIR:?}" "${RUN_ROOT:?}" "${TARGETS:?}" "${DATASET_DIR:?}"
mode="${1:?Use P or J}"
case "$mode" in P|J) ;; *) echo "Mode must be P or J" >&2; exit 2 ;; esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
git diff --quiet && git diff --cached --quiet || {
  echo "Commit code changes before evaluation" >&2
  exit 2
}
if [[ -n "${SUBMITTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$SUBMITTED_COMMIT" ]]; then
  echo "Checkout commit changed after submission" >&2
  exit 2
fi
model_dir="$RUN_ROOT/models/$mode"
test -f "$model_dir/hydra.yaml"
test -f "$model_dir/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
test -f "$TARGETS"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "GPU allocation missing"'
for condition in default purpleBlock redBlock; do
  case_root="$RUN_ROOT/eval/$mode/$condition"
  if [[ -f "$case_root/done.txt" ]]; then
    echo "Skipping completed $mode/$condition"
    continue
  fi
  output="$case_root/attempt-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$output"
  color_arg="env_kwargs_override=null"
  case "$condition" in
    purpleBlock) color_arg="env_kwargs_override={color:Purple}" ;;
    redBlock) color_arg="env_kwargs_override={color:Red}" ;;
  esac
  "$python_bin" plan.py --config-name adajepa_plan_gd_pushobj.yaml \
    "model_name=$(basename "$model_dir")" "ckpt_base_path=$model_dir" \
    "model_epoch=${TRAIN_EPOCHS:-3}" "eval_data_path=$TARGETS" \
    seed=100 n_evals=50 planner.max_iter=30 ood_corruption=null \
    planner.adapt.lr=0 planner.adapt.steps=0 \
    planner.adapt.finetune_encoder=false \
    "$color_arg" decode_for_viz=false \
    +wandb_logging=true +wandb_project=adajepa_gradient_ablation_mpc \
    "+wandb_run_name=gradient-$mode-$condition-frozen-gd-seed100" \
    "hydra.run.dir=$output" 2>&1 | tee "$output/console.log"
  test -s "$output/per_iter_logs.json"
  test -s "$output/logs.json"
  printf '%s\n' "$output" > "$case_root/done.txt"
done
