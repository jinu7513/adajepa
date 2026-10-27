#!/usr/bin/env bash
# Submitted by submit_full_transition.bash; Slurm may execute a spool copy.
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${SLURM_JOB_ID:?This script must run under sbatch}"
: "${REPO_DIR:?}" "${PYTHON_BIN:?}" "${DATASET_DIR:?}" "${SOURCE_MODEL_DIR:?}" "${TARGETS:?}" "${RUN_ROOT:?}"
stage="${1:?Missing stage}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
submitted_git="$(sed -n 's/^git=//p' "$RUN_ROOT/submission.txt" | head -n 1)"
[[ "$(git rev-parse HEAD)" == "$submitted_git" ]] || {
  echo "Repository commit changed since submission" >&2; exit 2;
}
sha256sum -c "$RUN_ROOT/input_sha256.txt" >/dev/null
exec 9>"$RUN_ROOT/logs/$stage.lock"
flock -n 9 || { echo "Stage $stage already running" >&2; exit 1; }
echo "stage=$stage job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"

case "$stage" in
  heldout)
    "$PYTHON_BIN" -m pytest tests/test_support_transition.py tests/test_support_mpc_integration.py tests/test_support_train_integration.py -q
    "$PYTHON_BIN" scripts/generate_transition_colors.py \
      --data-root "$DATASET_DIR/pusht_noise" --heldout-only --preview-only
    "$PYTHON_BIN" scripts/generate_transition_colors.py \
      --data-root "$DATASET_DIR/pusht_noise" --heldout-only
    ;;
  train_*)
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No GPU allocated"'
    variant="${stage#train_}"
    if [[ "$variant" == support_identity_vanilla ]]; then
      args=(--identity-only --feature-mode delta --affine-type mlp)
    else
      IFS='_' read -r prefix feature base affine <<< "$variant"
      [[ "$prefix" == support && "$feature" =~ ^(delta|residual)$ && "$base" =~ ^(frozen|trainable)$ && "$affine" =~ ^(mlp|linear)$ ]] || {
        echo "Unknown support variant $variant" >&2; exit 2;
      }
      args=(--feature-mode "$feature" --affine-type "$affine")
      if [[ "$base" == trainable ]]; then args+=(--train-base-predictor); else args+=(--no-train-base-predictor); fi
    fi
    run_dir="$RUN_ROOT/models/$variant"
    mkdir -p "$run_dir"
    "$PYTHON_BIN" train_transition_support.py \
      --source-dir "$SOURCE_MODEL_DIR" --data-dir "$DATASET_DIR" \
      --run-dir "$run_dir" "${args[@]}" --wandb-name "transition-$variant-seed0" \
      2>&1 | tee -a "$run_dir/console.log"
    [[ -f "$run_dir/checkpoints/model_3.pth" ]] || {
      echo "Epoch-3 checkpoint missing for $variant" >&2; exit 1;
    }
    ;;
  offline_*)
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No GPU allocated"'
    variant="${stage#offline_}"
    model_dir="$RUN_ROOT/models/$variant"
    output_dir="$RUN_ROOT/offline/$variant"
    if [[ -s "$output_dir/results.csv" ]]; then
      echo "Skipping completed offline $variant"
      exit 0
    fi
    "$PYTHON_BIN" eval_transition_support.py \
      --model-dir "$model_dir" --data-dir "$DATASET_DIR" \
      --output-dir "$output_dir" --max-batches="${SUPPORT_EVAL_BATCHES:-25}" \
      --max-episodes="${SUPPORT_EVAL_EPISODES:-50}" \
      --wandb-project=adajepa_transition_support_offline
    ;;
  mpc_*)
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No GPU allocated"'
    if [[ "$stage" =~ ^mpc_(support_.*)_k(0|2|4|8)$ ]]; then
      variant="${BASH_REMATCH[1]}"
      k="${BASH_REMATCH[2]}"
    else
      echo "Invalid MPC stage $stage" >&2; exit 2
    fi
    if [[ "$variant" == support_identity_vanilla && "$k" != 0 ]]; then
      echo "Identity control only supports K=0" >&2; exit 2
    fi
    model_dir="$RUN_ROOT/models/$variant"
    [[ -f "$model_dir/checkpoints/model_3.pth" ]] || {
      echo "MPC model not ready: $model_dir" >&2; exit 2;
    }
    for condition in default redBlock; do
        case_root="$RUN_ROOT/eval/$variant/k$k/$condition"
        mkdir -p "$case_root"
        if [[ -f "$case_root/done.txt" ]]; then
          echo "Skipping completed $variant/k$k/$condition"
          continue
        fi
        attempt="$case_root/attempt-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$attempt"
        color_override='env_kwargs_override=null'
        if [[ "$condition" == redBlock ]]; then color_override='env_kwargs_override={color:Red}'; fi
        "$PYTHON_BIN" plan.py --config-name adajepa_plan_gd_pushobj.yaml \
          "ckpt_base_path=$model_dir" "model_name=$(basename "$model_dir")" \
          model_epoch=3 "eval_data_path=$TARGETS" \
          seed=100 n_evals=50 planner.max_iter=30 ood_corruption=null \
          planner.adapt.lr=0 planner.adapt.steps=0 planner.adapt.finetune_encoder=false \
          "$color_override" "+support_k_override=$k" \
          +wandb_logging=true +wandb_project=adajepa_transition_support_mpc \
          "+wandb_group=$variant" \
          "+wandb_run_name=$variant-k$k-$condition-frozen-gd-seed100" \
          "hydra.run.dir=$attempt" 2>&1 | tee "$attempt/console.log"
        [[ -s "$attempt/logs.json" ]] && grep -q '"paper/success_rate_pct"' "$attempt/logs.json" || {
          echo "No success-rate curve: $attempt" >&2; exit 1;
        }
        printf '%s\n' "$attempt" > "$case_root/done.txt"
    done
    ;;
  aggregate_support)
    "$PYTHON_BIN" scripts/aggregate_transition_support.py --run-root "$RUN_ROOT"
    ;;
  *) echo "Unknown support stage: $stage" >&2; exit 2 ;;
esac
echo "stage=$stage completed=$(date -Is)"
