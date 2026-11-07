#!/usr/bin/env bash
# Run paired PushT MPC comparisons on a CIRCE GPU compute node.
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO_DIR:?}" "${PYTHON_BIN:?}" "${SOURCE_MODEL_DIR:?}" "${TARGETS:?}" "${OUT_ROOT:?}" "${SUBMITTED_COMMIT:?}"
cd "$REPO_DIR"
[[ "$(git rev-parse HEAD)" == "$SUBMITTED_COMMIT" ]] || {
  echo "Checkout changed after submission" >&2; exit 2;
}
[[ -f "$SOURCE_MODEL_DIR/hydra.yaml" && -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" && -f "$TARGETS" ]] || {
  echo "Model or targets missing" >&2; exit 2;
}
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU allocated"'
"$PYTHON_BIN" -m pytest tests/test_latent_adapter_mpc.py -q
mkdir -p "$OUT_ROOT/runs"
{
  echo "git=$SUBMITTED_COMMIT"
  echo "source_model=$SOURCE_MODEL_DIR"
  echo "targets=$TARGETS"
  sha256sum "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" "$TARGETS"
} > "$OUT_ROOT/inputs.txt"
for condition in default redBlock; do
  color_override=("env_kwargs_override=null")
  if [[ "$condition" == redBlock ]]; then
    color_override=("env_kwargs_override={color:Red}")
  fi
  for method in frozen adajepa latent_bias latent_mlp; do
    case_dir="$OUT_ROOT/runs/$condition/$method"
    if [[ -f "$case_dir/done.txt" ]]; then
      echo "Skipping completed $condition/$method"
      continue
    fi
    mkdir -p "$case_dir"
    attempt="$case_dir/attempt-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$attempt"
    adapt=("planner.adapt.mode=adajepa" "planner.adapt.steps=0" "planner.adapt.lr=0")
    if [[ "$method" == adajepa ]]; then
      adapt=("planner.adapt.mode=adajepa" "planner.adapt.steps=1" "planner.adapt.lr=5e-4")
    elif [[ "$method" == latent_bias || "$method" == latent_mlp ]]; then
      adapt=("planner.adapt.mode=$method" "planner.adapt.steps=1" "planner.adapt.lr=5e-4")
    fi
    echo "Starting $condition/$method at $(date -Is)"
    "$PYTHON_BIN" plan.py --config-name adajepa_plan_gd_pushobj.yaml \
      "ckpt_base_path=$SOURCE_MODEL_DIR" "model_name=$(basename "$SOURCE_MODEL_DIR")" \
      model_epoch=latest "eval_data_path=$TARGETS" \
      seed=100 n_evals=50 planner.max_iter=30 ood_corruption=null \
      "${color_override[@]}" "${adapt[@]}" \
      +wandb_logging=true +wandb_project=adajepa_redblock_latent_adapter_mpc \
      "+wandb_run_name=$condition-$method-gd-seed100" \
      "hydra.run.dir=$attempt" 2>&1 | tee "$attempt/console.log"
    "$PYTHON_BIN" - "$attempt/logs.json" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
curve = [row for row in rows if "paper/mpc_step" in row]
if not curve or "paper/success_rate_pct" not in curve[-1]:
    raise SystemExit("Missing MPC success curve")
print("Final:", curve[-1], flush=True)
PY
    printf '%s\n' "$attempt" > "$case_dir/done.txt"
  done
done
"$PYTHON_BIN" scripts/summarize_latent_adapter_mpc.py "$OUT_ROOT"
