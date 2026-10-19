#!/usr/bin/env bash
# Evaluate one self-contained Event predictor checkpoint under three PushT views.
set -euo pipefail
export PYTHONUNBUFFERED=1

# Under sbatch BASH_SOURCE points at Slurm's spool copy, not this checkout.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPO="${REPO_DIR:-$PWD}"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO="$(cd "$REPO" && pwd -P)"
[[ -f "$REPO/plan.py" ]] || {
  echo "Repository not found at $REPO; submit with --chdir=/absolute/path/to/adajepa" >&2
  exit 2
}
cd "$REPO"
MODEL_DIR="${1:?Usage: run_event_mpc_matrix.sh ABSOLUTE_MODEL_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
TARGETS="${TARGETS:-$REPO/data/pushobj_eval/val_T/plan_targets.pkl}"
MODEL_EPOCH="${MODEL_EPOCH:-3}"
MODEL_KIND="${MODEL_KIND:-event}" # event | original
N_EVALS="${N_EVALS:-50}"
MAX_ITER="${MAX_ITER:-30}"
SEED="${SEED:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-adajepa_trackA_event_mpc}"
EVAL_ROOT="${EVAL_ROOT:-$REPO/eval_outputs/event_patch_epoch${MODEL_EPOCH}_$(date +%Y%m%d_%H%M%S)}"

[[ "$MODEL_DIR" = /* ]] || { echo "MODEL_DIR must be absolute" >&2; exit 2; }
[[ "$MODEL_KIND" = event || "$MODEL_KIND" = original ]] || { echo "MODEL_KIND must be event or original" >&2; exit 2; }
[[ -f "$MODEL_DIR/hydra.yaml" ]] || { echo "Missing $MODEL_DIR/hydra.yaml" >&2; exit 2; }
[[ -f "$MODEL_DIR/checkpoints/model_${MODEL_EPOCH}.pth" ]] || {
  echo "Missing completed epoch $MODEL_EPOCH checkpoint" >&2; exit 2;
}
[[ -f "$TARGETS" ]] || { echo "Missing evaluation targets: $TARGETS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ ! -e "$EVAL_ROOT" ]] || { echo "Evaluation output already exists: $EVAL_ROOT" >&2; exit 2; }

"$PYTHON_BIN" - "$MODEL_DIR/hydra.yaml" "$MODEL_DIR/checkpoints/model_${MODEL_EPOCH}.pth" "$MODEL_EPOCH" "$MODEL_KIND" <<'PY'
from omegaconf import OmegaConf
import torch
import sys
cfg = OmegaConf.load(sys.argv[1])
assert cfg.env.name == "pusht", "Checkpoint must be a PushT predictor"
kind = sys.argv[4]
if kind == "event":
    assert cfg.encoder._target_ == "models.tracka_event_patch.TrackAEventPatchEncoder", \
        "Checkpoint must use the Event patch encoder"
    assert cfg.training.save_frozen_encoder, "Event checkpoint must contain the encoder"
else:
    assert cfg.encoder._target_ != "models.tracka_event_patch.TrackAEventPatchEncoder", \
        "Use MODEL_KIND=event for Event checkpoints"
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
if sys.argv[3] != "latest":
    assert checkpoint.get("epoch") == int(sys.argv[3]), "Checkpoint epoch does not match filename"
assert "predictor" in checkpoint, "Checkpoint is missing the predictor"
if kind == "event":
    assert "encoder" in checkpoint, "Event checkpoint is missing the embedded encoder"
PY

mkdir -p "$EVAL_ROOT"
printf 'condition,method,seed,n_evals,max_iter,output,status\n' > "$EVAL_ROOT/status.csv"
MODEL_NAME="$(basename "$MODEL_DIR")"
VARIANT="${VARIANT:-$MODEL_NAME}"
echo "Event MPC matrix: model=$MODEL_DIR epoch=$MODEL_EPOCH targets=$TARGETS output=$EVAL_ROOT"

run_case() {
  local condition="$1" method="$2" count="$3" max_iter="$4" smoke="$5"
  local run_dir="$EVAL_ROOT/$smoke/$condition/$method"
  local corruption=null
  local -a color_args=("env_kwargs_override=null")
  local -a adapt_args=("planner.adapt.lr=0" "planner.adapt.steps=0"
                       "planner.adapt.finetune_encoder=false")
  local -a smoke_args=()
  case "$condition" in
    default) ;;
    blur) corruption=blur ;;
    redBlock) color_args=("++env_kwargs_override.color=Red") ;;
    *) echo "Unsupported condition: $condition" >&2; return 2 ;;
  esac
  if [[ "$method" == adajepa ]]; then
    adapt_args=("planner.adapt.lr=5e-4" "planner.adapt.steps=1"
                "planner.adapt.finetune_encoder=true" "planner.adapt.encoder_lr=1e-5"
                "planner.adapt.last_layer_only=true" "planner.adapt.encoder_last_layer_only=true")
  fi
  if [[ "$smoke" == smoke ]]; then
    smoke_args=("planner.sub_planner.opt_steps=5")
  fi
  mkdir -p "$run_dir"
  echo "Starting $smoke $condition $method ($count samples, $max_iter MPC steps)"
  if "$PYTHON_BIN" plan.py --config-name adajepa_plan_gd_pushobj.yaml \
      "model_name=$MODEL_NAME" "ckpt_base_path=$MODEL_DIR" "model_epoch=$MODEL_EPOCH" \
      "eval_data_path=$TARGETS" "n_evals=$count" "seed=$SEED" \
      "planner.max_iter=$max_iter" "ood_corruption=$corruption" \
      "${color_args[@]}" "${adapt_args[@]}" "${smoke_args[@]}" \
      "decode_for_viz=false" \
      "+condition=$condition" "+method=$method" \
      "+wandb_logging=true" "+wandb_project=$WANDB_PROJECT" \
      "+wandb_group=event-comparison-epoch${MODEL_EPOCH}-seed${SEED}" \
      "+wandb_run_name=${VARIANT}-epoch${MODEL_EPOCH}-${smoke}-${condition}-${method}-gd-seed${SEED}" \
      "hydra.run.dir=$run_dir" 2>&1 | tee "$run_dir/console.log"; then
    if [[ ! -s "$run_dir/logs.json" ]] || ! grep -q '"paper/success_rate_pct"' "$run_dir/logs.json"; then
      echo "Evaluation returned without paper success-rate logs: $run_dir" >&2
      printf '%s,%s,%s,%s,%s,%s,failed-no-logs\n' \
        "$condition" "$method" "$SEED" "$count" "$max_iter" "$run_dir" >> "$EVAL_ROOT/status.csv"
      return 1
    fi
    printf '%s,%s,%s,%s,%s,%s,ok\n' \
      "$condition" "$method" "$SEED" "$count" "$max_iter" "$run_dir" >> "$EVAL_ROOT/status.csv"
  else
    printf '%s,%s,%s,%s,%s,%s,failed\n' \
      "$condition" "$method" "$SEED" "$count" "$max_iter" "$run_dir" >> "$EVAL_ROOT/status.csv"
    return 1
  fi
}

# Cheap end-to-end checks before the six full evaluations.
if [[ "${RUN_SMOKE:-1}" == 1 ]]; then
  run_case default frozen 1 2 smoke
  run_case default adajepa 1 2 smoke
fi
for condition in default blur redBlock; do
  for method in frozen adajepa; do
    run_case "$condition" "$method" "$N_EVALS" "$MAX_ITER" full
  done
done
echo "All six Event MPC evaluations finished: $EVAL_ROOT/status.csv"
