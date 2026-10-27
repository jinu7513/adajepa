#!/usr/bin/env bash
# Slurm payload; sbatch copies this script to a spool directory, so use REPO_DIR.
set -euo pipefail
export PYTHONUNBUFFERED=1
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run through sbatch, not on a login node" >&2; exit 2; }
: "${REPO_DIR:?}" "${PYTHON_BIN:?}" "${DATASET_DIR:?}" "${SOURCE_MODEL_DIR:?}" "${TARGETS:?}" "${RUN_ROOT:?}"
STAGE="${1:?Missing stage}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
submitted_git="$(sed -n 's/^git=//p' "$RUN_ROOT/submission.txt" | head -n 1)"
[[ "$(git rev-parse HEAD)" == "$submitted_git" ]] || {
  echo "Repository commit changed after submission; refusing a mixed-code experiment" >&2; exit 2;
}
sha256sum -c "$RUN_ROOT/input_sha256.txt" >/dev/null || {
  echo "Source checkpoint or evaluation targets changed after submission" >&2; exit 2;
}
export DATASET_DIR
exec 9>"$RUN_ROOT/logs/$STAGE.lock"
flock -n 9 || { echo "Stage $STAGE already running in $RUN_ROOT" >&2; exit 1; }
echo "stage=$STAGE job=$SLURM_JOB_ID host=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" -m pytest tests/test_transition_film.py -q
    "$PYTHON_BIN" scripts/generate_transition_colors.py \
      --data-root "$DATASET_DIR/pusht_noise" --preview-only \
      --max-default-mae="${COLOR_MAX_DEFAULT_MAE:-0.06}"
    "$PYTHON_BIN" scripts/generate_transition_colors.py \
      --data-root "$DATASET_DIR/pusht_noise" \
      --max-default-mae="${COLOR_MAX_DEFAULT_MAE:-0.06}"
    ;;
  train_B|train_C)
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU allocated"'
    [[ -f "$DATASET_DIR/pusht_noise/transition_color_manifest.json" ]] || {
      echo "Paired-color dataset manifest missing" >&2; exit 2;
    }
    variant="${STAGE#train_}"
    run_dir="$RUN_ROOT/models/$variant"
    mkdir -p "$run_dir"
    predictor=vit
    if [[ "$variant" == C ]]; then predictor=vit_transition; fi
    completed=0
    if [[ -f "$run_dir/checkpoints/model_latest.pth" ]]; then
      [[ -f "$run_dir/hydra.yaml" ]] || { echo "Checkpoint has no hydra.yaml" >&2; exit 2; }
      completed="$("$PYTHON_BIN" - "$run_dir" "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" "$predictor" <<'PY'
import sys
import torch
from omegaconf import OmegaConf
run, source, predictor = sys.argv[1:]
cfg = OmegaConf.load(run + "/hydra.yaml")
if cfg.training.initial_checkpoint != source or cfg.predictor._target_ != (
    "models.vit.TransitionFiLMPredictor" if predictor == "vit_transition" else "models.vit.ViTPredictor"
):
    raise SystemExit("Unsafe resume: source checkpoint or predictor differs")
checkpoint = torch.load(run + "/checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
if not all(key in checkpoint for key in ("encoder", "predictor", "action_encoder", "proprio_encoder", "predictor_optimizer", "action_encoder_optimizer")):
    raise SystemExit("Incomplete model checkpoint; refusing resume")
print(int(checkpoint["epoch"]))
PY
)"
    fi
    (( completed <= 3 )) || { echo "Completed epoch exceeds agreed 3-epoch budget" >&2; exit 2; }
    remaining=$((3 - completed))
    if (( remaining == 0 )); then
      if [[ ! -f "$run_dir/checkpoints/model_3.pth" ]]; then
        cp "$run_dir/checkpoints/model_latest.pth" "$run_dir/checkpoints/model_3.pth"
      fi
      echo "$variant already completed epoch 3"
      exit 0
    fi
    echo "Training $variant from completed epoch $completed to epoch 3"
    "$PYTHON_BIN" train.py --config-name train \
      env=pusht_transition predictor="$predictor" \
      model.train_encoder=false model.train_predictor=true model.train_decoder=false \
      has_decoder=false training.epochs="$remaining" training.batch_size=32 \
      training.seed=0 training.save_every_x_iterations=0 training.save_every_x_epoch=1 \
      env.num_workers=8 \
      "+training.initial_checkpoint=$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
      +training.save_frozen_encoder=true \
      +wandb_project=adajepa_transition_film \
      "+wandb_run_name=transition-film-$variant-seed0" \
      "hydra.run.dir=$run_dir"
    [[ -f "$run_dir/checkpoints/model_3.pth" ]] || {
      echo "Training returned but epoch-3 checkpoint is missing" >&2; exit 1;
    }
    ;;
  eval_A|eval_B|eval_C)
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU allocated"'
    variant="${STAGE#eval_}"
    if [[ "$variant" == A ]]; then
      model_dir="$SOURCE_MODEL_DIR"
      model_epoch=latest
    else
      model_dir="$RUN_ROOT/models/$variant"
      model_epoch=3
    fi
    [[ -f "$model_dir/hydra.yaml" && -f "$model_dir/checkpoints/model_${model_epoch}.pth" ]] || {
      echo "Evaluation model missing: $model_dir epoch $model_epoch" >&2; exit 2;
    }
    for condition in default redBlock; do
      case_root="$RUN_ROOT/eval/$variant/$condition"
      mkdir -p "$case_root"
      if [[ -f "$case_root/done.txt" ]]; then
        echo "Skipping completed $variant/$condition"
        continue
      fi
      attempt="$case_root/attempt-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
      mkdir -p "$attempt"
      color_override=("env_kwargs_override=null")
      if [[ "$condition" == redBlock ]]; then
        color_override=("env_kwargs_override={color:Red}")
      fi
      echo "Evaluating $variant/$condition: $attempt"
      "$PYTHON_BIN" plan.py --config-name adajepa_plan_gd_pushobj.yaml \
        "ckpt_base_path=$model_dir" "model_name=$(basename "$model_dir")" \
        "model_epoch=$model_epoch" "eval_data_path=$TARGETS" \
        seed=100 n_evals=50 planner.max_iter=30 ood_corruption=null \
        planner.adapt.lr=0 planner.adapt.steps=0 planner.adapt.finetune_encoder=false \
        "${color_override[@]}" \
        +wandb_logging=true +wandb_project=adajepa_transition_film_mpc \
        "+wandb_group=transition-film-seed0" \
        "+wandb_run_name=film-$variant-$condition-frozen-gd-seed100" \
        "hydra.run.dir=$attempt" 2>&1 | tee "$attempt/console.log"
      [[ -s "$attempt/logs.json" ]] && grep -q '"paper/success_rate_pct"' "$attempt/logs.json" || {
        echo "Missing success-rate curve for $variant/$condition" >&2; exit 1;
      }
      printf '%s\n' "$attempt" > "$case_root/done.txt"
    done
    ;;
  aggregate)
    "$PYTHON_BIN" scripts/aggregate_transition_film.py --run-root "$RUN_ROOT"
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
echo "stage=$STAGE completed=$(date -Is)"
