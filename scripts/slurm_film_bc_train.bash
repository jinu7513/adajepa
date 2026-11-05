#!/usr/bin/env bash
# GPU payload for the logging-fixed B/C comparison. Submit via submit_film_bc_train.bash.
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${SLURM_JOB_ID:?Run this script through sbatch}"
: "${REPO_DIR:?}" "${PYTHON_BIN:?}" "${DATASET_DIR:?}" "${SOURCE_MODEL_DIR:?}" "${RUN_ROOT:?}"
variant="${1:?Expected B or C}"
[[ "$variant" == B || "$variant" == C ]] || { echo "Expected B or C, got $variant" >&2; exit 2; }
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
submitted_git="$(sed -n 's/^git=//p' "$RUN_ROOT/submission.txt" | head -n 1)"
[[ "$(git rev-parse HEAD)" == "$submitted_git" ]] || {
  echo "Code checkout changed after submission" >&2; exit 2
}
sha256sum -c "$RUN_ROOT/input_sha256.txt" >/dev/null || {
  echo "Source checkpoint or color manifest changed after submission" >&2; exit 2
}
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU allocated"'
run_dir="$RUN_ROOT/models/$variant"
mkdir -p "$run_dir"
exec 9>"$RUN_ROOT/logs/train_$variant.lock"
flock -n 9 || { echo "Training $variant is already running" >&2; exit 1; }

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
expected = "models.vit.TransitionFiLMPredictor" if predictor == "vit_transition" else "models.vit.ViTPredictor"
if cfg.training.initial_checkpoint != source or cfg.predictor._target_ != expected:
    raise SystemExit("Unsafe resume: source checkpoint or predictor differs")
checkpoint = torch.load(run + "/checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
required = {"encoder", "predictor", "action_encoder", "proprio_encoder", "predictor_optimizer", "action_encoder_optimizer"}
if required - checkpoint.keys():
    raise SystemExit(f"Incomplete model checkpoint: {sorted(required - checkpoint.keys())}")
print(int(checkpoint["epoch"]))
PY
)"
fi
(( completed >= 0 && completed <= 3 )) || { echo "Invalid completed epoch: $completed" >&2; exit 2; }
remaining=$((3 - completed))
if (( remaining == 0 )); then
  [[ -f "$run_dir/checkpoints/model_3.pth" ]] || {
    echo "Latest checkpoint says epoch 3 but model_3.pth is missing" >&2; exit 2
  }
  echo "$variant already completed three epochs"
  exit 0
fi
echo "Training $variant from completed epoch $completed through epoch 3"
"$PYTHON_BIN" train.py --config-name train \
  env=pusht_transition predictor="$predictor" \
  model.train_encoder=false model.train_predictor=true model.train_decoder=false \
  has_decoder=false training.epochs="$remaining" training.batch_size=32 \
  training.seed=0 training.save_every_x_iterations=0 training.save_every_x_epoch=1 \
  env.num_workers=8 \
  "+training.initial_checkpoint=$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  +training.save_frozen_encoder=true \
  +wandb_project=adajepa_transition_film \
  "+wandb_run_name=transition-film-$variant-loggingfix-seed0" \
  "hydra.run.dir=$run_dir"
[[ -f "$run_dir/checkpoints/model_3.pth" ]] || {
  echo "Training returned without an epoch-3 checkpoint for $variant" >&2; exit 1
}
echo "Completed $variant epoch 3: $run_dir/checkpoints/model_3.pth"
