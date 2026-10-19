#!/usr/bin/env bash
# Slurm payload. Never invoke directly on a login node.
set -euo pipefail
export PYTHONUNBUFFERED=1
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Requires an sbatch allocation" >&2; exit 2; }
EXPERIMENT="$1"
RUN_DIR="$2"
export TRACKA_EVENT_ENCODER_CKPT="$3"
export DATASET_DIR="$4"
PYTHON_BIN="$5"
TARGET_EPOCHS="$6"
# sbatch executes a spool copy of this file; its dirname is not the repo.
REPO="${REPO_DIR:-$PWD}"
REPO="$(cd "$REPO" && pwd -P)"
[[ -f "$REPO/train.py" ]] || {
  echo "Repository not found at $REPO; submit with --chdir=/absolute/path/to/adajepa" >&2
  exit 2
}
cd "$REPO"
case "$EXPERIMENT" in
  A) RECIPE=event_frozen; ENCODER_RECIPE=tracka_event_patch; TRAIN_ENCODER=false; SCOPE=frozen ;;
  B) RECIPE=event_joint; ENCODER_RECIPE=tracka_event_patch; TRAIN_ENCODER=true; SCOPE=full ;;
  C) RECIPE=event_final_shared; ENCODER_RECIPE=tracka_event_patch; TRAIN_ENCODER=true; SCOPE=final_shared ;;
  D) RECIPE=event_scratch_joint; ENCODER_RECIPE=tracka_event_patch_scratch; TRAIN_ENCODER=true; SCOPE=full ;;
  *) echo "Unknown experiment" >&2; exit 2 ;;
esac

# A lock is held for the entire job. Never train two jobs into the same run dir.
exec 9>"$RUN_DIR/.training.lock"
flock -n 9 || { echo "Another job owns $RUN_DIR" >&2; exit 1; }
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "Slurm did not allocate a CUDA GPU"'
printf 'job=%s host=%s experiment=%s git=%s start=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$EXPERIMENT" "$(git rev-parse HEAD)" "$(date -Is)" \
  | tee -a "$RUN_DIR/job_history.log"

if [[ "$EXPERIMENT" == D ]]; then
  "$PYTHON_BIN" - "$TRACKA_EVENT_ENCODER_CKPT" \
    "$REPO/conf/encoder/tracka_event_patch_scratch.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
from tracka.common import load_checkpoint

reference = load_checkpoint(sys.argv[1])  # Architecture reference only; do not load its weights.
scratch = OmegaConf.to_container(OmegaConf.load(sys.argv[2]).model_config, resolve=True)
if reference.get("schema_version") != "tracka-event-v1" or reference.get("global_step") != 10000:
    raise SystemExit("Scratch comparison needs the same 10k Event architecture reference as B")
if dict(reference["config"]["model"]) != scratch:
    raise SystemExit("Scratch Event architecture does not match the pretrained reference")
print("Scratch architecture matches B; no pretrained encoder weights will be loaded")
PY
fi

# A complete epoch is the only supported resume point. Do not use iteration
# checkpointing: the upstream loader cannot restore its mid-epoch cursor.
COMPLETED=0
if [[ -f "$RUN_DIR/checkpoints/model_latest.pth" ]]; then
  COMPLETED="$("$PYTHON_BIN" - "$RUN_DIR" "$TRACKA_EVENT_ENCODER_CKPT" \
    "$DATASET_DIR" "$TRAIN_ENCODER" "$SCOPE" "$EXPERIMENT" <<'PY'
import sys
import torch
from omegaconf import OmegaConf

directory, event_checkpoint, dataset, train_encoder, scope, experiment = sys.argv[1:]
config = OmegaConf.load(directory + "/hydra.yaml")
expected = {
    "encoder._target_": "models.tracka_event_patch.TrackAEventPatchEncoder",
    "encoder.checkpoint_path": None if experiment == "D" else event_checkpoint,
    "env.dataset.data_path": dataset + "/pusht_noise",
    "env.dataset.include_previous_frames": True,
    "env.dataset.n_rollout": None,
    "model.train_encoder": train_encoder == "true",
    "model.train_predictor": True,
    "model.train_decoder": False,
    "has_decoder": False,
    "training.encoder_train_scope": scope,
    "training.batch_size": 32,
    "training.seed": 0,
    "training.save_every_x_iterations": 0,
    "training.strict_optimizer_resume": True,
    "frameskip": 5,
    "num_hist": 3,
    "num_pred": 1,
}
for key, value in expected.items():
    actual = OmegaConf.select(config, key)
    if actual != value:
        raise SystemExit(f"Unsafe resume: {key}={actual!r}, expected {value!r}")
if experiment == "D" and OmegaConf.select(config, "encoder.model_config") is None:
    raise SystemExit("Unsafe resume: scratch Event architecture is missing")
checkpoint = torch.load(directory + "/checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
required = ["predictor_optimizer", "action_encoder_optimizer"]
if train_encoder == "true":
    required.append("encoder_optimizer")
missing = [key for key in required if not isinstance(checkpoint.get(key), dict)]
if missing:
    raise SystemExit(f"Unsafe resume: missing optimizer states {missing}")
print(checkpoint["epoch"])
PY
)"
fi
if (( COMPLETED > TARGET_EPOCHS )); then
  echo "Checkpoint epoch $COMPLETED exceeds target $TARGET_EPOCHS" >&2; exit 2
fi
REMAINING=$((TARGET_EPOCHS - COMPLETED))
if (( REMAINING == 0 )); then
  echo "Target epoch $TARGET_EPOCHS already completed; leaving checkpoint untouched"
  exit 0
fi
export TRACKA_EVENT_ENCODER_CKPT DATASET_DIR
echo "Training experiment $EXPERIMENT from completed epoch $COMPLETED to $TARGET_EPOCHS"
"$PYTHON_BIN" train.py --config-name train \
  env=pusht encoder="$ENCODER_RECIPE" experiment="$RECIPE" \
  training.epochs="$REMAINING" \
  wandb_run_name="event-${EXPERIMENT}-seed0" \
  "hydra.run.dir=$RUN_DIR"
echo "Completed experiment $EXPERIMENT at $(date -Is)" | tee -a "$RUN_DIR/job_history.log"
