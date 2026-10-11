#!/usr/bin/env bash
# Resume a frozen RGB/Event encoder to 10k, smoke-test, then train a 3-epoch PushT predictor.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/event_encoder_run" >&2
  exit 2
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVENT_RUN_DIR="$(cd "$1" && pwd)"
EVENT_CKPT="$EVENT_RUN_DIR/checkpoint_latest.pt"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/data_full}"
export DATASET_DIR

if [ ! -f "$EVENT_CKPT" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Event checkpoint or executable Python: $EVENT_CKPT / $PYTHON_BIN" >&2
  exit 1
fi
if [ -f "$EVENT_RUN_DIR/run_status.json" ] &&
   grep -q '"status": "running"' "$EVENT_RUN_DIR/run_status.json"; then
  echo "An Event encoder job appears active; do not resume the same checkpoint concurrently" >&2
  exit 1
fi
if [ ! -f "$DATASET_DIR/pusht_noise/train/rel_actions.pth" ]; then
  echo "Full PushT trajectories not found under $DATASET_DIR/pusht_noise" >&2
  exit 1
fi
SMOKE_DIR="$REPO_DIR/checkpoints/pusht_event_patch_10k_smoke"
FULL_DIR="$REPO_DIR/checkpoints/pusht_event_patch_10k_3epoch"
if [ -e "$SMOKE_DIR" ] || [ -e "$FULL_DIR" ]; then
  echo "Refusing to reuse an existing predictor output directory: $SMOKE_DIR or $FULL_DIR" >&2
  exit 1
fi
cd "$REPO_DIR"
"$PYTHON_BIN" -m pytest tests/test_tracka_event_predictor_integration.py -q
STEP="$("$PYTHON_BIN" -c 'from tracka.common import load_checkpoint; import sys; c=load_checkpoint(sys.argv[1]); assert c.get("schema_version")=="tracka-event-v1"; print(c["global_step"])' "$EVENT_CKPT")"
if [ "$STEP" -gt 10000 ]; then
  echo "Event checkpoint is past the requested 10,000 steps: $STEP" >&2
  exit 1
fi
if [ "$STEP" -lt 10000 ]; then
  if [ "$STEP" -eq 5000 ]; then
    cp -n "$EVENT_CKPT" "$EVENT_RUN_DIR/checkpoint_step05000.pt"
  fi
  echo "Resuming Event encoder from step $STEP to 10000"
  "$PYTHON_BIN" -c 'from tracka.common import load_checkpoint; from tracka.event_train import train_event_encoder; import sys; p=sys.argv[1]; c=load_checkpoint(p)["config"]; c["training"]["resume"]=p; c["training"]["total_optimizer_updates"]=10000; c["logging"]["fallback_to_local"]=False; train_event_encoder(c)' "$EVENT_CKPT"
fi
STEP="$("$PYTHON_BIN" -c 'from tracka.common import load_checkpoint; import sys; print(load_checkpoint(sys.argv[1])["global_step"])' "$EVENT_CKPT")"
if [ "$STEP" -ne 10000 ]; then
  echo "Event encoder did not reach step 10000: $STEP" >&2
  exit 1
fi

export TRACKA_EVENT_ENCODER_CKPT="$EVENT_CKPT"

echo "Running 2-trajectory predictor smoke test"
"$PYTHON_BIN" train.py --config-name train \
  env=pusht encoder=tracka_event_patch env.dataset.include_previous_frames=true \
  env.dataset.n_rollout=2 env.num_workers=1 \
  model.train_encoder=false model.train_predictor=true model.train_decoder=false \
  has_decoder=false training.save_frozen_encoder=true \
  training.epochs=1 training.batch_size=2 training.seed=0 training.mixed_precision=no \
  training.rollout_eval_count=0 training.save_every_x_iterations=0 \
  wandb_project=adajepa_trackA_event_predictor \
  wandb_run_name=event-patch-10k-smoke-seed0 \
  "hydra.run.dir=$SMOKE_DIR"
if [ ! -f "$SMOKE_DIR/checkpoints/model_latest.pth" ]; then
  echo "Predictor smoke test did not save a checkpoint" >&2
  exit 1
fi

echo "Running full-data 3-epoch Event predictor training"
"$PYTHON_BIN" train.py --config-name train \
  env=pusht encoder=tracka_event_patch env.dataset.include_previous_frames=true \
  env.dataset.n_rollout=null env.num_workers=1 \
  model.train_encoder=false model.train_predictor=true model.train_decoder=false \
  has_decoder=false training.save_frozen_encoder=true \
  training.epochs=3 training.batch_size=8 training.seed=0 training.mixed_precision=bf16 \
  training.rollout_eval_count=0 training.save_every_x_iterations=0 \
  wandb_project=adajepa_trackA_event_predictor \
  wandb_run_name=event-patch-10k-3epoch-seed0 \
  "hydra.run.dir=$FULL_DIR"
echo "Event predictor pipeline complete: $FULL_DIR"
