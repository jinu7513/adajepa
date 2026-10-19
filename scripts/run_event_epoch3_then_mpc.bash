#!/usr/bin/env bash
# Slurm batch payload: resume one epoch if needed, then run Event MPC matrix.
# Submit with site-approved sbatch GPU/partition/QOS/time options; do not run
# alongside an existing trainer writing the same checkpoint directory.
set -euo pipefail
export PYTHONUNBUFFERED=1

# sbatch runs a spool copy of this script, so BASH_SOURCE is not in the repo.
# Our submit command sets --chdir to the repository; REPO_DIR can override it.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  REPO="${REPO_DIR:-$PWD}"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO="$(cd "$REPO" && pwd -P)"
[[ -f "$REPO/train.py" && -f "$REPO/scripts/run_event_mpc_matrix.bash" ]] || {
  echo "Repository not found at $REPO; submit with --chdir=/absolute/path/to/adajepa" >&2
  exit 2
}
cd "$REPO"
MODEL_DIR="${1:?Usage: run_event_epoch3_then_mpc.sh ABSOLUTE_MODEL_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
export DATASET_DIR="${DATASET_DIR:-$REPO/data_full}"
export TRACKA_EVENT_ENCODER_CKPT="${TRACKA_EVENT_ENCODER_CKPT:-$REPO/tracka_outputs/event_encoder_20261010_231557_c3bf41_recovered_from_5000/checkpoint_latest.pt}"
[[ "$MODEL_DIR" = /* ]] || { echo "MODEL_DIR must be absolute" >&2; exit 2; }
[[ -f "$MODEL_DIR/hydra.yaml" ]] || { echo "Missing model config: $MODEL_DIR/hydra.yaml" >&2; exit 2; }
[[ -f "$TRACKA_EVENT_ENCODER_CKPT" ]] || { echo "Missing Event encoder checkpoint" >&2; exit 2; }
[[ -f "$DATASET_DIR/pusht_noise/train/rel_actions.pth" ]] || { echo "Missing PushT training data" >&2; exit 2; }
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "GPU allocation required"'
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "WARNING: not running inside a Slurm job; SSH disconnect may stop this pipeline" >&2
fi

"$PYTHON_BIN" -m pytest tests/test_event_mpc_observations.py -q

if [[ ! -f "$MODEL_DIR/checkpoints/model_3.pth" ]]; then
  [[ -f "$MODEL_DIR/checkpoints/model_2.pth" ]] || { echo "Missing epoch 2 checkpoint" >&2; exit 2; }
  "$PYTHON_BIN" - "$MODEL_DIR/checkpoints/model_latest.pth" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert checkpoint.get("epoch") == 2, "Latest predictor checkpoint must be epoch 2"
PY
  echo "Resuming predictor from epoch 2 to epoch 3 in $MODEL_DIR"
  "$PYTHON_BIN" train.py --config-name train \
    env=pusht encoder=tracka_event_patch \
    env.dataset.include_previous_frames=true env.dataset.n_rollout=null env.num_workers=8 \
    env.dataset.video_reader_cache_size=4 env.dataset.video_frame_cache_size=32 \
    env.dataset.merge_video_reads=true env.dataset.window_block_size=2 \
    model.train_encoder=false model.train_predictor=true model.train_decoder=false \
    has_decoder=false training.save_frozen_encoder=true training.epochs=1 \
    training.batch_size=32 training.seed=0 training.mixed_precision=bf16 \
    training.rollout_eval_count=0 training.save_every_x_iterations=0 \
    training.save_every_x_epoch=1 \
    wandb_project=adajepa_trackA_event_predictor \
    wandb_run_name=event-patch10k-b32w8-block2-frame32-seed0 \
    "hydra.run.dir=$MODEL_DIR" 2>&1 | tee "$MODEL_DIR/resume_epoch3_console.log"
fi
[[ -f "$MODEL_DIR/checkpoints/model_3.pth" ]] || { echo "Epoch 3 did not complete" >&2; exit 1; }
bash "$REPO/scripts/run_event_mpc_matrix.bash" "$MODEL_DIR"
