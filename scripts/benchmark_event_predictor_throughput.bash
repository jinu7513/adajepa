#!/usr/bin/env bash
# Compare short Event-predictor runs on one GPU without saving model checkpoints.
# Usage: DATASET_DIR=/absolute/path/data_full bash scripts/benchmark_event_predictor_throughput.bash /absolute/path/checkpoint_latest.pt
# Set BENCH_ONLY_CASE=b32_w4,b64_w4 to run those two cases sequentially.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/step-10000-event-checkpoint" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/data_full}"
BENCH_N_ROLLOUT="${BENCH_N_ROLLOUT:-8}"
BENCH_ONLY_CASE="${BENCH_ONLY_CASE:-}"

case "$BENCH_ONLY_CASE" in
  '') CASE_SPECS=('b8_w1:8:1' 'b16_w1:16:1' 'b32_w1:32:1' 'b32_w2:32:2') ;;
  b8_w1) CASE_SPECS=('b8_w1:8:1') ;;
  b16_w1) CASE_SPECS=('b16_w1:16:1') ;;
  b32_w1) CASE_SPECS=('b32_w1:32:1') ;;
  b32_w2) CASE_SPECS=('b32_w2:32:2') ;;
  b32_w4) CASE_SPECS=('b32_w4:32:4') ;;
  b64_w4) CASE_SPECS=('b64_w4:64:4') ;;
  b32_w4,b64_w4) CASE_SPECS=('b32_w4:32:4' 'b64_w4:64:4') ;;
  b128_w4) CASE_SPECS=('b128_w4:128:4') ;;
  *) echo "Unknown BENCH_ONLY_CASE: $BENCH_ONLY_CASE" >&2; exit 2 ;;
esac

if [ ! -x "$PYTHON_BIN" ] || [ ! -f "$1" ]; then
  echo "Missing executable Python or Event checkpoint: $PYTHON_BIN / $1" >&2
  exit 1
fi
if [[ ! "$BENCH_N_ROLLOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCH_N_ROLLOUT must be a positive integer" >&2
  exit 2
fi
if [ ! -f "$DATASET_DIR/pusht_noise/train/seq_lengths.pkl" ] ||
   [ ! -f "$DATASET_DIR/pusht_noise/val/seq_lengths.pkl" ]; then
  echo "Full PushT dataset not found under $DATASET_DIR/pusht_noise" >&2
  exit 1
fi

EVENT_CKPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
export TRACKA_EVENT_ENCODER_CKPT="$EVENT_CKPT" DATASET_DIR

cd "$REPO_DIR"
"$PYTHON_BIN" -c 'from tracka.common import load_checkpoint; import sys; c=load_checkpoint(sys.argv[1]); assert c.get("schema_version") == "tracka-event-v1" and c.get("global_step") == 10000, "Expected a step-10000 Event checkpoint"' "$EVENT_CKPT"

# TrajSlicerDataset makes one window for each valid start frame.
WINDOW_COUNTS="$("$PYTHON_BIN" -c 'import pickle,sys; from pathlib import Path; root=Path(sys.argv[1]); n=int(sys.argv[2]); counts=[]; [(counts.append(sum(max(0,int(t)-20+1) for t in pickle.load(open(root/split/"seq_lengths.pkl","rb"))[:n]))) for split in ("train","val")]; print(*counts)' "$DATASET_DIR/pusht_noise" "$BENCH_N_ROLLOUT")"
read -r TRAIN_WINDOWS VAL_WINDOWS <<< "$WINDOW_COUNTS"
if [ "$TRAIN_WINDOWS" -eq 0 ] || [ "$VAL_WINDOWS" -eq 0 ]; then
  echo "Selected trajectories contain no valid training or validation windows" >&2
  exit 1
fi

BENCH_ROOT="$REPO_DIR/bench_outputs/event_predictor_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$BENCH_ROOT"
SUMMARY="$BENCH_ROOT/summary.csv"
printf 'case,batch_size,num_workers,train_windows,val_windows,elapsed_seconds,status\n' > "$SUMMARY"
echo "Benchmark output: $BENCH_ROOT"
echo "First $BENCH_N_ROLLOUT trajectories per split: train=$TRAIN_WINDOWS windows, val=$VAL_WINDOWS windows"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
fi

# Same data, model, precision and optimizer settings; only batch/workers change.
for case_spec in "${CASE_SPECS[@]}"; do
  IFS=: read -r case_name batch_size num_workers <<< "$case_spec"
  RUN_DIR="$BENCH_ROOT/$case_name"
  mkdir "$RUN_DIR"
  echo "Starting $case_name at $(date -Is)"
  started_at="$(date +%s)"
  if "$PYTHON_BIN" train.py --config-name train \
      env=pusht encoder=tracka_event_patch env.dataset.include_previous_frames=true \
      "env.dataset.n_rollout=$BENCH_N_ROLLOUT" "env.num_workers=$num_workers" \
      model.train_encoder=false model.train_predictor=true model.train_decoder=false \
      has_decoder=false training.save_frozen_encoder=true \
      training.epochs=1 "training.batch_size=$batch_size" training.seed=0 \
      training.mixed_precision=bf16 training.rollout_eval_count=0 \
      training.save_every_x_iterations=0 training.save_every_x_epoch=2 \
      wandb_project=adajepa_trackA_event_predictor_bench \
      "wandb_run_name=event-bench-n${BENCH_N_ROLLOUT}-${case_name}" \
      "hydra.run.dir=$RUN_DIR" 2>&1 | tee "$RUN_DIR/console.log"; then
    status=ok
  else
    status=failed
  fi
  elapsed_seconds="$(( $(date +%s) - started_at ))"
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$case_name" "$batch_size" "$num_workers" "$TRAIN_WINDOWS" "$VAL_WINDOWS" "$elapsed_seconds" "$status" >> "$SUMMARY"
  echo "Finished $case_name: $status in ${elapsed_seconds}s"
  if [ "$status" != ok ]; then
    echo "Stopping after failure. Inspect $RUN_DIR/console.log; completed cases remain in $SUMMARY" >&2
    exit 1
  fi
done

echo "Completed. Compare elapsed_seconds in $SUMMARY (includes setup and validation)."
