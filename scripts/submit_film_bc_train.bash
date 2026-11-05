#!/usr/bin/env bash
# Submit only the three-epoch B/C predictor runs; reuse the prepared color data.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_ROOT="${1:?Usage: CIRCE_PARTITION=... CIRCE_TIME=... bash $0 ABS_NEW_RUN_ROOT}"
[[ "$RUN_ROOT" = /* ]] || { echo "Run root must be absolute" >&2; exit 2; }
: "${CIRCE_PARTITION:?}" "${CIRCE_TIME:?}"
PYTHON_BIN="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
DATASET_DIR="${DATASET_DIR:-/home/j/jinwoo/adajepa/data_full}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift}"
for path in "$REPO" "$RUN_ROOT" "$PYTHON_BIN" "$DATASET_DIR" "$SOURCE_MODEL_DIR"; do
  [[ "$path" = /* ]] || { echo "Absolute path required: $path" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable missing: $PYTHON_BIN" >&2; exit 2; }
for path in "$SOURCE_MODEL_DIR/hydra.yaml" "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  "$DATASET_DIR/pusht_noise/transition_color_manifest.json"; do
  [[ -f "$path" ]] || { echo "Required file missing: $path" >&2; exit 2; }
done
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || {
  echo "Commit tracked code changes before submitting" >&2; exit 2
}
command -v sbatch >/dev/null || { echo "sbatch unavailable; run on CIRCE" >&2; exit 2; }
sinfo -h -p "$CIRCE_PARTITION" -o '%P' | grep -q . || {
  echo "Partition $CIRCE_PARTITION not visible in sinfo" >&2; exit 2
}
if [[ ! -e "$RUN_ROOT" ]]; then
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models"
  printf 'repo=%s\ngit=%s\nsource=%s\ndataset=%s\n' \
    "$REPO" "$(git -C "$REPO" rev-parse HEAD)" "$SOURCE_MODEL_DIR" "$DATASET_DIR" \
    > "$RUN_ROOT/submission.txt"
  sha256sum "$SOURCE_MODEL_DIR/hydra.yaml" \
    "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
    "$DATASET_DIR/pusht_noise/transition_color_manifest.json" \
    > "$RUN_ROOT/input_sha256.txt"
  : > "$RUN_ROOT/job_ids.txt"
else
  [[ -f "$RUN_ROOT/submission.txt" && -f "$RUN_ROOT/input_sha256.txt" && -f "$RUN_ROOT/job_ids.txt" ]] || {
    echo "Existing run root is incomplete: $RUN_ROOT" >&2; exit 2
  }
  for entry in "repo=$REPO" "git=$(git -C "$REPO" rev-parse HEAD)" \
    "source=$SOURCE_MODEL_DIR" "dataset=$DATASET_DIR"; do
    grep -Fxq "$entry" "$RUN_ROOT/submission.txt" || {
      echo "Run root was created for different inputs: $entry" >&2; exit 2
    }
  done
  sha256sum -c "$RUN_ROOT/input_sha256.txt" >/dev/null || {
    echo "Source checkpoint or color manifest changed" >&2; exit 2
  }
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models"
exec 9>"$RUN_ROOT/submission.lock"
flock -n 9 || { echo "Another submitter is running in $RUN_ROOT" >&2; exit 2; }

options=(--parsable --nodes=1 --ntasks=1 --cpus-per-task="${CIRCE_CPUS:-8}"
  --mem="${CIRCE_MEM:-64G}" --gres="${CIRCE_GPU_GRES:-gpu:1}"
  --partition="$CIRCE_PARTITION" --time="$CIRCE_TIME" --chdir="$REPO" --requeue)
if [[ -n "${CIRCE_QOS:-}" ]]; then options+=(--qos="$CIRCE_QOS"); fi
if [[ -n "${CIRCE_ACCOUNT:-}" ]]; then options+=(--account="$CIRCE_ACCOUNT"); fi
for variant in B C; do
  if grep -Eq "^train_$variant=[0-9]+$" "$RUN_ROOT/job_ids.txt"; then
    echo "train_$variant already submitted; inspect $RUN_ROOT/job_ids.txt"
    continue
  fi
  if ! id="$(sbatch "${options[@]}" --job-name="film-$variant-3epoch" \
      --output="$RUN_ROOT/logs/train_$variant-%j.out" \
      --error="$RUN_ROOT/logs/train_$variant-%j.out" \
      --export=ALL,REPO_DIR="$REPO",PYTHON_BIN="$PYTHON_BIN",DATASET_DIR="$DATASET_DIR",SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR",RUN_ROOT="$RUN_ROOT" \
      "$REPO/scripts/slurm_film_bc_train.bash" "$variant")"; then
    echo "Failed to submit train_$variant; accepted IDs remain in $RUN_ROOT/job_ids.txt" >&2
    exit 1
  fi
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job ID for train_$variant: $id" >&2; exit 2; }
  printf 'train_%s=%s\n' "$variant" "$id" >> "$RUN_ROOT/job_ids.txt"
  echo "Submitted train_$variant=$id"
done
echo "Training jobs and logs: $RUN_ROOT"
