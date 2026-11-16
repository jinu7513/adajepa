#!/usr/bin/env bash
# One independently namespaced A or B arm on a GPU compute node.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${REPO_DIR:?}" "${OUT_ROOT:?}" "${DATASET_DIR:?}" "${SOURCE_MODEL_DIR:?}"
track="${1:?cross_color_jepa or mi_jepa}"
arm="${2:?arm name}"
case "$track/$arm" in
  cross_color_jepa/same_fixed|cross_color_jepa/same_window_random|cross_color_jepa/cross_fixed|cross_color_jepa/cross_window_random) env_name=pusht_cross_color ;;
  mi_jepa/default_jepa_baseline|mi_jepa/default_jepa_plus_inverse) env_name=pusht_mi ;;
  *) echo "Unknown independent track/arm: $track/$arm" >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
git diff --quiet && git diff --cached --quiet || { echo "Commit code before training" >&2; exit 2; }
if [[ -n "${SUBMITTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$SUBMITTED_COMMIT" ]]; then
  echo "Submitted commit differs from checkout" >&2; exit 2
fi
run_dir="$OUT_ROOT/$track/models/$arm"
mkdir -p "$run_dir"
test -f "$SOURCE_MODEL_DIR/hydra.yaml"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
test -f "$DATASET_DIR/pusht_noise/train/states.pth"
"$python_bin" - "$SOURCE_MODEL_DIR/hydra.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
expected = {"num_hist": 3, "num_pred": 1, "frameskip": 5}
for key, value in expected.items():
    if cfg[key] != value:
        raise ValueError("Source {}={} expected {}".format(key, cfg[key], value))
if cfg.encoder._target_ != "models.encoder.resnet.SmallResNetGeM":
    raise ValueError("Source checkpoint is not the released PushT SmallResNetGeM model")
PY
if [[ "$track" == cross_color_jepa ]]; then
  test -f "$DATASET_DIR/pusht_noise/transition_color_manifest.json"
  test -f "$DATASET_DIR/pusht_noise/transition_heldout_manifest.json"
fi
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "GPU allocation missing"'
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$run_dir/hydra.yaml" && test -f "$run_dir/checkpoints/model_latest.pth" || {
    echo "No completed checkpoint to resume in $run_dir" >&2; exit 2;
  }
  "$python_bin" - "$run_dir" "$track" "$arm" "${TRAIN_SEED:-0}" "${TRAIN_EPOCHS:-3}" "${TRAIN_BATCH_SIZE:-32}" <<'PY'
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
root, track, arm, seed, epochs, batch = sys.argv[1:]
cfg = OmegaConf.load(str(Path(root) / "hydra.yaml"))
expected = {"research_track": track, "seed": int(seed), "epochs": int(epochs),
            "batch_size": int(batch)}
for key, wanted in expected.items():
    actual = cfg.training[key]
    if actual != wanted:
        raise ValueError("Resume mismatch {}: {} != {}".format(key, actual, wanted))
if track == "cross_color_jepa" and cfg.training.color_pair_mode != arm:
    raise ValueError("Resume color-pair mode differs from submitted arm")
ckpt = torch.load(Path(root) / "checkpoints/model_latest.pth", map_location="cpu")
epoch = int(ckpt["epoch"])
if not (0 <= epoch <= int(epochs)):
    raise ValueError("Invalid saved epoch")
print("Resuming {}/{} from epoch {} toward {}".format(track, arm, epoch, epochs), flush=True)
PY
  printf 'job=%s commit=%s time=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$(date -Is)" >> "$run_dir/resume_history.txt"
else
  test ! -f "$run_dir/hydra.yaml" || { echo "Run already exists; use RESUME=1" >&2; exit 2; }
  {
    printf 'track=%s\narm=%s\ngit=%s\nseed=%s\nepochs=%s\nbatch_size=%s\n' \
      "$track" "$arm" "$(git rev-parse HEAD)" "${TRAIN_SEED:-0}" "${TRAIN_EPOCHS:-3}" "${TRAIN_BATCH_SIZE:-32}"
    "$python_bin" -c 'import sys,torch,hydra,accelerate; print("python="+sys.version.split()[0]); print("torch="+torch.__version__); print("hydra="+hydra.__version__); print("accelerate="+accelerate.__version__)'
    sha256sum "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
  } > "$run_dir/provenance.txt"
fi
"$python_bin" train.py --config-name train \
  "env=$env_name" encoder=scratch_resnet_gem predictor=vit \
  "+experiment/$track=$arm" \
  "training.initial_checkpoint=$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  "training.seed=${TRAIN_SEED:-0}" "training.epochs=${TRAIN_EPOCHS:-3}" \
  "training.batch_size=${TRAIN_BATCH_SIZE:-32}" \
  "+wandb_project=adajepa_$track" \
  "+wandb_run_name=$track-$arm-seed${TRAIN_SEED:-0}" \
  "hydra.run.dir=$run_dir"
test -f "$run_dir/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
sha256sum "$run_dir/checkpoints/model_${TRAIN_EPOCHS:-3}.pth" > "$run_dir/checkpoint_sha256.txt"
