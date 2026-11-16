#!/usr/bin/env bash
# Run one matched P/J case on a GPU compute node. Submit via sbatch --export=ALL.
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO_DIR:?}" "${RUN_DIR:?}" "${DATASET_DIR:?}" "${SOURCE_MODEL_DIR:?}"
mode="${1:?Use P or J}"
case "$mode" in P|J) ;; *) echo "Mode must be P or J" >&2; exit 2 ;; esac
python_bin="${PYTHON_BIN:-/home/j/jinwoo/miniforge3/envs/ts/bin/python}"
cd "$REPO_DIR"
git diff --quiet && git diff --cached --quiet || {
  echo "Commit code changes before submitting a reproducible run" >&2
  exit 2
}
if [[ -n "${SUBMITTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$SUBMITTED_COMMIT" ]]; then
  echo "Checkout commit changed after submission" >&2
  exit 2
fi
test -f "$SOURCE_MODEL_DIR/hydra.yaml"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
test -f "$DATASET_DIR/pusht_noise/transition_color_manifest.json"
test -f "$DATASET_DIR/pusht_noise/transition_heldout_manifest.json"
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$RUN_DIR/hydra.yaml" || { echo "Cannot resume: hydra.yaml missing" >&2; exit 2; }
  test -f "$RUN_DIR/checkpoints/model_latest.pth" || {
    echo "Cannot resume: no complete checkpoint; use a new RUN_DIR" >&2
    exit 2
  }
  "$python_bin" - "$RUN_DIR" "$mode" "${TRAIN_SEED:-0}" "${TRAIN_EPOCHS:-3}" "${TRAIN_BATCH_SIZE:-32}" "$SOURCE_MODEL_DIR" <<'PY'
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
folder, mode, seed, epochs, batch, source = sys.argv[1:]
cfg = OmegaConf.load(str(Path(folder) / "hydra.yaml"))
expected = {"P": "predictor_only", "J": "joint"}[mode]
checks = {
    "training.gradient_mode": (cfg.training.gradient_mode, expected),
    "training.seed": (int(cfg.training.seed), int(seed)),
    "training.epochs": (int(cfg.training.epochs), int(epochs)),
    "training.batch_size": (int(cfg.training.batch_size), int(batch)),
    "training.initial_checkpoint": (str(cfg.training.initial_checkpoint),
                                    str(Path(source) / "checkpoints/model_latest.pth")),
}
for key, (actual, wanted) in checks.items():
    if actual != wanted:
        raise ValueError("Resume config mismatch {}: {} != {}".format(key, actual, wanted))
checkpoint = torch.load(Path(folder) / "checkpoints/model_latest.pth", map_location="cpu")
epoch = int(checkpoint["epoch"])
if not (0 <= epoch <= int(epochs)):
    raise ValueError("Invalid saved epoch {} for target {}".format(epoch, epochs))
if not (Path(folder) / "checkpoints" / "model_{}.pth".format(epoch)).is_file():
    raise ValueError("Latest checkpoint has no matching epoch checkpoint")
print("Resuming {} from completed epoch {} toward epoch {}".format(mode, epoch, epochs), flush=True)
PY
  printf 'job=%s commit=%s time=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$(date -Is)" >> "$RUN_DIR/resume_history.txt"
else
  test ! -e "$RUN_DIR/hydra.yaml" || {
    echo "RUN_DIR already has a run; set RESUME=1 after checking its checkpoint" >&2
    exit 2
  }
fi
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "GPU allocation missing"'
"$python_bin" - "$SOURCE_MODEL_DIR/hydra.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
expected = {
    "encoder": "models.encoder.resnet.SmallResNetGeM",
    "predictor": "models.vit.ViTPredictor",
}
for key, value in expected.items():
    if cfg[key]._target_ != value:
        raise ValueError("Released source {} is {}; expected {}".format(key, cfg[key]._target_, value))
for key, value in (("num_hist", 3), ("num_pred", 1), ("frameskip", 5)):
    if cfg[key] != value:
        raise ValueError("Released source {} is {}; expected {}".format(key, cfg[key], value))
PY
mkdir -p "$RUN_DIR"
"$python_bin" scripts/audit_gradient_color_data.py \
  --data-root "$DATASET_DIR/pusht_noise" --output "$RUN_DIR/data_audit.json"
if [[ "${RESUME:-0}" != 1 ]]; then
{
  printf 'git=%s\n' "$(git rev-parse HEAD)"
  printf 'mode=%s\nseed=%s\nepochs=%s\nbatch_size=%s\n' \
    "$mode" "${TRAIN_SEED:-0}" "${TRAIN_EPOCHS:-3}" "${TRAIN_BATCH_SIZE:-32}"
  printf 'source_model=%s\n' "$SOURCE_MODEL_DIR"
  "$python_bin" -c 'import sys,torch,hydra,accelerate; print("python="+sys.version.split()[0]); print("torch="+torch.__version__); print("hydra="+hydra.__version__); print("accelerate="+accelerate.__version__)'
  sha256sum "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
    "$DATASET_DIR/pusht_noise/transition_color_manifest.json" \
    "$DATASET_DIR/pusht_noise/transition_heldout_manifest.json"
} > "$RUN_DIR/provenance.txt"
fi
case "$mode" in P) experiment=gradient_p ;; J) experiment=gradient_j ;; esac
"$python_bin" train.py --config-name train \
  env=pusht_gradient encoder=scratch_resnet_gem predictor=vit \
  "+experiment=$experiment" \
  "training.initial_checkpoint=$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  "training.seed=${TRAIN_SEED:-0}" "training.epochs=${TRAIN_EPOCHS:-3}" \
  "training.batch_size=${TRAIN_BATCH_SIZE:-32}" \
  +wandb_project=adajepa_gradient_ablation \
  "+wandb_run_name=gradient-$mode-seed${TRAIN_SEED:-0}" \
  "hydra.run.dir=$RUN_DIR"
test -f "$RUN_DIR/checkpoints/model_${TRAIN_EPOCHS:-3}.pth"
sha256sum "$RUN_DIR/checkpoints/model_${TRAIN_EPOCHS:-3}.pth" > "$RUN_DIR/checkpoint_sha256.txt"
