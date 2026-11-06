# Handoff: transition-FiLM PushT training on an RTX A5500 (batch 16)

This document is for the Codex agent working in a **different SSH session**.
The user wants to start a controlled B-versus-C experiment on an NVIDIA RTX
A5500 with 24 GB of GPU memory, using a per-process batch size of **16**.
Inspect the target machine and its scheduler before launching long jobs.

## Experiment and existing work

- **A, reference:** the released vanilla AdaJEPA PushT checkpoint at
  `checkpoints/pusht_visual_shift` (no retraining for this comparison).
- **B, control:** initialize from the same released encoder, predictor,
  action encoder, and proprio encoder; freeze the visual encoder and train
  the original `vit` predictor for three epochs.
- **C, proposed method:** initialize from the same released checkpoint;
  freeze the visual encoder and train the `vit_transition` predictor with
  transition-conditioned FiLM for three epochs.
- B and C use the same physical PushT training/validation trajectories and
  deterministic `default`, `blue`, and `green` block-color views. The held-out
  `redBlock` condition is **evaluation only**. Blue is `#003D99` and green
  is `#006B3C`, as defined in `scripts/generate_transition_colors.py`.
- Use seed 0 for training. After training, compare A/B/C on the same PushT
  target file with frozen GD planning, seed 100, 50 targets, 30 MPC steps,
  separately for `default` and `redBlock`.

The original CIRCE B/C jobs **34347836** and **34347837** were submitted with
batch 32 from checkout
`/home/j/jinwoo/adajepa-transition-film-logging-fix`. They may still be
pending or running. **Do not pull, edit, or reuse that checkout or its run
root:** its Slurm payload checks that the code commit has not changed. This
A5500 batch-16 run must use its own checkout and output directories. Do not
cancel the existing jobs without an explicit user request.

## Branch and code

Use branch **`transition-film-logging-fix`** from
`https://github.com/jinu7513/adajepa.git`. It includes the epoch-metric JSON
serialization fix; the older `transition-conditioned-film` commit `612b0b3`
crashed at the first epoch's logging step before saving B/C checkpoints.
Clone into a new directory on the target machine. Git 1.8 on CIRCE does not
support `git -C`, so use `cd` followed by ordinary Git commands.

```bash
git clone -b transition-film-logging-fix https://github.com/jinu7513/adajepa.git adajepa-transition-film-a5500-b16
cd adajepa-transition-film-a5500-b16
git rev-parse --short HEAD
```

The checked-in `scripts/submit_film_bc_train.bash` and
`scripts/slurm_film_bc_train.bash` currently hardcode
`training.batch_size=32`. **Do not use those scripts as-is for batch 16.**
The training overrides below come directly from
`scripts/slurm_film_bc_train.bash` with the batch size changed to 16 and
distinct W&B names. If the target host has a scheduler, wrap these commands
in its GPU job mechanism; do not train on a login node. If it is a permitted
standalone GPU machine, run the commands there with a persistent session or
service so SSH disconnection does not terminate them.

## Data and checkpoint preflight

The dataset and released checkpoint are **not supplied by cloning Git**.
Locate or transfer the exact same inputs used on CIRCE:

- `SOURCE_MODEL_DIR/hydra.yaml` and
  `SOURCE_MODEL_DIR/checkpoints/model_latest.pth`; CIRCE source:
  `/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift`.
- `DATASET_DIR/pusht_noise/transition_color_manifest.json` and both the
  `train` and `val` splits. Each split needs `states.pth`, `rel_actions.pth`,
  `velocities.pth`, `seq_lengths.pkl`, and the generated MP4s under
  `colors/{default,blue,green}/obses/`. The loader also accepts `shapes.pkl`
  when present. CIRCE data parent: `/home/j/jinwoo/adajepa/data_full`.
- For later MPC evaluation, the same
  `data/pushobj_eval/val_T/plan_targets.pkl` used by A on CIRCE.

Set these variables to **real absolute paths on the target machine**. The
example values are CIRCE locations, not a claim that the other host has the
same filesystem.

```bash
export PYTHON_BIN=/absolute/path/to/python
export DATASET_DIR=/absolute/path/to/data_full
export SOURCE_MODEL_DIR=/absolute/path/to/checkpoints/pusht_visual_shift
export TARGETS=/absolute/path/to/data/pushobj_eval/val_T/plan_targets.pkl

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
test -f "$SOURCE_MODEL_DIR/hydra.yaml"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
test -f "$DATASET_DIR/pusht_noise/transition_color_manifest.json"
test -f "$DATASET_DIR/pusht_noise/train/colors/blue/obses/episode_000.mp4"
test -f "$DATASET_DIR/pusht_noise/val/colors/green/obses/episode_000.mp4"
"$PYTHON_BIN" -m pytest tests/test_train_epoch_logging.py tests/test_transition_film.py -q
```

Check each command's exit status; `test -f` prints nothing on success. Compare
the source checkpoint and color-manifest SHA256 hashes with CIRCE before
claiming the same inputs. Verify that the installed CUDA/PyTorch stack accepts
the configured `training.mixed_precision=bf16` on this GPU.

## Batch-16 training commands

Use a new absolute run directory and **one GPU process**. B and C should use
the same batch, seed, data, number of epochs, and worker setting. Four data
loader workers are a conservative starting point on an unknown host; if host
RAM and I/O permit, both runs can use eight. Worker count changes throughput
but not the deterministic color assigned to a particular trajectory window.

```bash
RUN_ROOT="$PWD/checkpoints/transition_film_a5500_b16_seed0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/B" "$RUN_ROOT/C" "$RUN_ROOT/logs"
printf 'git=%s\nsource=%s\ndataset=%s\nbatch=16\nseed=0\n' "$(git rev-parse HEAD)" "$SOURCE_MODEL_DIR" "$DATASET_DIR" > "$RUN_ROOT/experiment.txt"
sha256sum "$SOURCE_MODEL_DIR/hydra.yaml" "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" "$DATASET_DIR/pusht_noise/transition_color_manifest.json" > "$RUN_ROOT/input_sha256.txt"
```

**B: original predictor.** Execute in an allocated GPU job or permitted GPU
session, with the repository as the working directory:

```bash
PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py --config-name train \
  env=pusht_transition predictor=vit \
  model.train_encoder=false model.train_predictor=true model.train_decoder=false \
  has_decoder=false training.epochs=3 training.batch_size=16 \
  training.seed=0 training.save_every_x_iterations=0 training.save_every_x_epoch=1 \
  env.num_workers=4 \
  "+training.initial_checkpoint=$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  +training.save_frozen_encoder=true \
  +wandb_project=adajepa_transition_film \
  +wandb_run_name=transition-film-B-a5500-b16-seed0 \
  "hydra.run.dir=$RUN_ROOT/B"
```

**C: transition-FiLM predictor.** Run sequentially on a single A5500, or in
a separate GPU allocation if another GPU is available:

```bash
PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py --config-name train \
  env=pusht_transition predictor=vit_transition \
  model.train_encoder=false model.train_predictor=true model.train_decoder=false \
  has_decoder=false training.epochs=3 training.batch_size=16 \
  training.seed=0 training.save_every_x_iterations=0 training.save_every_x_epoch=1 \
  env.num_workers=4 \
  "+training.initial_checkpoint=$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" \
  +training.save_frozen_encoder=true \
  +wandb_project=adajepa_transition_film \
  +wandb_run_name=transition-film-C-a5500-b16-seed0 \
  "hydra.run.dir=$RUN_ROOT/C"
```

The config already selects BF16. Do not set `training.mixed_precision` unless
the environment requires a documented change. To keep the experiment
reviewable, record any such change and apply it to both B and C. Do not run
B and C simultaneously on the same 24 GB card.

## Progress, resume, and evaluation

`train.py` writes each run's `hydra.yaml`, `epoch_metrics.jsonl`, W&B run,
and `checkpoints/model_{1,2,3}.pth` plus `model_latest.pth`. Check:

```bash
ls -lh "$RUN_ROOT/B/checkpoints" "$RUN_ROOT/C/checkpoints"
tail -n 3 "$RUN_ROOT/B/epoch_metrics.jsonl"
tail -n 3 "$RUN_ROOT/C/epoch_metrics.jsonl"
```

Only an epoch checkpoint proves that epoch finished. If interrupted, inspect
the `epoch` in `model_latest.pth` and its saved `hydra.yaml`; the loader
resumes from the same `hydra.run.dir`. Set `training.epochs` to the **number of
remaining epochs** (`3 - saved_epoch`), not to 3 again. Preserve both the
checkpoint and its optimizer states. Do not try to resume a half-completed
epoch. Ask the Codex agent to verify the saved epoch and construct a new job
for exactly the remaining epochs before executing it.

After both `model_3.pth` files exist, run matched PushT MPC evaluations using
the argument pattern in `scripts/slurm_transition_film_stage.bash`:

- `plan.py --config-name adajepa_plan_gd_pushobj.yaml`
- `ckpt_base_path=<B-or-C-run-directory>`, `model_name=<that-directory-name>`,
  `model_epoch=3`, `eval_data_path=$TARGETS`
- `seed=100 n_evals=50 planner.max_iter=30 ood_corruption=null`
- `planner.adapt.lr=0 planner.adapt.steps=0`
  `planner.adapt.finetune_encoder=false`
- `env_kwargs_override=null` for default, or
  `env_kwargs_override={color:Red}` for redBlock
- `+wandb_logging=true +wandb_project=adajepa_transition_film_mpc` and
  a unique run name containing variant, batch 16, condition, and seed.

Use an **absolute** model directory because Hydra changes into the output
directory. Give each MPC condition its own `hydra.run.dir`. Verify that its
`logs.json` contains `paper/success_rate_pct` before calling the evaluation
complete. Keep A's released-checkpoint result alongside B and C, and report
success rate (%) versus MPC step and final success on the same 50 targets.

**Comparison caveat:** B16 versus C16 is a controlled comparison. The
existing CIRCE B32/C32 jobs have a different effective batch and therefore
fewer optimizer updates per epoch; label those separately. This repository
has no gradient-accumulation option in the current training path, so do not
claim the batch-16 and batch-32 runs have identical optimization budgets.
Preserve W&B run IDs, commit hashes, source/dataset hashes, checkpoints, and
actual GPU memory/time measurements in the final report.
