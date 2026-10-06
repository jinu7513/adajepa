# Track A: scratch robust visual encoder

Detailed method, equations, and experiment protocol are available in standalone
[English](overleaf/tracka_en.tex) and [Korean](overleaf/tracka_ko.tex) LaTeX documents.
See the [Overleaf instructions](overleaf/README.md) for compilation and scope.

Base: `66c5f18d00cc627ae35b9b183367f7e510972106` (`Flush W&B metrics on planning exit`).
Branch: `robust-encoder-scratch`. This experiment does not implement Flow-JEPA or
change MPC/planning algorithms. The shared corruption module retains old names and
their output semantics (`blur` at sigma zero is now explicitly identity).

## Inspection before implementation

- DINOv2 ViT-S/14 emits normalized patch tokens. The existing VWorldModel resizes
  a 224px image to 196px for DINO, yielding `[B,196,384]` visual tokens. With the
  usual 10D proprio and 10D action embeddings concatenated, the predictor sees 404D.
- The old `models/encoder/vit.py` imports the predictor Transformer, whose mask
  depends on module globals and is allocated on CUDA. Track A implements a new
  spatial bidirectional encoder instead.
- PushT state is agent xy, block xy, block angle, agent vx/vy. Block velocity,
  angular velocity, contacts and solver state are absent. `_set_state()` advances
  physics by 0.01 seconds; static rerendering is not full dynamics replay.
- The renderer supports block (`color`), agent and goal colors. Shapes receive
  colors at setup; rebuilding/resetting is required. Background is white.
- Dataset sources contain `states.pth`, `velocities.pth`, `seq_lengths.pkl`, optional
  `shapes.pkl`, and episode MP4s. Generation reads states directly and rerenders all
  four views rather than mixing compressed source video with newly rendered views.
- Existing image normalization maps uint8 to `[0,1]` then `[-1,1]`.
- Hydra configs live in `conf/`. Track A has its own config and local entrypoints;
  it does not inherit the existing training config's cluster-specific SLURM launcher.
- Existing corruption names are blur, snp1, snp5, dark. Gaussian and canonical
  salt_pepper are new names implemented in the same module.

## Environment and paths

Use the existing CIRCE `ts` environment. The implementation is compatible with
Python 3.9 syntax and uses PyTorch, NumPy, Pillow, Hydra/OmegaConf, W&B and the
existing PushT dependencies (pygame, pymunk 6.x, gym, shapely, OpenCV, scikit-image).
Probe plots use matplotlib; tests use pytest. No extra pretrained model is needed
unless `evaluation.encoder=dino` is requested.

All relative experiment data/output/checkpoint paths resolve against the repository
root, not Hydra's output directory. Output directories are unique by default.
Explicit nonempty training/evaluation directories are rejected to avoid mixing runs.

The commands below are Bash commands for the SSH server, from the repository root.
Replace the source path with the actual dataset directory containing `train/states.pth`.
Do not point it at `val_T/plan_targets.pkl`.

## 1. Preview and generate the four-view dataset

```bash
SOURCE=/absolute/path/to/pushT_dataset
PAIRS="$PWD/data/tracka_pairs"

python generate_pusht_pairs.py --config-name tracka \
  dataset.source_path="$SOURCE" dataset.output_path="$PAIRS" \
  generation.preview_only=true
```

Inspect `four_view_contact_sheet.png`, `corruption_contact_sheet.png`, and
`generation_report.json` in `$PAIRS`. The latter records the resolved candidate
ranges and reset errors. Corruption contact sheets show low/mid/high strengths and
a 50% spatial patch mask. Default ranges are conservative starting candidates, not
validated full-dataset thresholds. After inspecting representative real-data previews:

```bash
python generate_pusht_pairs.py --config-name tracka \
  dataset.source_path="$SOURCE" dataset.output_path="$PAIRS" \
  generation.preview_only=false generation.ranges_reviewed=true
```

For a small dataset smoke run, additionally set:
`dataset.max_states_per_trajectory=3 dataset.frame_stride=1`.
At least three source trajectories are necessary. Keep enough states for every
split; generation records rejected states and does not loosen tolerances silently.
`source_splits=[train]` is the default. Train/validation/test assignment is locked
in the generated manifest; existing source train/val pools are never silently merged.

## 2. Train the baselines and robust objective

W&B defaults to project `adajepa_trackA`, online mode. Authenticate on CIRCE using
`wandb login` if needed. Training records metrics while it is running. This task
contains no MPC evaluation, so there is no `paper/success_rate_pct` metric.

```bash
python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=clean_mae \
  training.seed=0 logging.group=trackA-seed0

python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=render_only \
  training.seed=0 logging.group=trackA-seed0

python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=corruption_only \
  training.seed=0 logging.group=trackA-seed0

python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=robust_alternating \
  training.seed=0 logging.group=trackA-seed0
```

These are full-training commands for later use, not commands run during implementation.
All use the same default update budget. `robust_alternating` requires an even budget.
For a smoke run use `training.total_optimizer_updates=2 training.batch_size=2`.
Use `training.output_dir=...` to select a fresh directory, or accept the printed
unique path. Run names identify objective, conditioning, mask ratio and seed.
`logging.mode=offline` stores W&B data for later sync; `disabled` uses local records.

Optional features (change one at a time for ablations):

```bash
python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=robust_alternating \
  decoder.conditioning=cross_attention

python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=robust_alternating \
  model.use_cls=true decoder.use_cls_global_decoder=true loss.lambda_cls_global=0.1
```

Objective-specific masking ablation (one shared encoder, not separate encoders):

```bash
python train_encoder.py --config-name tracka \
  dataset.output_path=data/pairs training.objective=robust_alternating \
  training.total_optimizer_updates=1000 training.batch_size=16 training.seed=0 \
  model.use_cls=true decoder.use_cls_global_decoder=true loss.lambda_cls_global=0.1 \
  mask.ratio=0.5 mask.render_ratio=0.5 mask.corruption_ratio=0.25 \
  logging.name=trackA-cls-render50-corr25-1k-seed0 logging.fallback_to_local=false
```

`mask.ratio` remains the fallback and the clean-MAE ratio; optional `render_ratio`
and `corruption_ratio` override their respective update modes. Both ratios must
remain strictly between 0 and 1 because MAE loss uses masked patches. The offline
corruption view mixes blur, Gaussian, and salt-pepper; `corruption_ratio` applies to
all three, not blur alone. Each update logs `mask/ratio` to W&B and local metrics.
A different mask schedule cannot be resumed from the checkpoint. Probe with the
same `evaluation.max_states_per_split` and seed as previous runs, using the new
run's printed checkpoint path.

### Optional label-free variance ablation

The optional `loss.lambda_variance` adds a VICReg-style **variance term only**;
it is not a full VICReg implementation and does not use state, position, or angle
labels. It applies a per-feature standard-deviation floor across different clean
images in a batch:

`L_var = mean_j max(0, variance_target_std - sqrt(Var_batch(z_j) + 1e-4))`.

Here `z` comes from an **unmasked** clean-image encoder pass, using patch-token
mean, CLS, or both as selected by `loss.variance_feature`. An unmasked pass is
important: otherwise independently sampled masks could supply between-sample
variance even if the model ignored the underlying scene. The same loss is added
on render and corruption updates. It requires training batch size at least two
and adds one full-image encoder forward pass per update. The default weight is
zero, so existing training behavior and checkpoints remain compatible.

For a CLS experiment matched to the render50/corruption25 pilot, first run a
two-update smoke test, then change `training.total_optimizer_updates` to 1000
for a new, separate run:

```bash
python train_encoder.py --config-name tracka \
  dataset.output_path=data/pairs training.objective=robust_alternating \
  training.total_optimizer_updates=2 training.batch_size=16 training.seed=0 \
  model.use_cls=true decoder.use_cls_global_decoder=true loss.lambda_cls_global=0.1 \
  mask.ratio=0.5 mask.render_ratio=0.5 mask.corruption_ratio=0.25 \
  loss.lambda_variance=0.1 loss.variance_target_std=0.1 loss.variance_feature=both \
  logging.name=trackA-cls-render50-corr25-var-smoke logging.fallback_to_local=false
```

The `0.1` standard-deviation floor and `0.1` loss weight are pilot settings, not
published VICReg defaults. W&B/local metrics report `loss/variance`,
`loss/weighted_variance`, per-feature variance losses, and unmasked clean feature
standard deviations. Compare against the zero-weight model with the same dataset,
seed, masking schedule, update count, and probe settings. A higher feature standard
deviation alone is not evidence of improved PushT control or robustness.

### SIGReg instead of the variance floor

`loss.lambda_sigreg` selects a label-free SIGReg ablation. It is **mutually
exclusive** with `loss.lambda_variance`: set the latter to zero. The default
settings apply SIGReg separately to the unmasked full-clean patch-token mean
and CLS token, then average the two losses (`loss.sigreg_feature=both`). CLS
requires `model.use_cls=true`. This keeps the reconstruction, invariance,
global-CLS, mask, and dataset paths unchanged; it is not a full LeJEPA model.

For each feature vector `z`, the implementation divides by the fixed
`loss.sigreg_target_std` (default `0.1`), samples 256 random unit directions,
and compares the empirical characteristic function of each projection to a
unit Gaussian at 17 knots on `[0, 3]`. It uses the Gaussian-windowed
positive-half quadrature and multiplies the mean statistic by batch size,
following the [official minimal implementation](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).
For selected features `F` (patch mean, CLS, or both), the added loss is
`lambda_sigreg / |F| * sum_{f in F} SIGReg(z_f / sigreg_target_std)`.
The statistic compares the cosine/sine batch means of projected features to
the standard Gaussian characteristic function `exp(-t^2/2)`.
Equivalently, the target for *raw* features is an isotropic Gaussian with
standard deviation `0.1`. Unlike a one-sided variance floor, this penalizes
distribution mismatch in either direction; the same target for patch mean and
CLS may therefore be restrictive. This is an experiment, not a performance
guarantee. Training batch size must be at least two; the rare validation batch
of one contributes zero to SIGReg.

From the repository root on CIRCE, run a short new experiment first:

```bash
python train_encoder.py --config-name tracka \
  dataset.output_path=data/pairs training.objective=robust_alternating \
  training.total_optimizer_updates=200 training.batch_size=16 training.seed=0 \
  model.use_cls=true decoder.use_cls_global_decoder=true loss.lambda_cls_global=0.1 \
  mask.ratio=0.5 mask.render_ratio=0.5 mask.corruption_ratio=0.25 \
  loss.lambda_variance=0 loss.lambda_sigreg=0.001 \
  loss.sigreg_target_std=0.1 loss.sigreg_feature=both \
  logging.name=trackA-sigreg-both-200-seed0 logging.fallback_to_local=false
```

The new W&B/local keys are `loss/sigreg`, `loss/weighted_sigreg`,
`loss/sigreg_patch_mean`, and `loss/sigreg_cls`; the existing full-clean
feature standard-deviation keys remain. Compare the weighted SIGReg magnitude
with reconstruction and invariance losses during this pilot; tune the weight
only after inspecting these values. `0.001` is specific to this batch-scaled,
fixed-target implementation and is not interchangeable with the old variance
weight `0.1`. For a fair ablation, start a **new** run from scratch rather than
resuming a variance checkpoint; the resume guard rejects loss changes. Then
run the same clean-trained frozen probes before deciding on expensive
predictor training.

## 3. Resume and recover logging

```bash
python train_encoder.py --config-name tracka \
  dataset.output_path="$PAIRS" training.objective=robust_alternating \
  training.resume=/absolute/path/to/run/checkpoint_latest.pt \
  training.total_optimizer_updates=20000

python recover_tracka_wandb.py /absolute/path/to/run
```

For resume, repeat any nondefault model/loss/optimizer/seed options from the original
run. Configuration and dataset mismatches are rejected. Online resume uses the exact
stored W&B run ID with `resume=must`. Offline resume starts a new offline W&B segment;
local metrics retain their global steps. A checkpoint older than locally recorded
steps is rejected rather than silently duplicating steps. Keep `save_every` small if
frequent interruptions are expected. Recovery uploads local metrics/tables into a
clearly named NEW recovery run and leaves the old run intact.

Local files include flushed `metrics.jsonl`, config and metadata JSON, W&B identity,
`run_status.json`, periodic reconstruction PNGs, and the latest atomic checkpoint.
Exceptions produce `exception.txt` and an unsuccessful W&B finish. SIGKILL/power
failure cannot execute finalization; prior flushed local records remain recoverable.
If W&B is unavailable, a warning is emitted and local logging continues by default;
set `logging.fallback_to_local=false` to require successful W&B initialization.

Training graph X axis: `train/global_step`. Alternating updates use one monotonically
increasing counter; both objectives are sampled at the configured logging cadence.
Probe condition metrics have their own keys and full result tables.
W&B also receives a render-condition bar chart and per-corruption strength curves.
Artifact caches/staging default to the run directory (explicit `WANDB_CACHE_DIR`
and `WANDB_DATA_DIR` environment overrides are respected), avoiding dependence on
writable home-directory caches on shared cluster nodes.
Run metadata also includes a hash of the Track A source files, so uncommitted local
code is distinguishable even when the git commit still points at the reviewed base.

## 4. Frozen probes and references

```bash
python eval_encoder_probes.py --config-name tracka \
  dataset.output_path="$PAIRS" \
  evaluation.encoder=checkpoint \
  evaluation.checkpoint=/absolute/path/to/run/checkpoint_latest.pt

# Evaluate the same checkpoint using its CLS token instead of patch-token mean.
# The checkpoint must have been trained with model.use_cls=true.
python eval_encoder_probes.py --config-name tracka \
  dataset.output_path="$PAIRS" \
  evaluation.encoder=checkpoint \
  evaluation.checkpoint=/absolute/path/to/run/checkpoint_latest.pt \
  evaluation.feature=cls logging.name=trackA-probe-cls

python eval_encoder_probes.py --config-name tracka \
  dataset.output_path="$PAIRS" evaluation.encoder=random

python eval_encoder_probes.py --config-name tracka \
  dataset.output_path="$PAIRS" evaluation.encoder=dino
```

Repeat the checkpoint command for each trained objective/seed. Physical probes use
only clean train features and canonical-render-state labels; regularization is selected
on clean validation data. The SAME fitted probe evaluates all test conditions. Encoder
weights remain frozen. RGB regression probes use variable-color A/B views. Corruption
strength regressors are fit separately for each type because strengths have different
units. Insufficient per-type data is explicitly reported.

`evaluation.feature=patch_mean` is the default and preserves historical probe
results. `evaluation.feature=cls` uses the full-image CLS token from a scratch
checkpoint (or a random scratch encoder with `model.use_cls=true`); DINO in this
repository exposes patch tokens only, so CLS selection is rejected. Each CLS run
fits a new clean-only linear probe with the same split, seed, and sample cap, then
uses that fixed probe across all visual shifts. The run metadata records the selected
feature. Compare the resulting `results.csv` against a separate patch-mean run;
CLS variance alone is not a measure of state information or robustness.

Results: `results.csv/json`, `diagnostics.json`, `physical_probe.npz`, RGB probe NPZs,
`evaluation_views.json` (actual colors), and `robustness.png`. Metrics are reported for
full, early, middle and late test segments. Red interventions and fixed severity grids
are generated separately from the exactly-four-view training dataset.

The DINO reference preserves the existing wrapper's resize/normalization behavior:
196px input, patch14, 196 tokens at default configuration. Scratch uses 224px input,
patch16, 196 tokens. This resolution difference is documented in run metadata. DINO
may need its pinned torch.hub weights downloaded; it is never retrained.

```bash
python aggregate_tracka_results.py \
  /absolute/path/to/probe_clean /absolute/path/to/probe_robust \
  --output-dir "$PWD/tracka_outputs/comparison"
```

Aggregation checks dataset/protocol compatibility, exports combined CSV and plots
mean plus sample SD over distinct training seeds. Use one checkpoint per objective/seed.
Different model/decoder/loss/mask/update configurations remain separate groups;
`experiment_configs.json` maps each figure label back to its configuration.
This figure reports probe error, not policy success or the paper's MPC curve.

## Data and nuisance schema

`manifest.json` records schema version, exact source pool, split trajectory IDs,
selected timesteps, all sampling seeds/caps/config, metadata hash, git SHA/dirty status,
sample counts and maximum reset errors. `samples.jsonl` has one row per accepted state:

- stable source ID `source_split/episode_index`, timestep, valid length and segment;
- original source state, post-clean-reset canonical state, verification errors;
- four relative PNG paths; RGB arrays ordered block/agent/goal for clean/A/B;
- continuous sampled HSV/RGB and actual quantized RGB for A/B;
- corruption type, strength, independent realization seed, render and dataset seeds.

Models receive images and target RGB/255 only. Source/canonical states and corruption
metadata are absent from training batches. `eta_clean` is explicitly supplied for
clean MAE and denoising targets. Colors are sampled independently of physical state
using separate RNG streams. Distances use rendered RGB/255 Euclidean distance;
per-object inter-view rejection is stronger than merely requiring distinct tuples.

Gaussian sigma is in uint8 intensity units; salt_pepper strength is the total Bernoulli
selected-pixel probability with exclusive salt/pepper outcomes; blur sigma is in pixels
and kernel size is `max(3, 2*round(3*sigma)+1)`. Legacy snp1/snp5 retain their original
separate sampling with replacement for salt and pepper, so their parameter is not the
canonical total probability. Different pixels can already be white/black: probability
describes selected pixels, not necessarily observed changed pixels.

## Shapes, losses and checkpoint contract

At defaults: input `[B,3,224,224]`; patch pixels `[B,196,768]`; embedded tokens
`[B,196,384]`; shared-mask visible tokens `[B,98,384]`; decoder projection
`[B,98,192]`; restored decoder tokens `[B,196,192]`; prediction `[B,196,768]`.
Encoder positional embeddings follow original `ids_keep` indices. Decoder restores
the raster order with `ids_restore` before adding its positional embeddings.

Clean: masked pixel MSE only. Render: average four directed clean/A/B masked MSEs,
plus lambda times average of clean-A and clean-B visible-token MSEs. Corruption:
masked clean-target MSE from corrupted inputs, plus lambda times clean-corrupt
visible-token MSE. Gradients flow into both sides of invariance terms; there is no
teacher, stop-gradient, GRL or state supervision. Reconstruction remains active.

Additive decoder: shared RGB MLP + learned object-type embeddings, ordered concat,
linear projection, added to every decoder patch token. Cross-attention: three typed
nuisance tokens are keys/values, spatial tokens are queries. One decoder is shared
across all objectives. Optional CLS global predictions use a projected CLS broadcast
over positional queries and the same decoder blocks/head; their full-patch MSE is
weighted by `lambda_cls_global`. `forward()` always returns only patch tokens.
With norm_pix_loss enabled, targets are standardized per patch; RGB reconstruction
previews are replaced by blank predictions because target statistics cannot be
recovered from the normalized prediction alone.

Checkpoint dictionaries contain encoder/decoder/optimizer state_dicts, epoch (sample
draw count divided by train states), update/cycle, next objective, config, manifest hash,
schema, git metadata, RNG states, W&B ID, sample counter and output path. Training uses
float32 AdamW with constant LR, so no AMP scaler or scheduler state is applicable.
Optional CLS/global parameters are included in encoder/decoder state_dicts. Only load
trusted checkpoints: saved RNG structures require `weights_only=False`.

## Validation and limitations

Implementation map:

| File | Responsibility |
| --- | --- |
| `conf/tracka.yaml` | All generation, model, training, evaluation and W&B options |
| `tracka/data.py` | Static paired rendering, manifests, integrity checks and loader |
| `tracka/model.py` | Bidirectional ViT, mask/restore, decoder, objective losses |
| `tracka/train.py` | Four training modes, validation, resume and checkpoints |
| `tracka/logging.py` | Durable local logging and W&B lifecycle |
| `tracka/probes.py` | Frozen reference encoders, probes, visual-shift evaluation |
| `tracka/common.py` | Paths, hashes, RNG and checkpoint utilities |
| `generate_pusht_pairs.py` | Dataset generation Hydra entrypoint |
| `train_encoder.py` | Training Hydra entrypoint |
| `eval_encoder_probes.py` | Evaluation Hydra entrypoint |
| `recover_tracka_wandb.py` | Local metrics/tables recovery to a new W&B run |
| `aggregate_tracka_results.py` | Cross-run CSV and comparison figure |
| `planning/image_corruption.py` | Shared canonical/legacy corruption registry |
| `tests/test_tracka.py` | Smoke/regression tests |

```bash
python -m pytest tests/test_tracka.py -q
```

Tests cover real PushT static rerendering, wrapped angles, dataset integrity, color
constraints, corruption/legacy semantics, mask restoration, bidirectional influence,
all objective/conditioning modes, invariance gradients, optional CLS, wrapper compatibility,
resume equivalence, linear probes, end-to-end probe reporting and real W&B offline files.
Online server delivery requires user credentials and network access and must be checked
on CIRCE. Synthetic smoke trajectories do not establish real-data performance.

Source reset tolerances can reject fast-moving/contact states. Inspect rejection rates
and per-field maxima by source/segment before training; do not simply widen them until
all samples pass. Canonical labels avoid systematic source/render label mismatch.
Static single-image velocity is not generally identifiable: velocity probes are auxiliary
diagnostics and must not be the sole criterion for encoder quality. Equal optimizer
updates are not equal compute: render updates process three encoded views and four
directed decoder passes, corruption two views and one decoder pass, clean one each.
Latent norm/distance alone cannot establish absence of collapse; also examine variance
and state probes. Fixed offline corruptions are repeatable but do not expose new noise
every epoch. Training colors always change all objects; single-object and combined
color+corruption generalization are not guaranteed. Optional OOD color ranges need to
be kept disjoint from any user-modified training ranges.

Recommended next experiment (not automatically run): clean_mae vs robust_alternating,
identical dataset and update budget, additive conditioning, CLS disabled, three seeds,
then evaluate the same clean-trained probes on default and redBlock/blur. Downstream
planning integration is future work: matching token shapes does not make a scratch
encoder a drop-in semantic replacement for a DINO-trained predictor checkpoint.
