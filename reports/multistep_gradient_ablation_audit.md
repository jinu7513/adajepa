# AdaJEPA action-conditioned gradient-flow audit

## What the current implementation actually trains

The relevant path is `datasets/pusht_dset.py` -> `TrajSlicerDataset` ->
`Trainer.train` in `train.py` -> `VWorldModel.forward` in
`models/visual_world_model.py` -> the causal `ViTPredictor` in `models/vit.py`.
The released PushT visual checkpoint uses `SmallResNetGeM` with one 384-D visual
token; the predictor also receives proprioception and action embeddings. No
separate target encoder or EMA teacher appears in this training path. The same
encoder produces source and target latents; `stop_grad=true` detaches the target
tensor before MSE. Its target values move as J's encoder weights change, even
though target-branch gradients are blocked. In P the encoder weights and its
BatchNorm statistics remain fixed.

`conf/train.yaml` uses `num_hist=3`, `num_pred=1`, and `frameskip=5` by default.
The slicer loads four observation frames at indices `s, s+5, s+10, s+15`
and four corresponding 5-action blocks. The model inputs the first three
observation/action latents and compares three causal predictor output slots to
the following three observation latents. `models/vit.py` applies a lower
triangular attention mask, so output slot 0 cannot access source slots 1 or 2.
The last action block and its latent target action dimensions are excluded from
the training loss. MSE averages over batch, three output slots, token(s), and
the visual/proprio feature dimensions, excluding action dimensions.

Thus the default loss is **three teacher-forced, one-step predictions**, not an
autoregressive `H=3` future rollout loss. `VWorldModel.rollout` does perform
autoregressive multi-step prediction for evaluation and planning, but the
training objective does not optimize a separate loss at each rollout horizon.
`num_pred>1` is not a verified multi-step training switch. This phase keeps
`num_pred=1` and does not claim to introduce a novel multi-step loss.

Let `z_i = concat(E(o_{s+5i}), P(p_{s+5i}), A(a_{s+5i:s+5i+4}))`, for `i=0,1,2,3`.
The implemented predictive objective for this configuration is

```
L = MSE( F_phi(z_0, z_1, z_2)[:, :, visual+proprio],
         stopgrad([z_1, z_2, z_3])[:, :, visual+proprio] )
```

where the causal mask limits each output slot to current and earlier input
slots. In both P and J, the predictor, action embedding and proprio embedding
are trainable. P freezes the visual encoder. J trains the full visual encoder.
Both keep the encoder in evaluation mode so BatchNorm running statistics and
dropout mode do not become a second experimental variable; evaluation mode
does **not** block gradients in J. No new architecture, decoder, FiLM,
curvature loss, VCReg, or target encoder is added.

## Controlled setup and limits

Both conditions start from the same released checkpoint, use the same
`default/blue/green` paired-color videos, deterministic window-color rule,
physical train/val split, seed, batch size, epochs, action normalization,
three-history/one-prediction frame window, and validation color. P/J differ
in encoder parameter gradients and the resulting learned weights. Because J
updates more parameters and retains a moving shared target representation,
the two modes are not compute-identical and the result isolates an operational
training choice rather than a single scalar gradient in an otherwise fixed
system. Record GPU time and trainable parameter counts for both.

The three training colors are default `#778899`, blue `#003D99`, green
`#006B3C`. Purple `#A020F0` is validation-only and red `#FF0000` is
test-appearance-only. `scripts/generate_transition_colors.py` renders the
same saved physical state sequence under each color and checks the default
re-render against the source MP4. `scripts/audit_gradient_color_data.py`
checks manifests, videos, physical train/val episode fingerprints, and
per-color window counts. Color draws are deterministic and approximately
balanced; the audit records exact counts. The red appearance test reuses
physical validation trajectories, so it is **not** an independent physical
test split. Simulator state labels are used only by the frozen-encoder probe.

## Implementation and evidence

`training.gradient_mode` is opt-in (`predictor_only` or `joint`); `null`
retains the old training behavior. The mode validator rejects mismatched
`model.train_encoder`, target stop-gradient disabled, decoder/regularizer
losses, or unsupported `num_pred`. P does not construct an encoder optimizer;
J does. Both save a self-contained encoder in the checkpoint. The training
loop logs predictor and encoder gradient norms. `tests/test_gradient_ablation.py`
checks the gradient route, future-target stop-gradient, causal source/target
alignment, and validation color separation. Full CIRCE smoke tests and
trainings are **pending**; no success rates are claimed here.
Each run writes `optimizer_groups.json` with active optimizer parameter counts,
train/validation window counts, batches per epoch, frame settings, and seed.

| Condition | Seed | Trainable params | Clean MPC | Red MPC | One-step error | Multi-step rollout error | State probe | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P predictor-only | 0 planned | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| J joint | 0 planned | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

The released AdaJEPA checkpoint can be evaluated as context, but its earlier
training distribution/budget is not established as matched to P/J.

## Reproduction on CIRCE

Set paths to the actual cloned checkout and existing released assets. These
commands are **not executed by writing this report**. Rendered color data
and checkpoints are too large for Git and must already be present on CIRCE.

```bash
export REPO_DIR=/home/j/jinwoo/adajepa-multistep-gradient-ablation
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export DATASET_DIR=/home/j/jinwoo/adajepa/data_full
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export TARGETS=/home/j/jinwoo/adajepa/data/pushobj_eval/val_T/plan_targets.pkl
export RUN_ROOT="$REPO_DIR/gradient_outputs/seed0"
cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/logs"
export SUBMITTED_COMMIT="$(git rev-parse HEAD)"
PREVIEW_ONLY=1 sbatch --job-name=gradient-preview --partition=amd_2021 --cpus-per-task=4 --mem=32G --time=04:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/preview-%j.out" --export=ALL scripts/slurm_gradient_prepare.bash
```

Inspect `transition_colors_episode_*.png` under the train/val data directories
and the preview report. Then submit the full CPU render stage and run the smoke
tests before training:

```bash
sbatch --job-name=gradient-data --partition=amd_2021 --cpus-per-task=4 --mem=32G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/data-%j.out" --export=ALL scripts/slurm_gradient_prepare.bash
"$PYTHON_BIN" -m pytest tests/test_gradient_ablation.py tests/test_gradient_aggregate.py -q
test -f "$RUN_ROOT/preparation_done.txt" && cat "$RUN_ROOT/data_audit.json"
```

Check the audit and visual previews before the expensive jobs. The CIRCE Slurm
partition/QoS pair `muma_2021/preempt_short` worked in prior user runs; check
current eligibility and quota before submitting. The following are one job
per condition and may run after SSH disconnect:

```bash
export RUN_DIR="$RUN_ROOT/models/P"
sbatch --job-name=gradient-P --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/train-P-%j.out" --export=ALL scripts/slurm_gradient_ablation.bash P
export RUN_DIR="$RUN_ROOT/models/J"
sbatch --job-name=gradient-J --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/train-J-%j.out" --export=ALL scripts/slurm_gradient_ablation.bash J
```

Wait for both `model_3.pth` files. Evaluate frozen GD MPC on the same target
file, 50 targets, seed 100, and 30 replans, with no online adaptation:

```bash
sbatch --job-name=gradient-P-mpc --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/eval-P-%j.out" --export=ALL scripts/slurm_gradient_eval.bash P
sbatch --job-name=gradient-J-mpc --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/eval-J-%j.out" --export=ALL scripts/slurm_gradient_eval.bash J
"$PYTHON_BIN" scripts/aggregate_gradient_mpc.py "$RUN_ROOT"
```

Each completed evaluation writes a `done.txt` pointer to its unique attempt
directory. Rerunning after a failed attempt keeps old logs and starts a new
attempt. Aggregate after the Slurm jobs finish; missing conditions remain N/A.

Probe both **frozen** encoders with the same fit/evaluation protocol. These
commands use state labels only to fit diagnostic linear probes:

```bash
sbatch --job-name=gradient-P-probe --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/probe-P-%j.out" --export=ALL scripts/slurm_gradient_probe.bash P
sbatch --job-name=gradient-J-probe --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/probe-J-%j.out" --export=ALL scripts/slurm_gradient_probe.bash J
```

The probe uses the same 256 train and 128 validation episodes per model,
one midpoint frame per episode, ridge coefficient 1.0, and trains on all
three train colors. It records per-episode predictions, position/angle error,
normalized matched-color latent distance, feature variance, norm, and
effective rank. It cannot replace MPC success. The opt-in training path also
writes `rollout_horizon.jsonl` for validation trajectories and logs horizons
1 through 5, both raw visual MSE and MSE divided by that model's target-latent
mean squared magnitude. The latter is only a scale-aware diagnostic;
cross-model latent geometry still differs. Results remain pending until the
P/J jobs are actually run.
