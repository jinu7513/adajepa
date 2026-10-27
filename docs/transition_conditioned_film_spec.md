# Transition-conditioned FiLM for deterministic AdaJEPA: implementation specification

Status: design for review; no model or dataset code has been changed yet.

Branch: `transition-conditioned-film`, based on `main` at `66c5f18`. This is a
separate worktree from the existing Event-encoder experiments. Do not alter or
reuse their in-progress checkpoints.

## Research question and scope

Can a predictor use a *completed, observed* transition to cope with a new
PushT block appearance when the visual encoder is frozen? Extend the existing
deterministic predictor with a small transition-conditioned FiLM module. Do not
change the encoder, add a goal representation, use Flow-JEPA, alter the
prediction horizon/loss, or perform online gradient updates in the primary
comparison. This is transition **conditioning**, not test-time AdaJEPA gradient
adaptation. Evaluate the latter, if desired, as a separate 2-by-2 factor.

Encoder decision: use the **trained visual encoder from the released vanilla
AdaJEPA PushT model in `checkpoints/pusht_visual_shift`**, not a freshly
instantiated DINOv2 encoder and not the Event/Track A encoder. The repository
README explicitly identifies this directory as a released PushT visual-shift
checkpoint, and the user selected it as the source. Freeze that exact encoder
for A, B and C. Before implementation on CIRCE, verify which `model_*.pth`
file is to be used, that its `hydra.yaml` and checkpoint contain the expected
encoder and predictor, and record the resolved path, epoch, model config and
file hash. Do not infer that `model_latest.pth` is identical to a numbered
checkpoint without checking.

The clean-only original model (A) is a reference. The controlled comparison is
color-trained deterministic predictor (B) versus identically color-trained
FiLM predictor (C): same frozen encoder, physical trajectories, color draws,
mini-batch order, action/proprio inputs, base predictor initialization, and
optimizer-update budget. Record the extra parameters and computation in C.

Confirmed first-pass budget: initialize B and C from the **same released
predictor weights** associated with the selected encoder and fine-tune both
for **three complete training epochs**, saving a checkpoint after every epoch.
Match the exact number/order of physical windows and color draws, batch size,
optimizer and LR schedule across B/C. Select checkpoints using default and
train-color validation only; never use redBlock for model selection. The
released clean model A is evaluated as-is, not retrained for three epochs.

## Verified repository contracts

- `conf/train.yaml` defaults to `num_hist=3`, `num_pred=1`, `frameskip=5`, and
  `concat_dim=1`. The current PushT slicer supplies four observations and four
  five-step action chunks per sample. The action chunk connecting observed
  frames `i-1` and `i` is chunk `i-1`, *not* chunk `i`.
- `VWorldModel.forward` predicts from `z[:, :3]` and compares three output
  slots with detached `z[:, 1:]`: slot 0 predicts observation 1, slot 1
  predicts observation 2, and slot 2 predicts observation 3. Preserve all
  three losses and the existing normalization/reductions.
- `VWorldModel.predict` flattens time and patch dimensions before passing
  tokens to `ViTPredictor`. Its `pool` argument does **not** introduce a CLS
  token. Pool *visual patch embeddings only* in the new context branch; do
  not pool the concatenated proprio/action channels, create a fake CLS, or
  replace the predictor's patch tokens.
- `ViTPredictor` applies causal attention by frame and consists of residual
  attention/FFN blocks followed by a final normalization. Preserve token
  order, causal mask and base transformer weights. Inspect the chosen
  checkpoint for actual embedding size and depth; do not hardcode 404 or six
  blocks in the implementation.
- `VWorldModel.rollout` encodes initial real observations, then repeatedly
  appends predicted latent frames for candidate actions. `GDPlanner` calls
  this rollout repeatedly for *hypothetical* candidates. These predictions
  are not completed physical transitions.
- The PushT training dataset currently reads stored MP4 frames. The PushT
  environment can set block color at render time, but the existing dataset
  loader does not recolor the training MP4s. A paired-color training path is
  required; merely setting an environment argument is not enough.
- Existing checkpoints serialize model objects. `train.py` currently saves
  the encoder in a new checkpoint only when `model.train_encoder=true`.
  Because B/C freeze it, make their artifacts self-contained or explicitly
  record and verify the immutable source encoder checkpoint at evaluation;
  a standalone B/C checkpoint must not silently load fresh DINO weights.
  Keep disabled-FiLM behavior and old-checkpoint loading unchanged. The predictor attention mask is
  currently allocated directly on CUDA; any CPU test support must fix this
  without changing the mask's values or causal semantics.

## Model and causal information flow

Let `v_i = E(o_i)` denote frozen visual patch embeddings and `a_i` the
*normalized five-step action chunk* beginning at observed frame `i`. Use the
existing action embedding where possible. For an available completed
transition define

`c_i = MLP([mean_patches(v_{i-1}), action_embed(a_{i-1}), mean_patches(v_i)])`.

For each prediction slot `i` in a training window:

| Source slot | Target | Completed transition available to that slot |
| --- | --- | --- |
| `i=0` | `o_1` | None inside the sampled window: invalid context |
| `i=1` | `o_2` | `(o_0, a_0, o_1)` |
| `i=2` | `o_3` | `(o_1, a_1, o_2)` |

This alignment is essential: using `(o_1,a_1,o_2)` for slot 0, for example,
would reveal the slot-0 target. Do not silently switch to only the last
prediction loss. If additional earlier history is genuinely available, a
future version may supply valid context for slot 0; do not synthesize it from
the target.

The FiLM adapter produces per-channel `gamma_{l,i}` and `beta_{l,i}` for the
last two existing transformer blocks (or fewer if the checkpoint has fewer
than two). Expand each slot's parameters over *that slot's* patch tokens:

`h'_{l,i,p} = (1 + m_i * gamma_l(c_i)) * h_{l,i,p}
              + m_i * beta_l(c_i)`,

where `m_i` is an explicit per-slot validity mask. Apply modulation after the
block's existing attention-plus-FFN residual output and before the next block
or final norm. Zero-initialize only the final gamma/beta projection; this
makes C prediction-identical to B immediately after loading the same base
predictor, including with valid context. Mask *after* the projection so an
invalid context remains identity even once projection biases have trained.
The adapter should be new optional state, not a change to default predictor
outputs or existing checkpoint keys. A zero context vector without a mask is
not sufficient.

Training uses only observed frames `o_0,o_1,o_2` when forming contexts.
The context branch must not receive `o_3`, target latents, later action
chunks, goal images, success labels, or previous episodes. Freeze `E`; train
the same base predictor parameters in B and C, plus the FiLM/context weights
in C. If the experiment also trains action/proprio encoders, match that choice
between B and C and document it.

### Planning/runtime state

Maintain a per-environment buffer of **actually observed** frames and
executed action chunks. At reset there is no completed transition, so all
unavailable contexts have `m=0`. After executing a five-step chunk and
receiving the resulting observation, append exactly that completed transition
and update the latest valid context. Never construct `c` from imagined
rollout latents or candidate actions. During one candidate rollout, use the
available per-slot contexts for real-history slots; once slots become
imagined, hold the latest *real* context fixed. The planner must pass this
state separately from the candidate action tensor and clear it on every
episode reset, including vectorized/per-sample resets. The conditioning
policy must be identical for GD and CEM if both are used.

This real-context/held-context boundary creates a possible train/inference
distribution difference for long imagined rollouts. Log how many slots use
invalid, real, and held context; test the actual MPC code path rather than
claiming this difference is absent. Do not back-fill a future context using
predicted observations to hide the difference.

## Appearance data protocol

The intervention is **block color only**. Keep agent, anchor/goal,
background, geometry, physics, physical state, action sequence, reward, and
success condition fixed. Use episode-disjoint train/validation/test physical
trajectories; different renderings of one episode must never cross splits.
Persist a manifest with source episode IDs, state/action hashes, split,
renderer revision, block RGB values, and frame-to-action alignment.

Confirmed color split: **never use red block images for B/C training or model
selection**; reserve `redBlock` as a held-out evaluation domain. Use this
fixed block-color table for the first experiment:

| Split | Label | Renderer color argument | RGB |
| --- | --- | --- | --- |
| Train | default | `LightSlateGray` | `(119, 136, 153)` |
| Train | blue | `#003D99` | `(0, 61, 153)` |
| Train | green | `#006B3C` | `(0, 107, 60)` |
| Held-out test only | redBlock | `Red` | `(255, 0, 0)` |

These exact color values were checked with the repository's `pygame.Color`
rendering interface. The blue is deliberately darker than the `RoyalBlue`
agent `(65,105,225)` and the green darker than the `LightGreen` target
`(144,238,144)`. Confirm their perceptual separation in a same-state render
contact sheet before generating the full dataset. Do not tune the train colors
based on redBlock evaluation results. Use the exact same color draws for B
and C. Baseline A uses only the original clean training frames. No validation
images from the held-out red domain may influence checkpoint selection or
hyperparameter tuning.

Preferred generation path: render the saved physical states with the same
camera/settings and block-only color override, then verify paired frames.
Before bulk generation, compare same-state default renders to source MP4s
and verify agent/anchor/background pixels, block silhouette, and physical
trajectories. If re-rendering is not faithful, investigate rather than using
naive pixel substitution. Do not contaminate validation/test with train
episode variants. Feature caching is optional and must be keyed by encoder
checkpoint, image transform, renderer version, color, and source episode.

## Controlled evaluation

1. Evaluate A (clean reference), B, C using the same PushT target file and
   planning settings. First-pass values: **GD**, evaluation seed **100**,
   **50** target episodes, and **30** maximum MPC steps. The checked-in
   `conf/adajepa_plan_gd_pushobj.yaml` uses seed 100 and 50 evaluations but
   defaults to 20 MPC steps, so 30 must be an explicit, recorded override.
   Use the same `val_T/plan_targets.pkl` target file for default and redBlock.
   Record actual target-file path/hash, checkpoint ID, selected 50 target IDs
   and all overrides in each run manifest. The longer 30-step budget matches
   the user's earlier curves and must not be compared directly to 20-step
   runs without noting the difference.
2. Primary B-versus-C comparison: **no test-time gradient adaptation** for
   either arm. Evaluate default and held-out redBlock first; then any
   additional implemented visual shifts (for example blur) as secondary
   robustness checks. A block-color method is not automatically a blur
   method. Keep the underlying targets fixed and apply only the specified
   evaluation-time appearance intervention.
3. Report success rate (%) versus MPC step, final success, episode count,
   seeds, confidence intervals/paired bootstrap, and paired per-episode
   outcomes. Also measure one-step latent prediction loss on matched physical
   transitions by domain. Preserve per-step JSON/CSV and W&B history with
   variant, color, seed, checkpoint, and code revision in run metadata.
4. Diagnostic ablations: correct versus zero versus mismatched context at
   fixed query inputs; optional current-image-only and shuffled-action
   contexts. Those distinguish a real transition benefit from simply reading
   color from the current image. A mismatched context is *not* a deployment
   policy. Report cold-start separately from after real observed transitions.
5. If evaluating online AdaJEPA gradients, cross B/C with gradient updates
   on/off as a **separate** comparison, using identical adaptation budgets.
   Do not label transition-FiLM alone as online gradient adaptation.

## Configuration and implementation boundaries

Fit the repository's Hydra convention, but finalize field names only while
implementing. The essential independent controls are: FiLM on/off (default
off), adapter context size and selected layers, context validity/source,
training color manifest, frozen encoder checkpoint, base predictor
checkpoint, dataset split/seed, and evaluation color. Do not assume that a
Hydra override works merely because its name is proposed here.

Likely touch points: `models/vit.py`, `models/visual_world_model.py`,
`datasets/pusht_dset.py` or a new paired-render dataset module, planning
state/rollout code, `train.py`, Hydra config files, focused tests, and the
experiment README. Keep the default model constructor and old checkpoint
loader functional. Do not import unrelated Event-encoder changes from the
other worktree without a separate design decision.

## Acceptance tests before full runs

- In evaluation mode, C with zero-initialized FiLM matches B's predictions
  numerically for the same loaded predictor weights, for valid and invalid
  contexts. Old checkpoints still load when FiLM is disabled.
- Changing `o_2` or later actions cannot change slot-0 output; changing
  `o_3` cannot change any input context. Check exact five-step action-chunk
  alignment and causal attention mask.
- Gradients reach context MLP, FiLM and base predictor as configured, but
  not the frozen visual encoder. Test batch size 1 and >1 and supported
  devices. Avoid CUDA-only mask allocation in CPU smoke tests.
- Candidate actions/predicted latent states cannot mutate observed context;
  running GD/CEM candidate rollouts in another order gives identical context.
  Reset clears context independently for each environment.
- Paired color images differ only as intended, with matching physical
  states/actions and episode-disjoint split IDs.
- A small end-to-end train/save/load/plan smoke test writes metrics to disk
  and W&B when enabled; distinguish it from a final success-rate result.

Do not launch full training or evaluation until the same-state render preview
passes, the source checkpoint file and common evaluation budget are fixed,
and the smoke tests pass.
