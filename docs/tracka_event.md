# Trajectory-contiguous RGB + Event-proxy encoder

This is a **separate, reconstruction-free experiment** on the
`rgb-event-robust-encoder` branch. It does not alter the existing Track A MAE
encoder, predictor, or MPC planner. The event maps are **proxies derived from two
RGB frames**, not asynchronous recordings from an event camera. The model is
trained from scratch; its teacher is an EMA copy of its student, not DINO.

## Dataset contract and preview

Start with the existing verified `data/pairs` on CIRCE. For every accepted center
state `(trajectory_id,t)` except trajectory boundaries, the generator retrieves
**t-1,t,t+1 from the original trajectory states**, even when the centers were
sampled at stride 5. It never pairs neighboring `samples.jsonl` rows. The
source pair manifest's train/validation/test trajectory assignments are reused.
All six images in a window share the same render seed; a color palette stays
fixed across the window. Gaussian/salt-pepper noise is independently sampled
per frame but uses one strength for the window. The generator checks the center
clean image against the old verified dataset byte-for-byte and validates reset
physics. Boundary and invalid-reset windows are rejected and reported. A
different renderer version fails closed rather than silently introducing fake
motion. The shifted next image is stored for future ablations but not decoded
by the baseline training loader. Inspect the image and event contact sheets
before full generation.

```bash
python generate_event_windows.py --config-name tracka_event \
  dataset.source_pairs_path=data/pairs dataset.output_path=data/event_preview \
  generation.preview_only=true

# After reviewing data/event_preview/window_contact_sheet.png and
# data/event_preview/event_contact_sheet.png:
python generate_event_windows.py --config-name tracka_event \
  dataset.source_pairs_path=data/pairs dataset.output_path=data/event_windows \
  generation.preview_only=false generation.ranges_reviewed=true
```

The baseline uses two-channel positive/negative thresholded changes of log
luminance. RGB PNGs are converted from `uint8` to `[-1,1]` by the loader and
**back to `[0,1]`** before luminance/log. Configurable modes are `hard_signed`,
`hard_two_channel`, `delta_log`, and `soft_clipped`. `model.event_threshold=0.2`
is a **provisional pilot setting**, not a calibrated sensor parameter. Compare
the preview's clean/shift event densities, especially for blur and noise,
before accepting it. `model.shift_previous=false` keeps the one-pixel-shift
ablation off. Density is the fraction of pixels with `|Delta log I| >= C` in
all modes; per-patch density is average-pooled on the same patch grid.

## Network and training

With 224x224 RGB, 16x16 patches, and D=384, both streams produce `[B,196,384]`.
The RGB and Event patch embeddings are distinct. Each stream has one specific
transformer block; five shared blocks preserve six blocks along either path.
Independent LayerNorms establish comparable stream scale before fixed sum:
`z = LN(z_rgb) + LN(z_event)`. A **single shared CLS** is inserted after fusion;
the output patch grid remains `[B,196,384]` and CLS is `[B,384]`. The optional
patch-wise gate takes `[LN(z_rgb),LN(z_event),event_density]` and produces
`z_rgb + sigmoid(gate)*z_event`; the default is fixed sum with event weight 1.
`rgb_only` with `model.rgb_input_mode=pair` is the two-RGB-frame control.
`event_only`, `gated_sum`, `mean_pool`, and alternate event modes are selectable.

Student input is the shifted previous/current RGB pair and its derived event;
teacher input is the clean pair. `q` independently predicts each teacher patch
at the **same time** from a student patch. `F` predicts teacher next-step CLS
from the student current CLS, without action. Both are D -> 2D -> D MLPs.

```
L = lambda_rob * d(q(Z_student_t), stopgrad(Z_teacher_t))
  + lambda_temp * d(F(g_student_t), stopgrad(g_teacher_t+1))
  + lambda_ac * L_anti_collapse
```

The default distance is `1-cosine`; `normalized_mse` and `mse` are available.
There is **no RGB/Event reconstruction head and no masking**. After each
optimizer step, teacher parameters receive
`teacher <- momentum*teacher + (1-momentum)*student`; teacher gets no gradient.
The default anti-collapse loss is disabled. Optional SIGReg is on a training-only
global-feature projector (CLS by default), or VICReg-style variance/covariance is applied to student global
features. Those are ablations, not claims that collapse is solved.

Run tests and a two-update smoke before a longer job:

```bash
python -m pytest tests/test_tracka_event.py -q
python train_event_encoder.py --config-name tracka_event \
  dataset.output_path=data/event_windows training.total_optimizer_updates=2 \
  training.batch_size=2 training.rank_samples=8 training.validation_batches=1 \
  training.validate_every=2 training.save_every=2 logging.log_every=1 \
  logging.name=trackA-event-fixed-sum-smoke logging.fallback_to_local=false
```

Training writes `metrics.jsonl`, `checkpoint_latest.pt`, config/provenance, and
W&B in project `adajepa_trackA_event`. `train/global_step` is the W&B x-axis.
Diagnostics include clean/shift event density, local patch density, log-intensity
change, global/patch raw variance and norm, clean/shift latent distance,
effective rank and covariance spectrum concentration on 256 validation
samples, gate statistics, teacher-input temporal error and a persistence
baseline. `loss.lambda_ac=0` is a monitored experiment: stop and report if
raw-feature rank/variance collapse; do not infer success from projector spread.

For a pilot after the smoke test, change only the update budget and run name.
Do not launch 10,000 updates before verifying the preview and early metrics.
Resuming requires `training.resume=/absolute/path/to/checkpoint_latest.pt` and
a larger `training.total_optimizer_updates`; the model, optimizer, teacher,
RNG, dataset hash, and original W&B run ID are restored.

## Frozen evaluation and fair controls

The Event checkpoint has its own evaluator; the old single-frame
`eval_encoder_probes.py` cannot load it. The **student** encoder is evaluated by
default; set `evaluation.weights=teacher` for a separately named ablation.
Clean train features fit a linear state probe; clean validation chooses its
ridge coefficient; that frozen probe is evaluated on clean, stored shifts,
redBlock/redAgent/redAnchor, Gaussian, salt-pepper, blur, and dark test windows.
The same trajectory split, sample cap and seed must be used in all comparisons.

```bash
python eval_event_encoder.py --config-name tracka_event \
  dataset.output_path=data/event_windows \
  evaluation.checkpoint=/absolute/path/to/run/checkpoint_latest.pt \
  evaluation.feature=global evaluation.weights=student \
  evaluation.max_states_per_split=1000 \
  logging.name=trackA-event-fixed-sum-probe logging.fallback_to_local=false
```

The evaluator saves `results.csv`, `diagnostics.json`, the fitted clean-state
probe, and robustness plots, and mirrors values to W&B. It reports clean
teacher-CLS -> `F` -> next teacher CLS, shifted student-CLS -> `F` -> next
teacher CLS, patch prediction, and teacher-CLS persistence. These temporal
errors use the configured representation distance, not physical state units.

Fair controls require separately trained encoders with the same windows,
training budget and seed:

```
RGB current only: model.fusion=rgb_only model.rgb_input_mode=current
RGB two frames:  model.fusion=rgb_only model.rgb_input_mode=pair
Event only:      model.fusion=event_only
RGB + Event:     model.fusion=fixed_sum
Gated fusion:    model.fusion=gated_sum
```

The two-frame RGB control distinguishes Event representation from simply
seeing another frame. The optional branches have different parameter/FLOP
counts; report these when comparing efficiency. Run metadata records active
encoder parameters separately from all instantiated parameters, along with
`q` and `F` counts; wall-clock time is logged per update. These are not FLOP
measurements. A low self-supervised loss, event density or effective rank
alone is not evidence of better state or
control information. Because `F` has no action input, future-latent error is
**action-free predictive representation learning**, not an action-conditioned
world model. Direct MPC success-rate testing needs a separate previous-frame
input adapter and predictor/planner integration, intentionally not included.
Patch-wise alignment averages foreground and background equally, so a small
`loss/robust` can still be obtained largely from static background/position
structure. Judge dynamics relevance with the state probe and the persistence
comparison; foreground weighting would be a later, separately reported ablation.
