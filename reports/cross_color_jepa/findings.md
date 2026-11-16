# Track A: Cross-Color JEPA

## Hypothesis and isolated change

Can the original action-conditioned JEPA prediction loss learn a more transferable visual representation when the predictor's observed images and the future target images show **the same physical trajectory under different T-block colors**? The physical states, actions, proprioception, timing, camera, shape and goals are paired. No inverse head or MI-inspired term is created in this track.

The original model uses `SmallResNetGeM` (one 384-dimensional visual token), a 6-layer causal `ViTPredictor`, `num_hist=3`, `num_pred=1`, `frameskip=5`, and a shared visual encoder with `stop_grad` on the target branch. Its training loss is three teacher-forced one-step visual/proprio latent MSE terms per four-frame window. The new paired loader supplies `visual` for all input history frames and `visual_target` for all future target frames; `VWorldModel.forward` keeps the existing loss reduction and predictor unchanged.

## Arms

| Arm | Input and target color | Mapping across epochs |
|---|---|---|
| `same_fixed` | same | existing deterministic window rule |
| `same_window_random` | same | seeded redraw each epoch |
| `cross_fixed` | distinct; six ordered pairs | balanced deterministic window rule |
| `cross_window_random` | distinct; six ordered pairs | seeded redraw each epoch |

The primary comparison is `same_window_random` versus `cross_window_random`. The fixed arms diagnose mapping effects. All arms use only default `(119,136,153)`, blue `(0,61,153)`, green `(0,107,60)` for training. Purple `(160,32,240)` is validation appearance and red `(255,0,0)` is final unseen appearance. Purple and red reuse the same physical validation trajectories, so this is an appearance holdout rather than an independent physical test split.

All arms initialize the encoder, predictor and action/proprio embedders from the same released PushT checkpoint, keep the encoder in eval mode while updating its parameters, and use the same seed, batch size, original JEPA loss and total epoch budget. The paired input/target render path adds a second image-encoding pass even in same-color controls; this holds the compute path more closely matched within Track A.

## Evidence and interpretation status

| Arm | Seed | Checkpoint | Default MPC | Purple MPC | Red MPC | Horizon 1–5 error | Frozen pose/contact transfer | Collapse |
|---|---:|---|---:|---:|---:|---|---|---|
| same_fixed | 0 planned | pending | N/A | N/A | N/A | N/A | N/A | N/A |
| same_window_random | 0 planned | pending | N/A | N/A | N/A | N/A | N/A | N/A |
| cross_fixed | 0 planned | pending | N/A | N/A | N/A | N/A | N/A | N/A |
| cross_window_random | 0 planned | pending | N/A | N/A | N/A | N/A | N/A | N/A |

No Track A training or evaluation has been executed in this worktree. A lower latent loss alone will not establish better robustness; use target-paired MPC success and frozen probe transfer. Raw latent MSE is not directly comparable across learned encoders without the within-model normalized measure. Actual color-pair counts, contact-stratified coverage and performance remain to be measured.
