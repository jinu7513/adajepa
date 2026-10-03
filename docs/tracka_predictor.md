# Frozen Track A patch tokens -> PushT world-model predictor

This is a new predictor experiment, not a zero-shot swap into the released DINO
predictor. `TrackAPatchEncoder` loads a trusted Track A `checkpoint_latest.pt`,
returns the 196 patch tokens (`[B,196,384]` for the 224/16 model), and leaves its
CLS token out of the predictor interface. The CLS token still participates in
the encoder's self-attention. The encoder is frozen; the predictor and
action/proprio embeddings are trained on trajectories. The decoder is disabled.
The resulting world-model checkpoint stores the frozen encoder as well as the
predictor, so `plan.py` does not need the original Track A file at evaluation.

Before a GPU run, the CPU integration tests exercise checkpoint loading,
patch-token shapes, one predictor forward/backward pass, and the planner's
self-contained checkpoint loader:

```bash
python -m pytest tests/test_tracka_predictor_integration.py -q
```

`PushTDataset` divides video pixels by 255, and `default_transform` then maps
them to `[-1,1]` at 224 px, matching Track A's normalization and resolution.
The trainer checks the encoder's patch
grid against the predictor configuration before constructing the predictor.

## Required inputs on CIRCE

From the repository root, set the encoder artifact and dataset root to absolute
paths. The shown checkpoint is the 10k continuation from this experiment.

```bash
export TRACKA_ENCODER_CKPT="$PWD/tracka_outputs/robust_alternating_resume5300/checkpoint_latest.pt"
export DATASET_DIR="$PWD/data"
```

The predictor dataset must be the **complete** PushT trajectory dataset, not
`data/pairs`. Under `$DATASET_DIR/pusht_noise`, **both** `train` and `val` need
`states.pth`, `rel_actions.pth`, `seq_lengths.pkl`, `velocities.pth`, and
`obses/episode_000.mp4` (plus subsequent episode videos). The earlier CIRCE
listing showed only states, velocities, and sequence lengths in `train`, so
check/download the missing actions, videos, and validation split before running.
The new preflight stops before W&B initialization if required files or any
video referenced by the selected trajectory count are absent. It checks file
presence, not video integrity.

```bash
test -f "$TRACKA_ENCODER_CKPT" && echo "Track A checkpoint OK"
for split in train val; do
  for name in states.pth rel_actions.pth seq_lengths.pkl velocities.pth obses/episode_000.mp4; do
    test -f "$DATASET_DIR/pusht_noise/$split/$name" || echo "MISSING: $split/$name"
  done
done
```

## Small GPU smoke test (only after all inputs exist)

Use a fresh output directory. `env.dataset.n_rollout=2` limits each split to
two trajectories, `training.rollout_eval_count=0` skips costly open-loop
rollouts, and one epoch is only a wiring test, **not** a trained model.

```bash
python train.py --config-name train env=pusht encoder=tracka_patch model.train_encoder=false model.train_predictor=true model.train_decoder=false has_decoder=false training.save_frozen_encoder=true training.epochs=1 training.batch_size=2 training.mixed_precision=no training.rollout_eval_count=0 training.save_every_x_iterations=0 env.dataset.n_rollout=2 env.num_workers=1 wandb_project=adajepa_trackA_predictor wandb_run_name=trackA-patch10k-predictor-smoke hydra.run.dir="$PWD/checkpoints/pusht_tracka_patch_10k_smoke"
```

Verify the run creates `hydra.yaml` and `checkpoints/model_latest.pth` in that
directory. The W&B project/name are explicit. The model checkpoint contains
`encoder`, `predictor`, `action_encoder`, and `proprio_encoder`. The smoke run
is not suitable for MPC conclusions.

## Full predictor training (after smoke succeeds)

This uses the repository's 20-epoch default and the full available PushT
trajectory data. It has not been run or validated on CIRCE by this change.

```bash
python train.py --config-name train env=pusht encoder=tracka_patch model.train_encoder=false model.train_predictor=true model.train_decoder=false has_decoder=false training.save_frozen_encoder=true training.epochs=20 training.batch_size=32 wandb_project=adajepa_trackA_predictor wandb_run_name=trackA-patch10k-predictor-seed0 hydra.run.dir="$PWD/checkpoints/pusht_tracka_patch_10k_seed0"
```

Compare W&B `train_z_visual_loss`, `val_z_visual_loss`, and open-loop rollout
errors before MPC. A drop in latent prediction loss alone does not establish
control success. Keep the selected encoder checkpoint and all dataset paths
recorded with the run. The full model checkpoint also embeds the encoder, so
MPC uses the exact trained latent space even if the source file later moves.

## Planning, only after predictor validation

`plan.py` reads the self-contained world-model checkpoint in the training run
directory, **not** the standalone Track A checkpoint. With the same PushT
targets used for the DINO baseline, first run a small frozen-planner check:

```bash
python plan.py --config-name adajepa_plan_gd_pushobj.yaml model_name=pusht_tracka_patch_10k_seed0 ckpt_base_path="$PWD/checkpoints/pusht_tracka_patch_10k_seed0" eval_data_path="$PWD/data/pushobj_eval/val_T/plan_targets.pkl" planner.adapt.lr=0 planner.adapt.steps=0 planner.adapt.finetune_encoder=false n_evals=5 planner.max_iter=5 +wandb_logging=true +wandb_project=adajepa_trackA_predictor_eval hydra.run.dir="$PWD/eval_outputs/tracka_patch_10k_frozen_smoke"
```

Then use matched evaluation seeds, target files, MPC budgets, and corruption
settings for comparison with the released model. Do not interpret the frozen
probe's standardized state MSE as MPC success rate.

Current limitations: this path uses patch tokens only; there is no CLS-token
predictor or Flow-specific architecture here. The existing DINO predictor
weights are not transferred. Planning with AdaJEPA adaptation has not been
separately verified for this wrapped encoder; start with frozen adaptation as
shown above.
