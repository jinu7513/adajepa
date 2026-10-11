# RGB/Event encoder -> PushT predictor (training-only integration)

The `tracka-event-v1` checkpoint is not compatible with `encoder=tracka_patch`.
`encoder=tracka_event_patch` loads the frozen **student** backbone and passes its
196 spatial patch tokens to a newly initialized action-conditioned predictor.
The Event encoder receives each video frame and its actual preceding frame
(`t-1`), even though predictor frames are sampled every `frameskip=5`. Frame
zero uses itself as its preceding frame. Training uses the full PushT trajectory
dataset, not the Event-window pairs, and fits the predictor plus action/proprio
embeddings; it does not retrain the encoder or transfer the Event auxiliary `F`.

This changes the predictor training data path only. MPC/evaluation does not yet
provide `visual_previous` for environment observations; **do not use this
checkpoint with `plan.py` yet**. Matching predictor input at MPC time needs
the actual preceding rendered frame and separate integration/testing. The
pretraining temporal `F` is action-free and is not the world-model predictor.

On CIRCE, the following script verifies the Event checkpoint, resumes it to
10,000 updates if necessary, runs CPU integration tests, runs a 2-trajectory
one-epoch smoke test, and only then trains for three epochs using all PushT
train/validation trajectories. Each command is chained with fail-fast shell
behavior. It requires `data_full/pusht_noise/{train,val}` with complete actions,
states, velocities, lengths and videos, or an absolute `DATASET_DIR` override.
It uses the known CIRCE `ts` Python by default; override `PYTHON_BIN` if needed.
Outputs have fixed names and the script refuses to reuse them, preventing an
accidental second three-epoch continuation on rerun.
Do not launch it concurrently with an Event encoder job already using the same
checkpoint; the script also rejects a run marked `running`.

```bash
cd /home/j/jinwoo/adajepa
bash scripts/run_event_predictor_pipeline.sh \
  "$PWD/tracka_outputs/event_encoder_20261010_231557_c3bf41"
```

For an SSH-disconnect-safe launch, from the repository root use a detached tmux
session and keep a log. If `event_pipeline` already exists, choose another name.

```bash
tmux new-session -d -s event_pipeline \
  "bash '$PWD/scripts/run_event_predictor_pipeline.sh' '$PWD/tracka_outputs/event_encoder_20261010_231557_c3bf41' > '$PWD/event_pipeline.log' 2>&1"
```

Monitor with `tail -f event_pipeline.log`. The full predictor run is written to
`checkpoints/pusht_event_patch_10k_3epoch`; W&B project is
`adajepa_trackA_event_predictor`, run name `event-patch-10k-3epoch-seed0`.
The smoke run is separate and uses just two trajectories. Both predictor runs
save a frozen encoder in their checkpoints. Three epochs are a pilot budget,
not a claim of adequate predictive or MPC performance; compare training and
validation latent loss before extending it. `env.num_workers=1` avoids the
earlier 16-worker validation kill, while batch size 8 is conservative for the
L40S. This command has not yet been run on CIRCE and is not a GPU performance
or successful-completion guarantee.
