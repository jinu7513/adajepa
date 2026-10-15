# RGB/Event encoder -> PushT predictor and MPC evaluation

## Fair Event encoder comparison (A/B/C)

The existing Event predictor checkpoint is experiment **A**: pretrained Event
encoder frozen, original AdaJEPA predictor plus action/proprio modules trained.
This is a freezing ablation, **not** the original AdaJEPA visual-encoder
baseline. A paper-style comparison also needs the original AdaJEPA checkpoint
evaluated on the same targets, seeds, corruption definitions, planner, and
adaptation modes. Its training data/epochs must be checked separately before
claiming a training-budget-matched comparison.
Do not delete, overwrite, or restart its directory while it is running. The
new recipes keep the same full PushT train/val trajectories, actual previous
video frame, `frameskip=5`, 3 history frames, 1 prediction frame, seed 0,
batch 32, and three completed epochs. **B** (`experiment=event_joint`) starts
from the *same* pretrained Event encoder but jointly updates all its student
parameters and the predictor. **C** (`experiment=event_final_shared`) updates
only the last shared Event transformer block plus `final_norm` and the
predictor. All three use the existing latent-dynamics MSE objective, with the
target latent stop-grad; B/C do **not** reuse the Event pretraining teacher or
auxiliary temporal head. An equal epoch count matches data exposure, not GPU
hours. B/C can be slower and B may need a smaller batch if the L40S runs out
of memory; record any deviation before claiming a controlled comparison.

The recipe files are `conf/experiment/event_frozen.yaml`,
`event_joint.yaml`, and `event_final_shared.yaml`. The optimizer LR for the
Event encoder is the existing `training.encoder_lr=1e-5`; predictor and
action/proprio LR retain the original settings. Each completed epoch saves an
embedded encoder, predictor, all relevant optimizer states, W&B data, and
`epoch_metrics.jsonl`. Iteration checkpointing is disabled for these recipes:
the existing dataloader does not restore a mid-epoch cursor, so interruption
can replay at most the unfinished epoch. The atomic `model_latest.pth` and
`model_N.pth` writes avoid treating an interrupted write as a valid checkpoint.

### CIRCE batch submission and resumption

From a CIRCE **login** shell, inspect the account's live scheduling options
before submitting; this repository cannot query your CIRCE account remotely:

```bash
sinfo -o '%P %a %l %G'
sacctmgr show assoc user="$USER" format=Account,Partition,QOS
myquota
df -h "$PWD"
```

The repository's old Hydra/Submitit launcher requests `gpu:h100:1` and 512 GB,
which is **not** suitable evidence that those resources are available to your
account. The independent `sbatch` wrapper instead requires a real partition
and walltime from the commands above; QOS/account are optional if the site's
defaults are valid. USF's [GPU job instructions](https://wiki.rc.usf.edu/index.php/SLURM_GPU)
show the `--gres=gpu:1` pattern, but their example hardware is dated.

Wait for the currently running A job to finish before pulling changed code
into that same checkout. Or use a separate worktree/clone for B/C. Then:

```bash
cd /home/j/jinwoo/adajepa
export DATASET_DIR="$PWD/data_full"
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export CIRCE_PARTITION=YOUR_GPU_PARTITION
export CIRCE_TIME=2-00:00:00
# export CIRCE_QOS=YOUR_QOS           # if required by your account
# export CIRCE_ACCOUNT=YOUR_ACCOUNT   # if required by your account
export EVENT_CKPT="$PWD/tracka_outputs/event_encoder_20261010_231557_c3bf41_recovered_from_5000/checkpoint_latest.pt"
bash scripts/submit_event_predictor.bash B "$PWD/checkpoints/event_joint_seed0" "$EVENT_CKPT" 3
bash scripts/submit_event_predictor.bash C "$PWD/checkpoints/event_final_shared_seed0" "$EVENT_CKPT" 3
```

Replace `EVENT_CKPT` with the verified 10,000-step pretrained checkpoint; the
encoder config rejects any other step. For a **new** A replication, use `A`
and a *new* output directory. Never pass the currently running A directory as
a fresh-run target. Submitting does not depend on a live SSH/VPN session; the
actual work starts only when Slurm allocates a GPU compute node.

```bash
squeue -u "$USER"                                      # pending/running jobs
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
tail -f checkpoints/event_joint_seed0/slurm-JOB_ID.out
tail -n 5 checkpoints/event_joint_seed0/epoch_metrics.jsonl
scancel JOB_ID                                         # cancel if necessary
RESUME=1 bash scripts/submit_event_predictor.bash B \
  "$PWD/checkpoints/event_joint_seed0" "$EVENT_CKPT" 3
```

Only resume after the old job has ended, and only if a completed-epoch
`model_latest.pth` and `hydra.yaml` exist. The job checks the recipe, data,
seed, windows, and checkpoint path before continuing; it subtracts the saved
epoch from the target, so `3` means a **total** of three epochs rather than
three additional epochs. The output-directory lock prevents two of these jobs
from writing the same checkpoint. If an epoch takes longer than the chosen
walltime, ask for a permitted longer allocation or reduce the per-job target;
repeated jobs otherwise restart that unfinished epoch.

### Matched deployment evaluation

After A/B/C have a completed `model_3.pth`, evaluate **each** with the same
`scripts/run_event_mpc_matrix.bash` script. It runs default, blur, and redBlock
under both Frozen and AdaJEPA, same `val_T/plan_targets.pkl`, seed 100, 50
samples and 30 MPC steps. Use a unique `EVAL_ROOT` and `VARIANT` per model; a
GPU `sbatch` allocation is recommended because these are expensive. Example
for B after live partition/QOS/walltime checks:

```bash
"$PYTHON_BIN" scripts/check_event_comparison.py \
  "$PWD/checkpoints/YOUR_EXISTING_A_RUN" \
  "$PWD/checkpoints/event_joint_seed0" \
  "$PWD/checkpoints/event_final_shared_seed0"
mkdir -p slurm_logs
MODEL_DIR="$PWD/checkpoints/event_joint_seed0"
export EVAL_ROOT="$PWD/eval_outputs/event_joint_seed0"
export VARIANT=event-joint-seed0
sbatch --partition="$CIRCE_PARTITION" --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time="$CIRCE_TIME" --output="$PWD/slurm_logs/event-mpc-%j.out" \
  scripts/run_event_mpc_matrix.bash "$MODEL_DIR"
```

Add `--qos="$CIRCE_QOS"`/`--account="$CIRCE_ACCOUNT"` when required.
Repeat for A and C with their own `MODEL_DIR`, `EVAL_ROOT`, and `VARIANT`.
For the separately trained **original AdaJEPA** checkpoint, the same evaluator
also accepts `MODEL_KIND=original` and, if needed, `MODEL_EPOCH=latest`:

```bash
export MODEL_KIND=original MODEL_EPOCH=latest
export EVAL_ROOT="$PWD/eval_outputs/original_adajepa_seed100"
export VARIANT=original-adajepa
MODEL_DIR="$PWD/checkpoints/pusht_visual_shift"
sbatch --partition="$CIRCE_PARTITION" --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time="$CIRCE_TIME" --output="$PWD/slurm_logs/original-mpc-%j.out" \
  scripts/run_event_mpc_matrix.bash "$MODEL_DIR"
```

This requires a valid `hydra.yaml` and `checkpoints/model_latest.pth` in that
directory. Verify that this original model's training data and budget are
comparable before interpreting it as a controlled original-vs-Event result.
Compare `paper/success_rate_pct` against `paper/mpc_step` at each step and
across matched seeds; the training validation latent loss alone is **not**
PushT task success. Current repo code includes the six-condition evaluator,
but none of the new B/C training or MPC jobs has been run here.

The `tracka-event-v1` checkpoint is not compatible with `encoder=tracka_patch`.
`encoder=tracka_event_patch` loads the frozen **student** backbone and passes its
196 spatial patch tokens to a newly initialized action-conditioned predictor.
The Event encoder receives each video frame and its actual preceding frame
(`t-1`), even though predictor frames are sampled every `frameskip=5`. Frame
zero uses itself as its preceding frame. Training uses the full PushT trajectory
dataset, not the Event-window pairs, and fits the predictor plus action/proprio
embeddings; it does not retrain the encoder or transfer the Event auxiliary `F`.

MPC evaluation supports this checkpoint through `planning/event_observation.py`.
For each rendered rollout, the Event encoder receives the actual preceding
rendered frame; frame zero is paired with itself. The goal's preceding frame is
the penultimate frame from the replayed target segment. Visual corruption is
applied before pairing so `visual` and `visual_previous` refer to the same
corrupted stream. The pretraining temporal `F` is action-free and is not the
world-model predictor. This evaluation integration has CPU tests but still
requires an end-to-end CIRCE smoke run before a full benchmark.

For a completed epoch-3 predictor checkpoint, `scripts/run_event_mpc_matrix.bash`
runs default, blur, and redBlock under both Frozen and AdaJEPA (GD) with the
same segment targets and seed. It first runs a one-sample, two-MPC-step smoke
case for each method, then six 50-sample, 30-step evaluations; each result has
a separate directory and W&B run. Frozen uses `adapt.steps=0`, while AdaJEPA
uses one adaptation step and restricts Event encoder adaptation to the final
shared transformer block and norm. Results are not claimed to reproduce the
paper until the CIRCE smoke and full evaluations complete. Use an absolute
model directory containing `hydra.yaml` and `checkpoints/model_3.pth`:

```bash
cd /home/j/jinwoo/adajepa
bash scripts/run_event_mpc_matrix.bash \
  "$PWD/checkpoints/pusht_event_patch_10k_3epoch_b32_w8_block2_frame32_20261012_132654"
```

To resume epoch 3 if necessary and then run the matrix, use
`scripts/run_event_epoch3_then_mpc.bash` as a Slurm batch payload. Do not launch
it while another training process writes the same checkpoint directory. An
interactive SSH command is **not** automatically made disconnect-safe by this
script; submit it with site-approved GPU partition, QOS, CPU, memory, and
walltime options via `sbatch`. The batch script reuses the same Hydra output
directory and W&B training run ID, and fails fast if the smoke test fails.

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

### Interrupted encoder recovery

An interrupted Event run can log update 5060 while its latest checkpoint is
still step 5000 (`save_every=100`). The extra logged 60 updates have **no
recoverable optimizer/teacher state**. In-place resume intentionally refuses
to rewind and duplicate the durable JSONL/W&B history. The pipeline detects
this case and forks from the verified 5000-step checkpoint into a fresh sibling
directory named `<original>_recovered_from_5000`, with a **new W&B run ID**.
The old metrics, W&B history and checkpoint are left untouched. The recovered
run records its source path, step and SHA-256 in metadata, then uses its own
10000-step checkpoint for predictor training. Never run the pipeline alongside
another process writing the source checkpoint.
