# A/B/C plus causal transition-support AdaJEPA on CIRCE

This extends (without replacing) the [A/B/C workflow](transition_film_slurm.md).
Pulling code **does not submit jobs**. The default submitter below schedules
the earlier A/B/C matrix, a matched-window vanilla control, and all eight
new support-model architecture arms.
It leaves the released `pusht_visual_shift` checkpoint and Event checkpoints
untouched. Use the separate `adajepa-transition-film` checkout so a Git pull
does not change files read by existing Event jobs.

## Scientific setup

- Released AdaJEPA `A`: reference encoder and predictor `F0`.
- `B`: same frozen encoder, base predictor trained on paired default/blue/green.
- `C`: earlier single-transition FiLM predictor on the same paired colors.
- Matched-window vanilla: identical 10-frame physical windows and query-only
  objective as the support models, but FiLM fixed to identity (K=0). This
  is the fairest predictor-only control for the new architecture.
- New support models: `delta` vs frozen-reference `residual` features, frozen
  vs jointly fine-tuned base predictor, and MLP vs linear FiLM generator
  (2 × 2 × 2 = 8 separate checkpoints, plus the vanilla control). Every arm starts from the *same*
  released checkpoint and uses the same 3 training colors, train/val files,
  query-only MSE excluding action channels, 3 epochs, seed 0, and fixed batch
  order. The visual encoder and action/proprio encoders remain frozen in the
  new arms. `gradient_accumulation=4` with 8 samples/microbatch yields 32
  samples/optimizer update. The smaller microbatch is needed for ten frames
  per training example; B/C use four frames, so equal epochs do **not** imply
  equal optimizer updates or compute. Report this difference with results.
- Support has up to 8 completed transitions, from frames 0→1 through 7→8.
  Query is at frame 8; frame 9 is **only** the target. Random cycling over
  K=2/4/8 trains one checkpoint per architecture arm; K=0/2/4/8 are evaluated.
  At K=0, FiLM is identity even after training. The support mean is masked for
  partial online windows; it never includes imagined MPC futures. The optional
  residual uses a separately frozen copy of `F0`, not the trainable base.
- Colors of the PushT block: training `LightSlateGray`, `#003D99`, `#006B3C`;
  held-out **val-only** `Red` and `Purple`. The generator replays identical
  saved physical states and actions. It checks default re-render fidelity
  before training. No held-out color is used for weight selection.
- Offline diagnostics: same physical validation windows for each color,
  one-step and 3-step open-loop latent MSE, K=0/2/4/8, correct vs paired
  other-color vs transition-pair-shuffled vs identity context. Merely
  permuting the order of support transitions would be a no-op under mean
  pooling, so “shuffled” deliberately breaks `next`↔`previous/action` pairing.
- Online control: A/B/C, the matched-window vanilla, and every new architecture arm get **default and
  redBlock**, frozen-GD, seed 100, 50 common `val_T/plan_targets.pkl` targets,
  30 MPC steps. All eight new arms run the full K=0/2/4/8 sweep as separately
  resumable Slurm jobs, so the default matrix is computationally expensive
  (64 new-arm condition runs, plus the vanilla and A/B/C controls). Lower
  latent MSE alone is not a control improvement. W&B projects are `adajepa_transition_support`,
  `adajepa_transition_support_offline`, and
  `adajepa_transition_support_mpc` (A/B/C retain their existing projects).

## CIRCE preflight and submission

Use these literal commands in the CIRCE SSH shell, one line at a time. If
another checkout exists, `cd` there and `git pull --ff-only` instead of
cloning over it. The paths below reflect the established CIRCE layout but
must exist on the server. `CIRCE_PARTITION`/`CIRCE_QOS` are examples from the
previous allocation, not a claim that today's allocation is unchanged.

```bash
cd /home/j/jinwoo/adajepa-transition-film
git pull --ff-only origin transition-conditioned-film
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export DATASET_DIR=/home/j/jinwoo/adajepa/data_full
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export TARGETS=/home/j/jinwoo/adajepa/data/pushobj_eval/val_T/plan_targets.pkl
test -f "$SOURCE_MODEL_DIR/hydra.yaml" && echo 'source config OK'
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" && echo 'source checkpoint OK'
test -f "$DATASET_DIR/pusht_noise/train/rel_actions.pth" && echo 'dataset OK'
test -f "$TARGETS" && echo 'targets OK'
"$PYTHON_BIN" -m pytest tests/test_transition_film.py tests/test_support_transition.py tests/test_support_mpc_integration.py tests/test_support_train_integration.py -q
sinfo -h -o '%P %G' | grep muma_2021
```

Only after those checks succeed:

```bash
export CIRCE_PARTITION=muma_2021
export CIRCE_QOS=preempt_short
export CIRCE_TIME=2-00:00:00
export CIRCE_CPU_TIME=2-00:00:00
RUN_ROOT="$PWD/eval_outputs/transition_full_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_full_transition.bash "$RUN_ROOT"
```

This queues the old A/B/C jobs, val-only held-out-color rendering, 9 serial
GPU training jobs (to limit simultaneous filesystem/GPU pressure), each
arm's offline and full K×condition MPC jobs after its checkpoint, then aggregation. Each job
runs on a Slurm compute allocation. SSH/VPN/laptop disconnection **does not
cancel submitted jobs**. The output root is unique and persists on CIRCE.
This matrix can consume many GPU-days and substantial shared storage; inspect
the queue/allocation and free space before submitting it. The renderer checks
remaining disk against the *missing* output videos and skips completed ones.

## Check progress, stop, and resume

```bash
squeue -u "$USER"
cat "$RUN_ROOT/job_ids.txt"
sacct -j <JOBID> -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
tail -F "$RUN_ROOT/logs/train_support_delta_frozen_mlp-<JOBID>.out"
tail -n 4 "$RUN_ROOT/models/support_delta_frozen_mlp/metrics.jsonl"
ls -lh "$RUN_ROOT/models/support_delta_frozen_mlp/checkpoints/"
scancel <JOBID>
```

`train_transition_support.py` atomically saves `model_latest.pth` every 1000
training batches at an optimizer boundary, and `model_1.pth`/`model_2.pth`/
`model_3.pth` at completed epochs. Checkpoints include optimizer, scaler,
RNG, next batch, and partial epoch loss for deterministic local resumption.
W&B resumes the same run ID from `wandb_identity.json`. Restart only **after
the prior DAG has stopped** and only on the same Git commit, data and released
checkpoint; the submitter checks these invariants and protects each stage
with a lock:

```bash
RESUME=1 bash scripts/submit_full_transition.bash "$RUN_ROOT"
```

The Slurm `afterok` chain will hold dependent stages if a predecessor fails.
Inspect that job's log before re-submitting. Do not edit code or input data
under a running DAG. This workflow uses a re-runnable script, not an
unverified claim that Slurm will automatically retry every failure.

## Outputs

- `models/{B,C}` and `models/support_*`: checkpoint and training metrics.
- `offline/support_*/results.csv`: same-window color/K/support diagnostics.
- `eval/{A,B,C}` and `eval/support_*/k*/{default,redBlock}`: W&B/logs.json
  paper curves and per-attempt console logs.
- `summary/offline_all_ablation.csv`, `summary/mpc_success_curves.csv`, and
  `summary/success_rate_vs_mpc_step.png` / `offline_one_step_vs_support_k.png`:
  combined results.

No CIRCE job or expensive evaluation was launched by implementing this code
on Windows. The CPU model tests and script syntax can be checked locally;
checkpoint loading, rendering fidelity, training and MPC must be validated
on the actual CIRCE allocation before making performance claims.
