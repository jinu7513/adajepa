# Two independent JEPA experiments on CIRCE

This branch is `two-track-jepa`. It leaves the existing P/J experiments and their checkpoints intact. Track A changes only the appearance pairing in the original JEPA prediction loss. Track B trains on default appearance only and optionally adds an inverse-action auxiliary loss. Do not combine their flags or interpret Track B as a color-training experiment.

## Shared protocol

- Source: released PushT `SmallResNetGeM` encoder, predictor, action and proprio embedders from `SOURCE_MODEL_DIR/checkpoints/model_latest.pth`.
- Four observed frames per window, `num_hist=3`, `num_pred=1`, raw `frameskip=5`; three teacher-forced one-step latent targets.
- Same physical training windows and three completed epochs per arm; seed 0, batch 32, encoder and predictor trainable, encoder held in eval mode.
- Evaluation: Frozen GD, 50 paired PushT targets, seed 100, 30 MPC steps, default, purpleBlock, redBlock. Online adaptation is disabled.
- Actual physical-window and appearance counts are written by the real loader. A preempted partial epoch is replayed on epoch-level resume, and `sampling_progress.jsonl` records the extent of replay.

## One-time setup on CIRCE

Run these from a CIRCE login shell in the checkout of this branch. The source and dataset paths are the existing shared assets from the original checkout.

```bash
cd /home/j/jinwoo/adajepa-two-track
git branch --show-current
export REPO_DIR="$PWD"
export DATASET_DIR=/home/j/jinwoo/adajepa/data_full
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export TARGETS=/home/j/jinwoo/adajepa/data/pushobj_eval/val_T/plan_targets.pkl
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export OUT_ROOT="$PWD/two_track_outputs/seed0"
export SUBMITTED_COMMIT="$(git rev-parse HEAD)"
export TRAIN_SEED=0 TRAIN_EPOCHS=3 TRAIN_BATCH_SIZE=32
mkdir -p "$OUT_ROOT/logs"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
test -f "$DATASET_DIR/pusht_noise/train/states.pth"
test -f "$TARGETS"
"$PYTHON_BIN" -m pytest tests/test_two_track_jepa.py -q
```

The `sinfo` and `sacctmgr` output from this account previously supported `muma_2021` and `preempt_short`; confirm availability before submitting. Change the partition/QOS if CIRCE reports a different entitlement. Submit from a login node; Slurm runs the job on a GPU compute node.

## Data audit before training

```bash
"$PYTHON_BIN" -m scripts.audit_two_track_data --data-root "$DATASET_DIR/pusht_noise" --output "$OUT_ROOT/cross_color_jepa/audit" --track cross_color_jepa --mode cross_window_random --seed 0 --epochs 3 --batch-size 32
"$PYTHON_BIN" -m scripts.audit_two_track_data --data-root "$DATASET_DIR/pusht_noise" --output "$OUT_ROOT/mi_jepa/audit" --track mi_jepa --seed 0 --epochs 3 --batch-size 32
```

Check the generated `data_sampling_audit.json`, `data_sampling_audit.md`, and Track A `appearance_pair_counts.json`. Counts in the source report remain pending until these commands run against CIRCE data.

## Train independently

The four Track A arms are `same_fixed`, `same_window_random`, `cross_fixed`, and `cross_window_random`. The primary comparison is the two `window_random` arms. The two Track B arms are `default_jepa_baseline` and `default_jepa_plus_inverse`.

```bash
sbatch --job-name=two-track-A --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/A-cross_window_random-%j.out" --export=ALL scripts/slurm_two_track_train.bash cross_color_jepa cross_window_random
sbatch --job-name=two-track-B --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/B-default_jepa_plus_inverse-%j.out" --export=ALL scripts/slurm_two_track_train.bash mi_jepa default_jepa_plus_inverse
```

For controls, repeat the exact command with the arm and log basename changed to `same_window_random`, `same_fixed`, `cross_fixed`, or `default_jepa_baseline`. Avoid submitting all jobs at once if the account reaches `QOSMaxSubmitJobPerUserLimit`.

Each arm writes `hydra.yaml`, `provenance.txt`, `sampling_progress.jsonl`, `data_sampling_audit.json`, epoch metrics and `checkpoints/model_1.pth` through `model_3.pth` in `$OUT_ROOT/<track>/models/<arm>`. Confirm a completed `model_3.pth` before evaluation.

## Monitor and resume

```bash
squeue -u "$USER" -o '%.18i %.2t %.30j %.30R'
sacct -j JOB_ID -X --format=JobID,State,ExitCode,Elapsed
tail -n 40 "$OUT_ROOT/logs/A-cross_window_random-JOB_ID.out"
ls -lh "$OUT_ROOT/cross_color_jepa/models/cross_window_random/checkpoints/"
scancel JOB_ID
```

After preemption, rerun the same `sbatch` command with `RESUME=1` exported in the submitting shell. The job checks the saved arm, seed, batch and epoch budget before loading its last completed epoch. The interrupted epoch restarts from its beginning; compare `sampling_progress.jsonl` and attempt logs to report any extra window visits.

## Evaluate and aggregate

Run these only after the corresponding final checkpoint exists. Replace `cross_color_jepa cross_window_random` with another arm as needed; both tracks use the same evaluation script and paired targets.

```bash
sbatch --job-name=two-track-mpc --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/mpc-A-%j.out" --export=ALL scripts/slurm_two_track_eval.bash cross_color_jepa cross_window_random
sbatch --job-name=two-track-probe --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/probe-A-%j.out" --export=ALL scripts/slurm_two_track_probe.bash cross_color_jepa cross_window_random
sbatch --job-name=two-track-latent --partition=muma_2021 --qos=preempt_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/latent-A-%j.out" --export=ALL scripts/slurm_two_track_latent.bash cross_color_jepa cross_window_random
"$PYTHON_BIN" -m scripts.aggregate_two_track_mpc "$OUT_ROOT"
```

The aggregate writes **separate** `cross_color_jepa/mpc_summary.csv` and `mi_jepa/mpc_summary.csv`, 50-target counts and Wilson 95% intervals. Missing evaluations are shown as `N/A`. The probes write frozen pose/contact transfer and paired cosine metrics; latent evaluation writes teacher-forced and autoregressive horizon errors. These are diagnostic measurements, not substitutes for MPC success. No Track A/B results are claimed until the CIRCE jobs finish.
