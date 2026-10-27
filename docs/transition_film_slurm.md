# CIRCE Slurm workflow for the transition-FiLM PushT comparison

This branch implements the first-pass A/B/C matrix in
`docs/transition_conditioned_film_spec.md`. **No job is submitted by pulling
the branch**. The commands below submit GPU work to Slurm compute nodes.

The released `checkpoints/pusht_visual_shift` model is A. B and C start from
its *same* encoder/predictor/action/proprio weights. Both freeze the visual
encoder and train for three complete epochs on identical physical PushT
trajectories and deterministic default/blue/green color draws. B has the
unmodified predictor; C adds transition-FiLM. Red block is evaluation-only.
The primary evaluation is frozen (no online gradient updates), GD, seed 100,
50 targets, 30 MPC steps, default and redBlock. The checked-in eval config
defaults to 20 steps, so the job makes the agreed 30-step override explicit.

## One-time preflight on CIRCE

Use a separate checkout so switching branches cannot affect existing Event
jobs still reading `/home/j/jinwoo/adajepa`. The clone does not duplicate
the large dataset or released checkpoint. If the destination already exists,
inspect it rather than cloning over it.

```bash
git clone --branch transition-conditioned-film git@github.com:jinu7513/adajepa.git /home/j/jinwoo/adajepa-transition-film
cd /home/j/jinwoo/adajepa-transition-film
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export DATASET_DIR=/home/j/jinwoo/adajepa/data_full
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export TARGETS=/home/j/jinwoo/adajepa/data/pushobj_eval/val_T/plan_targets.pkl
test -f "$SOURCE_MODEL_DIR/hydra.yaml"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
test -f "$DATASET_DIR/pusht_noise/train/rel_actions.pth"
test -f "$TARGETS"
"$PYTHON_BIN" -m pytest tests/test_transition_film.py -q
```

For an existing separate checkout, use
`git pull --ff-only origin transition-conditioned-film` there instead of
cloning again. Leave the original Event checkout on its current branch.

If any check fails, do not submit. Inspect the checkpoint/dataset paths first.
The data-preparation job also re-renders the saved physical states; it stops
if default re-renders disagree strongly with the source MP4s. The preview
contact sheets and report are placed in `data_full/pusht_noise/{train,val}/`
and `data_full/pusht_noise/transition_color_preview_report.json`.

## Submit the complete DAG

Use only a partition/QoS combination permitted for your CIRCE allocation;
the values below matched previous jobs but must be rechecked if permissions
change. A unique run root prevents overwriting existing results.

```bash
cd /home/j/jinwoo/adajepa-transition-film
export CIRCE_PARTITION=muma_2021
export CIRCE_QOS=preempt_short
export CIRCE_TIME=2-00:00:00
export CIRCE_CPU_TIME=2-00:00:00
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export DATASET_DIR=/home/j/jinwoo/adajepa/data_full
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export TARGETS=/home/j/jinwoo/adajepa/data/pushobj_eval/val_T/plan_targets.pkl
RUN_ROOT="$PWD/eval_outputs/transition_film_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_transition_film.bash "$RUN_ROOT"
```

The submitter checks the released checkpoint, full PushT train/val data,
evaluation targets, Python and partition *before* queueing jobs. It submits
seven jobs, recording each ID as soon as Slurm accepts it in
`$RUN_ROOT/job_ids.txt` and a timestamped per-submission file:

```text
prepare (CPU: unit tests, same-state previews, paired-color MP4 generation)
  +-- train_B (GPU, 3 epochs) --> eval_B (GPU, default + redBlock) --+
  +-- train_C (GPU, 3 epochs) --> eval_C (GPU, default + redBlock) --+--> aggregate
  +-- eval_A (GPU, released checkpoint, default + redBlock) --------+
```

Slurm `afterok` dependencies prevent an evaluation from starting after failed
data preparation or training. B/C may run concurrently on separate allocated
GPUs; Slurm queues them if resources are unavailable. Each training job saves
after every *complete* epoch. If preempted, requeue/restaging resumes from
the latest completed epoch; no mid-epoch cursor is claimed. Evaluation
attempts use distinct folders and write a done marker only after the paper
success-rate curve has been logged.

## Status, logs, cancellation, resume

```bash
squeue -u "$USER"
cat "$RUN_ROOT/job_ids.txt"
tail -F "$RUN_ROOT/logs/train_B-<JOBID>.out"
sacct -j <JOBID> -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
scancel <JOBID>
```

To re-submit missing/failed stages after reconnecting, use the **same** run
root with `RESUME=1`. The submitter rejects a changed code commit, dataset
path, source training files, released checkpoint/config or evaluation targets.
It does not delete previous
checkpoints/logs. Re-submit only after the original DAG has stopped; a
running copy of the same stage holds a lock and will reject a duplicate:

```bash
RESUME=1 bash scripts/submit_transition_film.bash "$RUN_ROOT"
```

Results are in `$RUN_ROOT/models/{B,C}/`, `$RUN_ROOT/eval/{A,B,C}/`, and
`$RUN_ROOT/aggregate/{success_curve.csv,final_success.csv,success_vs_mpc_step.png}`.
The A/B/C planning runs also log `paper/mpc_step` and
`paper/success_rate_pct` to W&B project `adajepa_transition_film_mpc`.

Limitations: full training, re-render fidelity and W&B synchronization cannot
be verified on a machine without CIRCE's checkpoint/dataset/GPU. The 50
episodes at one seed are a first-pass comparison, not a multi-seed confidence
interval or a claim that transition evidence (rather than current appearance)
caused an improvement. The code intentionally does not schedule blur, CEM or
test-time AdaJEPA-gradient ablations until the primary comparison finishes.
