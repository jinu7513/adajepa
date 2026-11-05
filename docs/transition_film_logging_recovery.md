# B/C three-epoch recovery after Tensor JSON logging failure

The October 27 A/B/C Slurm matrix at commit `612b0b3` finished its color-data
preparation and A evaluation, but B and C stopped after the first epoch's
training/validation when `json.dumps(epoch_log)` received a Tensor. The old
`train.py` calls `logs_flash()` before `save_ckpt()`, and the submitted jobs
used `training.save_every_x_iterations=0`. Therefore that run has no B/C
training checkpoint to resume. W&B history is not a model checkpoint.

This branch converts each metric to a Python float in `Trainer.logs_update`.
`scripts/submit_film_bc_train.bash` submits **only B and C** as separate GPU
Slurm jobs for three epochs. It reuses the already prepared paired-color
dataset and the same released encoder/predictor checkpoint, seed, batch size,
data loader workers, and trainability flags as the original A/B/C matrix.
It does not rerun the five-hour preparation or the completed A evaluation.

Use a **new CIRCE checkout** for this branch. Do not pull it into the old
`adajepa-transition-film` checkout while support jobs from that root may
still read it and verify commit `612b0b3`.

```bash
cd /home/j/jinwoo
cd /home/j/jinwoo/adajepa-transition-film
REPO_URL="$(git config --get remote.origin.url)"
cd /home/j/jinwoo
git clone --branch transition-film-logging-fix "$REPO_URL" \
  adajepa-transition-film-logging-fix
cd /home/j/jinwoo/adajepa-transition-film-logging-fix
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export DATASET_DIR=/home/j/jinwoo/adajepa/data_full
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export CIRCE_PARTITION=muma_2021
export CIRCE_QOS=preempt_short
export CIRCE_TIME=2-00:00:00
"$PYTHON_BIN" -m pytest tests/test_train_epoch_logging.py -q
test -f "$DATASET_DIR/pusht_noise/transition_color_manifest.json"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth"
RUN_ROOT="$PWD/eval_outputs/transition_film_bc_3epoch_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_film_bc_train.bash "$RUN_ROOT"
```

The partition/QoS pair must still be permitted by the current CIRCE
allocation. If the per-user Slurm submission limit is reached, inspect
`$RUN_ROOT/job_ids.txt`; a successfully submitted B job is recorded there
and will not be duplicated when the submit command is retried later. Slurm
jobs keep running after SSH disconnect. Check progress with:

```bash
cat "$RUN_ROOT/job_ids.txt"
squeue -u "$USER"
tail -F "$RUN_ROOT/logs/train_B-<JOBID>.out"
tail -F "$RUN_ROOT/logs/train_C-<JOBID>.out"
ls -lh "$RUN_ROOT/models/B/checkpoints/model_3.pth" \
       "$RUN_ROOT/models/C/checkpoints/model_3.pth"
```

The old root's A default/redBlock MPC runs remain the baseline. B/C
deployment evaluation should use the same 50 targets, seed 100, GD planner,
30 MPC steps, and default/redBlock conditions after both `model_3.pth`
files exist. Do not interpret first-epoch validation loss as MPC success.
