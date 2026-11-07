# Online latent adapter MPC comparison

This experiment measures actual PushT MPC success. It uses the released gray
PushT checkpoint, one common `val_T/plan_targets.pkl`, seed 100, 50 targets,
GD with 100 optimization steps per replan, and up to 30 MPC replans. Default
and redBlock use the same physical targets; redBlock rerenders the moving T as
red at evaluation time. The four methods are:

| Method | Online trainable parameters | Initial state |
|---|---|---|
| frozen | none | released model |
| adajepa | predictor last transformer layer and encoder last projection | released model |
| latent_bias | 384-dimensional bias after the frozen visual encoder | zero |
| latent_mlp | 384-32-384 residual MLP after the frozen visual encoder | identity |

The adapter is applied to observed and goal visual latents, before the frozen
predictor; it does not reprocess predicted latents. Online adaptation uses the
same observed transition segments and one-step prediction loss as AdaJEPA,
plus `0.01` times an identity penalty on the adapter output. Each method uses
one optimizer update after each MPC replan, with `recent5` replay and learning
rate `5e-4`. Every target starts from the same initial adapter/model weights.
The `bias` and `MLP` do not use the privileged paired gray/red labels from
`probe_redblock_alignment.py`.

This is a controlled engineering comparison, not yet a paper-level multi-seed
estimate. The optimizer step is the same, but the trainable parameter counts
and regularizer differ. Report adaptation seconds from `per_iter_logs.json`
alongside success; do not infer speed solely from parameter count.

## CIRCE execution

Clone or update the dedicated `redblock-latent-adapter-probe` checkout after
the corresponding commit is pushed. Avoid pulling into a checkout used by
other active Slurm jobs. Within `/home/j/jinwoo/adajepa-redblock-probe`:

```bash
export REPO_DIR="$PWD"
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export TARGETS=/home/j/jinwoo/adajepa/data/pushobj_eval/val_T/plan_targets.pkl
"$PYTHON_BIN" -m pytest tests/test_latent_adapter_mpc.py -q
test -f "$SOURCE_MODEL_DIR/hydra.yaml" && echo "model config OK"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" && echo "model checkpoint OK"
test -f "$TARGETS" && echo "targets OK"
```

Submit a single serial GPU job for all eight cases. This keeps below CIRCE's
job submission limit and allows the same output root to resume after a
preemption. Keep `OUT_ROOT` to resume an incomplete matrix later.

```bash
export OUT_ROOT="$PWD/eval_outputs/redblock_adapter_$(date +%Y%m%d_%H%M%S)"
export SUBMITTED_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$OUT_ROOT/logs"
sbatch --job-name=redblock-adapter-mpc --partition=muma_2021 --qos=preempt_short \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00 \
  --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/slurm-%j.out" \
  --export=ALL "$REPO_DIR/scripts/slurm_latent_adapter_mpc.bash"
```

Check `squeue -j JOBID`, `sacct -j JOBID -X`, and
`tail -f "$OUT_ROOT/logs/slurm-JOBID.out"`. On success, inspect
`$OUT_ROOT/summary.csv` and `$OUT_ROOT/curves.csv`. Each run also keeps its
`logs.json`, `per_iter_logs.json`, console log and W&B run. If interrupted,
resubmit the same script with the *same* `OUT_ROOT` and `SUBMITTED_COMMIT`;
cases with `done.txt` are skipped. The script never edits the checkpoint.
