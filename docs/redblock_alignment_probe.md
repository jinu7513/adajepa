# Test whether redBlock's latent shift is repairable after the frozen encoder

This is a **diagnostic**, not a new MPC method or a redBlock test result. The
released PushT checkpoint contains a frozen visual encoder trained on gray
moving T blocks. `scripts/probe_redblock_alignment.py` checks whether its
features for red moving T blocks still contain block pose, and whether a
small residual adapter can align them to gray features. The paper suggests
that color may be used to distinguish the moving block from the fixed green
anchor; this test examines a necessary condition for repairing that shift
with an encoder-output adapter.

## What is measured

The script samples one frame per episode from existing PushT **train** and
**val** physical states, renders each state twice with only the moving block
color changed, and encodes both frames with the released frozen encoder.
It never reads the final 50 MPC target file and never trains the encoder or
predictor. Check episode provenance separately before treating val metrics
as statistically independent of that target set. It writes
`paired_render_preview.png` for visual review.

1. Raw same-state gray/red latent MSE and cross-color nearest-neighbor
   retrieval on held-out val states.
2. A fixed linear pose probe trained on gray train features, tested on gray
   and red val features. A separate red-trained probe tests whether pose
   information remains linearly available in red features. Pose labels are
   used **only for these diagnostics**.
3. A simple mean latent shift and a 384→32→384 identity-initialized residual
   MLP trained on paired **train** latents. Both are privileged **oracle
   diagnostics** because gray/red pairs are unavailable to online MPC. Their
   held-out val alignment and pose metrics test whether a small mapping is
   feasible. The oracle is not the proposed self-supervised online adapter.

Interpretation: if the red-trained probe is reasonably accurate while the
gray-trained probe fails on red, geometry may still be present but the
coordinate system has shifted. If the oracle corrects held-out val features
without destroying gray features, a small post-encoder adapter is plausible.
If only the oracle works, the difficult part is finding its mapping from
short online transitions. If neither probe nor oracle works, investigate
the encoder, rendering, sample size, and probe capacity before claiming that
information is irretrievably lost. Lower latent MSE alone is not evidence of
better MPC planning.

## CIRCE run (new checkout; leave existing MPC jobs alone)

The older CIRCE checkout
`/home/j/jinwoo/adajepa-transition-film-logging-fix` may still be used by
Slurm jobs that check its Git commit. **Do not pull into it.** Clone this
branch into a separate checkout. CIRCE's old Git does not support `git -C`.

```bash
cd /home/j/jinwoo
git clone -b redblock-latent-adapter-probe https://github.com/jinu7513/adajepa.git adajepa-redblock-probe
cd /home/j/jinwoo/adajepa-redblock-probe
git rev-parse --short HEAD
```

Check the actual paths; the large inputs are outside Git:

```bash
export REPO_DIR="$PWD"
export PYTHON_BIN=/home/j/jinwoo/miniforge3/envs/ts/bin/python
export SOURCE_MODEL_DIR=/home/j/jinwoo/adajepa/checkpoints/pusht_visual_shift
export DATA_ROOT=/home/j/jinwoo/adajepa/data_full/pusht_noise
test -f "$SOURCE_MODEL_DIR/hydra.yaml" && echo "config OK"
test -f "$SOURCE_MODEL_DIR/checkpoints/model_latest.pth" && echo "checkpoint OK"
test -f "$DATA_ROOT/train/states.pth" && echo "train states OK"
test -f "$DATA_ROOT/val/states.pth" && echo "val states OK"
"$PYTHON_BIN" -m pytest tests/test_redblock_alignment_probe.py -q
```

Submit one small GPU job to the same partition/QoS pair previously accepted
by the user's CIRCE allocation. The job uses 256 train and 128 val states,
stores all metrics locally, and logs them to W&B project
`adajepa_transition_film_diagnostics`.

```bash
export OUT_ROOT="$PWD/diagnostics/redblock_$(date +%Y%m%d_%H%M%S)"
export SUBMITTED_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$OUT_ROOT/logs"
sbatch --job-name=redblock-probe --partition=muma_2021 --qos=preempt_short \
  --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --chdir="$REPO_DIR" --output="$OUT_ROOT/logs/slurm-%j.out" \
  --export=ALL "$REPO_DIR/scripts/slurm_redblock_probe.bash"
```

After Slurm completes, inspect
`$OUT_ROOT/analysis/results.json`, `$OUT_ROOT/analysis/paired_render_preview.png`,
and `$OUT_ROOT/logs/slurm-<JOBID>.out`. Preserve the output directory and
paste the `metrics` object from `results.json` to compare the three cases.
The experiment does **not** perform MPC or online adaptation. Those should be
implemented and evaluated only after this feasibility check is interpreted.
