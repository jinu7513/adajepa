# Track A implementation validation

Date: 2026-09-27. Base commit: `66c5f18d00cc627ae35b9b183367f7e510972106`.
No full-scale training or real benchmark evaluation was launched.

Final command:

```text
.venv-tracka\Scripts\python.exe -m pytest tests/test_tracka.py -q --basetemp=tracka_outputs/pytest_verified
24 passed, 1 warning in 36.73s
```

Final local compatibility environment: Windows, Python 3.11.14, torch 2.3.0,
torchvision 0.18.0, numpy 1.26.4, OpenCV 4.10.0, hydra-core 1.3.2,
wandb 0.22.3, pymunk 6.6.0. Python 3.9 syntax was parsed explicitly; Python 3.9
runtime and CUDA execution were not available locally and remain unverified.

The single warning concerns unavailable point_maze/d4rl. PushT tests execute its
actual renderer and physics reset successfully without those optional dependencies.

Verified:

- Canonical salt/pepper probability, Gaussian/blur shape and determinism, historical
  snp1 output byte-for-byte, sigma-zero blur identity.
- Real colored PushT static resets, canonical state equality, wrapped angle
  comparison and color rejection constraints.
- Four-view generation, all three splits, stored canonical labels and integrity.
- Full default token dimensions `[B,196,384]`, masked token dimensions `[B,98,384]`,
  correct original-position gathering and raster token restoration.
- Influence from the last spatial patch to the first (bidirectional attention).
- All three per-update objectives under both decoder conditioning modes,
  reconstruction/invariance gradients, optional CLS/global branch, and existing
  VWorldModel encode_obs compatibility.
- All four training schedules, checkpoint dictionaries, exact CPU parameter equality
  for interrupted/resumed versus uninterrupted alternating training, both objectives
  logged under an even logging interval.
- Known-linear physical/RGB probe toy problems and full synthetic probe report
  generation, including default, random color, red interventions and severity grids.
- W&B offline metric files and custom chart/table artifacts, local flushing during
  active runs, failure exit status, simulated unreachable W&B with local fallback.

Additional CLI checks:

- Hydra train config resolved successfully.
- Generated and visually inspected 224px clean/color/corruption contact sheets,
  including low/mid/high corruption ranges with a 50% spatial mask.
- Two-step alternating training through `train_encoder.py` produced a checkpoint,
  live scalar records, reconstruction image and W&B offline run summary.
- `eval_encoder_probes.py` loaded that checkpoint and produced result JSON/CSV,
  saved probes and a figure.
- `aggregate_tracka_results.py` produced combined CSV and comparison PNG.
- `recover_tracka_wandb.py --mode offline` replayed 20 metric records, result and
  diagnostic tables, render bar chart and three corruption curves. Final recovered
  status was `finished`, `wandb_error: null`. W&B cache/staging initially failed in
  a restricted home directory; routing defaults into the output directory fixed it.
- `git diff --check` passed.

Not validated: real CIRCE dataset generation, representative real-data reset rejection
rates, GPU training, pretrained DINO weight loading, or authenticated W&B cloud
delivery. Offline artifacts prove local SDK serialization, not successful online sync.
See `tracka.md` for CIRCE commands and the scientific limitations of this experiment.
