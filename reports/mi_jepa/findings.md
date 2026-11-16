# Track B: MI-inspired JEPA with same-color observations

## Hypothesis and isolated change

Can an inverse-dynamics auxiliary objective improve the utility of a default-color AdaJEPA encoder? Both arms use the original **default-color observation videos** for input and target; no paired-color training or cross-color loss exists in this track. Both retain the same original JEPA predictor and loss.

The extra head in `default_jepa_plus_inverse` receives **real observed** endpoint visual latents from the shared encoder. For model-step horizons `H=1,2,3` (each step spans five raw actions), it predicts all intervening normalized action chunks from the first frame and the actual frame `H` steps later. It uses a small two-layer MLP with a fixed unit-variance Gaussian NLL (constant omitted). The configured pilot coefficient is `lambda_inv=0.1`; the matching `default_jepa_baseline` sets it to zero and creates no head. The inverse head is training-only and does not enter MPC.

```text
L_total = L_original_JEPA + lambda_inv * mean_{H=1,2,3} [0.5 * MSE(q_H(E(o_0), E(o_H)), A_{0:H})]
```

The inverse branch permits gradients to both real endpoint encodings and its head. JEPA's future target remains stop-gradient. This inverse-action reconstruction is **MI-inspired**, not an exact conditional mutual-information objective, certified lower bound, or empowerment estimator: minimizing `H(A|Z,Z')` alone does not control `H(A|Z)` as the encoder changes. Different actions can lead to the same endpoint and the head may learn agent motion rather than block contact. The default-only baseline and frozen pose/contact transfer are therefore necessary.

## Evidence and interpretation status

| Arm | Seed | Checkpoint | Default MPC | Purple MPC | Red MPC | Horizon 1–5 error | Frozen pose/contact transfer | Collapse |
|---|---:|---|---:|---:|---:|---|---|---|
| default_jepa_baseline | 0 planned | pending | N/A | N/A | N/A | N/A | N/A | N/A |
| default_jepa_plus_inverse | 0 planned | pending | N/A | N/A | N/A | N/A | N/A | N/A |

No Track B training or evaluation has been executed in this worktree. Future purple/red evaluation tests OOD appearance transfer even though training is default-only. It is not a combined cross-color experiment. The inverse head increases parameter count and encoder compute; record wall time and peak GPU memory before interpreting comparisons.
