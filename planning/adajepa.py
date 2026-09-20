import logging
import math

import torch
import torch.nn as nn

from utils import move_to_device

log = logging.getLogger(__name__)


class AdaJEPATrainer:
    """AdaJEPA trainer: test-time adaptation of the predictor (and optionally
    the encoder) on trajectory segments observed from the environment during
    planning. The default path uses the original sliding-window 1-step latent
    prediction loss; an opt-in path recursively evaluates configured horizons.
    Defaults follow the paper: one adaptation step on the predictor's last
    transformer layer and the encoder's head."""

    def __init__(
        self,
        wm,
        lr: float,
        steps: int = 1,
        optimizer_name: str = "adam",
        finetune_encoder: bool = True,
        last_layer_only: bool = True,
        encoder_lr: float = None,
        encoder_last_layer_only: bool = True,
        encoder_adapt_mode: str = None,
        multi_step_adaptation: bool = False,
        adapt_horizons=None,
        horizon_weights=None,
    ):
        self.wm = wm
        self.lr = lr
        self.encoder_lr = encoder_lr if encoder_lr is not None else lr
        self.steps = steps
        self.optimizer_name = optimizer_name.lower()
        self._legacy_encoder_selection = encoder_adapt_mode is None
        self.encoder_adapt_mode = self._resolve_encoder_adapt_mode(
            encoder_adapt_mode, finetune_encoder, encoder_last_layer_only
        )
        self.finetune_encoder = self.encoder_adapt_mode != "freeze_all"
        self.last_layer_only = last_layer_only
        self.encoder_last_layer_only = encoder_last_layer_only
        self.multi_step_adaptation = bool(multi_step_adaptation)
        self.adapt_horizons, self.horizon_weights = self._validate_horizons(
            adapt_horizons, horizon_weights
        )
        self.device = next(wm.parameters()).device
        self.criterion = nn.MSELoss()
        self._ada_predictor_params = self._select_predictor_params()
        self._ada_encoder_params = self._select_encoder_params() if self.finetune_encoder else []
        self._set_requires_grad(self.wm.predictor, self._ada_predictor_params, False)
        self._set_requires_grad(self.wm.encoder, self._ada_encoder_params, False)
        self.encoder_stats = self._encoder_parameter_stats()
        self.configuration = {
            "encoder_adapt_mode": self.encoder_adapt_mode,
            "multi_step_adaptation": self.multi_step_adaptation,
            "adapt_horizons": list(self.adapt_horizons),
            "horizon_weights": list(self.horizon_weights),
            "encoder_total_params": self.encoder_stats["adajepa/encoder_total_params"],
            "encoder_trainable_params": self.encoder_stats[
                "adajepa/encoder_trainable_params"
            ],
            "encoder_trainable_modules": list(
                getattr(self, "_ada_encoder_block_names", [])
            ),
        }
        self.last_metrics = dict(self.encoder_stats)
        self._warned_short_horizons = set()
        self._snapshot = self._take_snapshot()

    @staticmethod
    def _resolve_encoder_adapt_mode(mode, finetune_encoder, legacy_last_only):
        # Keep the legacy kill switch authoritative for old CLI overrides.
        if not finetune_encoder:
            return "freeze_all"
        if mode is None:
            return "last_block" if legacy_last_only else "full_encoder"
        allowed = {"freeze_all", "last_block", "last_2_blocks", "full_encoder"}
        mode = str(mode)
        if mode not in allowed:
            raise ValueError(
                f"Unknown encoder_adapt_mode {mode!r}; choose from {sorted(allowed)}"
            )
        return mode

    @staticmethod
    def _validate_horizons(horizons, weights):
        horizons = [1] if horizons is None else list(horizons)
        weights = [1.0] if weights is None else list(weights)
        if len(horizons) != len(weights):
            raise ValueError(
                "adapt_horizons and horizon_weights must have the same length; "
                f"got {len(horizons)} and {len(weights)}"
            )
        if not horizons:
            raise ValueError("adapt_horizons must contain at least one horizon")
        if any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
            raise ValueError(f"adapt_horizons must be positive integers; got {horizons!r}")
        if len(set(horizons)) != len(horizons):
            raise ValueError(f"adapt_horizons must be unique; got {horizons!r}")
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
            raise ValueError(
                f"horizon_weights must be finite and non-negative; got {weights!r}"
            )
        if not any(float(w) > 0 for w in weights):
            raise ValueError("at least one horizon weight must be greater than zero")
        return horizons, [float(w) for w in weights]

    def _take_snapshot(self):
        """Snapshot the tensors adaptation can touch: selected params + train-mode buffers (e.g. BN stats)."""
        tensors = list(self._ada_predictor_params)
        tensors += [b for _, b in self.wm.predictor.named_buffers()]
        if self.finetune_encoder:
            tensors += self._ada_encoder_params
            tensors += [b for _, b in self.wm.encoder.named_buffers()]
        return [(t, t.detach().clone()) for t in tensors]

    @torch.no_grad()
    def reset(self):
        """Restore the pre-adaptation values of the adapted tensors."""
        for tensor, saved in self._snapshot:
            tensor.copy_(saved)

    def finetune(self, obs_seqs: list, act_seqs: list, merge: bool = True) -> list:
        """Run `steps` optimization steps on the given trajectory segments.

        merge=True concatenates temporally contiguous segments into one long
        sequence; use merge=False for non-contiguous segments.
        Returns the per-step prediction losses.
        """
        if not obs_seqs or self.steps <= 0:
            return []
        if merge and len(obs_seqs) > 1:
            obs_seqs, act_seqs = self._merge_segments(obs_seqs, act_seqs)
        segments = [self._prepare_segment(o, a) for o, a in zip(obs_seqs, act_seqs)]
        if not self.finetune_encoder:
            # Encoder frozen: embeddings can be precomputed once.
            with torch.no_grad():
                all_z = [self.wm.encode(o, a).detach() for o, a in segments]

        self._set_requires_grad(self.wm.predictor, self._ada_predictor_params, True)
        self.wm.predictor.train()
        if self.finetune_encoder:
            self._set_requires_grad(self.wm.encoder, self._ada_encoder_params, True)
            self.wm.encoder.train()
            base_model = getattr(self.wm.encoder, "base_model", None)
            if base_model is not None:
                base_model.eval()  # keep the frozen backbone in eval mode

        optimizer = self._make_optimizer()
        detach_src = not self.finetune_encoder
        detach_tgt = True if not self.finetune_encoder else bool(getattr(self.wm, "stop_grad", True))
        step_losses = []
        for step in range(self.steps):
            optimizer.zero_grad()
            if self.finetune_encoder:
                # Re-encode each step so gradients reach the encoder.
                all_z = [self.wm.encode(o, a) for o, a in segments]
            if self.multi_step_adaptation:
                segment_results = [
                    self._multi_step_prediction_loss(
                        z, detach_src=detach_src, detach_tgt=detach_tgt
                    )
                    for z in all_z
                ]
                loss = torch.stack([result[0] for result in segment_results]).mean()
                horizon_losses = {}
                for horizon in self.adapt_horizons:
                    available = [
                        result[1][horizon]
                        for result in segment_results
                        if horizon in result[1]
                    ]
                    if available:
                        horizon_losses[horizon] = torch.stack(available).mean()
            else:
                loss = torch.stack(
                    [
                        self._prediction_loss(
                            z, detach_src=detach_src, detach_tgt=detach_tgt
                        )
                        for z in all_z
                    ]
                ).mean()
                horizon_losses = {1: loss}
            loss.backward()
            encoder_grad_norm = self._grad_norm(self._ada_encoder_params)
            predictor_grad_norm = self._grad_norm(self._ada_predictor_params)
            optimizer.step()
            step_losses.append(loss.item())
            self.last_metrics = {
                **self.encoder_stats,
                "adajepa/total_adaptation_loss": float(loss.detach()),
                "adajepa/encoder_grad_norm": encoder_grad_norm,
                "adajepa/predictor_grad_norm": predictor_grad_norm,
                **{
                    f"adajepa/horizon_{h}_loss": float(value.detach())
                    for h, value in horizon_losses.items()
                },
            }
            log.info("AdaJEPA step %d/%d  pred_loss=%.6f", step + 1, self.steps, step_losses[-1])

        self.wm.predictor.eval()
        self._set_requires_grad(self.wm.predictor, self._ada_predictor_params, False)
        if self.finetune_encoder:
            self.wm.encoder.eval()
            self._set_requires_grad(self.wm.encoder, self._ada_encoder_params, False)
        return step_losses

    @torch.no_grad()
    def score_segments(self, obs_seqs: list, act_seqs: list) -> list:
        """Prediction loss per segment under the current weights (higher = harder)."""
        return [
            float(self._prediction_loss(self.wm.encode(*self._prepare_segment(o, a))))
            for o, a in zip(obs_seqs, act_seqs)
        ]

    def _select_predictor_params(self):
        predictor = self.wm.predictor
        if self.last_layer_only:
            params = list(predictor.transformer.layers[-1].parameters())
            params += list(predictor.transformer.norm.parameters())
            log.info(
                "AdaJEPA predictor adaptation restricted to last transformer layer (%d tensors).",
                len(params),
            )
            return params
        return list(predictor.parameters())

    def _select_encoder_params(self):
        """Select parameter-bearing top-level blocks in their forward order."""
        encoder = self.wm.encoder
        if self._legacy_encoder_selection:
            return self._select_encoder_params_legacy(encoder)
        blocks = [
            (name, module)
            for name, module in encoder.named_children()
            if any(True for _ in module.parameters())
            and not (hasattr(encoder, "base_model") and name == "base_model")
        ]
        if self.encoder_adapt_mode == "full_encoder":
            selected = blocks
            params = list(encoder.parameters())
        else:
            depth = 1 if self.encoder_adapt_mode == "last_block" else 2
            if len(blocks) < depth:
                raise ValueError(
                    f"encoder_adapt_mode={self.encoder_adapt_mode!r} requires {depth} "
                    f"parameter-bearing blocks, but {type(encoder).__name__} has "
                    f"only {[name for name, _ in blocks]}"
                )
            selected = blocks[-depth:]
            params = [p for _, module in selected for p in module.parameters()]
        self._ada_encoder_block_names = [name for name, _ in selected]
        log.info(
            "AdaJEPA encoder mode=%s; trainable blocks=%s (%d tensors).",
            self.encoder_adapt_mode,
            self._ada_encoder_block_names,
            len(params),
        )
        return params

    def _select_encoder_params_legacy(self, encoder):
        """Reproduce the pre-mode selector for configs that omit the new flag."""
        if hasattr(encoder, "base_model"):
            if hasattr(encoder, "projector") and getattr(
                encoder, "projector_name", None
            ) in ("channel", "global"):
                self._ada_encoder_block_names = ["projector"]
                return list(encoder.projector.parameters())
            params = [
                p
                for name, p in encoder.named_parameters()
                if not name.startswith("base_model.")
            ]
            self._ada_encoder_block_names = sorted(
                {
                    name.split(".", 1)[0]
                    for name, _ in encoder.named_parameters()
                    if not name.startswith("base_model.")
                }
            )
            return params
        if self.encoder_last_layer_only:
            children = list(encoder.named_children())
            if children:
                name, module = children[-1]
                self._ada_encoder_block_names = [name]
                return list(module.parameters())
        self._ada_encoder_block_names = [
            name
            for name, module in encoder.named_children()
            if any(True for _ in module.parameters())
        ]
        return list(encoder.parameters())

    def _encoder_parameter_stats(self):
        total = sum(p.numel() for p in self.wm.encoder.parameters())
        trainable = sum(p.numel() for p in self._ada_encoder_params)
        pct = 100.0 * trainable / total if total else 0.0
        names = getattr(self, "_ada_encoder_block_names", [])
        log.info(
            "AdaJEPA encoder parameters: trainable=%d / total=%d (%.2f%%); modules=%s",
            trainable,
            total,
            pct,
            names,
        )
        return {
            "adajepa/encoder_total_params": total,
            "adajepa/encoder_trainable_params": trainable,
            "adajepa/encoder_trainable_pct": pct,
        }

    @staticmethod
    def _grad_norm(params):
        squared = [p.grad.detach().pow(2).sum() for p in params if p.grad is not None]
        if not squared:
            return 0.0
        return float(torch.stack(squared).sum().sqrt())

    @staticmethod
    def _set_requires_grad(module, ada_params, enabled: bool):
        """Freeze all params of `module`; if enabled, re-enable the adapted subset."""
        for p in module.parameters():
            p.requires_grad_(False)
        if enabled:
            for p in ada_params:
                p.requires_grad_(True)

    def _make_optimizer(self):
        param_groups = [{"params": self._ada_predictor_params, "lr": self.lr}]
        if self.finetune_encoder:
            param_groups.append({"params": self._ada_encoder_params, "lr": self.encoder_lr})
            log.info("AdaJEPA optimizer: predictor_lr=%.2e  encoder_lr=%.2e", self.lr, self.encoder_lr)
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        if self.optimizer_name not in optimizers:
            raise ValueError(f"Unknown AdaJEPA optimizer: {self.optimizer_name!r}")
        return optimizers[self.optimizer_name](param_groups)

    def _prepare_segment(self, obs, act):
        """Move one (obs, act) segment to device and pad a dummy action for the
        last frame so encode() sees T+1 (frame, action) pairs. The dummy action
        never enters the prediction loss: action dims are excluded from the loss
        and the last frame never appears in a source window."""
        obs = move_to_device({k: v.clone() for k, v in obs.items()}, self.device)
        act = act.to(self.device)
        act = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
        return obs, act

    @staticmethod
    def _merge_segments(obs_seqs, act_seqs):
        """Concatenate contiguous segments; the last frame of segment i equals
        the first frame of segment i+1, so the duplicate frame is dropped."""
        obs = {k: v.clone() for k, v in obs_seqs[0].items()}
        act = act_seqs[0]
        for obs_i, act_i in zip(obs_seqs[1:], act_seqs[1:]):
            for k in obs:
                obs[k] = torch.cat([obs[k], obs_i[k][:, 1:]], dim=1)
            act = torch.cat([act, act_i], dim=1)
        return [obs], [act]

    def _prediction_loss(
        self,
        z: torch.Tensor,
        detach_src: bool = True,
        detach_tgt: bool = True,
    ) -> torch.Tensor:
        """Sliding-window 1-step MSE on obs tokens (visual+proprio, no action).

        z: (b, T+1, p, d) embeddings of T+1 frames.
        """
        T = z.shape[1] - 1
        if T < 1:
            return torch.tensor(0.0, device=self.device)
        window = min(self.wm.num_hist, T)
        losses = []
        for t in range(T - window + 1):
            z_src = z[:, t : t + window]
            z_tgt = z[:, t + 1 : t + 1 + window]
            if detach_src:
                z_src = z_src.detach()
            if detach_tgt:
                z_tgt = z_tgt.detach()
            z_pred = self.wm.predict(z_src)
            if self.wm.concat_dim == 0:
                loss = self.criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :])
            else:
                drop = self.wm.action_dim
                loss = self.criterion(z_pred[:, :, :, :-drop], z_tgt[:, :, :, :-drop])
            losses.append(loss)
        return torch.stack(losses).mean()

    def _obs_loss(self, prediction, target):
        if self.wm.concat_dim == 0:
            return self.criterion(prediction[:, :, :-1, :], target[:, :, :-1, :])
        drop = self.wm.action_dim
        return self.criterion(prediction[..., :-drop], target[..., :-drop])

    def _replace_encoded_action(self, prediction, target):
        """Inject the known future action embedding before recursive prediction."""
        if self.wm.concat_dim == 0:
            return torch.cat(
                [prediction[:, :, :-1, :], target[:, :, -1:, :]], dim=2
            )
        drop = self.wm.action_dim
        return torch.cat([prediction[..., :-drop], target[..., -drop:]], dim=-1)

    def _multi_step_prediction_loss(
        self,
        z: torch.Tensor,
        detach_src: bool = True,
        detach_tgt: bool = True,
    ):
        """Recursively predict requested future horizons from valid history windows."""
        transition_count = z.shape[1] - 1
        if transition_count < 1:
            raise ValueError(
                "multi-step adaptation needs at least one transition (two frames)"
            )
        per_horizon = {}
        for horizon, weight in zip(self.adapt_horizons, self.horizon_weights):
            losses = []
            last_source_t = transition_count - horizon
            for source_t in range(last_source_t + 1):
                first_context_t = max(0, source_t - self.wm.num_hist + 1)
                context = z[:, first_context_t : source_t + 1]
                if detach_src:
                    context = context.detach()
                prediction = None
                for rollout_step in range(1, horizon + 1):
                    prediction = self.wm.predict(context[:, -self.wm.num_hist :])[:, -1:]
                    target_index = source_t + rollout_step
                    if rollout_step < horizon:
                        prediction = self._replace_encoded_action(
                            prediction, z[:, target_index : target_index + 1]
                        )
                        context = torch.cat([context, prediction], dim=1)
                target = z[:, source_t + horizon : source_t + horizon + 1]
                if detach_tgt:
                    target = target.detach()
                losses.append(self._obs_loss(prediction, target))
            if losses:
                per_horizon[horizon] = torch.stack(losses).mean()
            elif horizon not in self._warned_short_horizons:
                log.warning(
                    "Skipping adaptation horizon %d: segment has only %d transitions.",
                    horizon,
                    transition_count,
                )
                self._warned_short_horizons.add(horizon)
        if not per_horizon:
            raise ValueError(
                "none of adapt_horizons can be evaluated for a trajectory with "
                f"{transition_count} transitions"
            )
        total = sum(
            weight * per_horizon[horizon]
            for horizon, weight in zip(self.adapt_horizons, self.horizon_weights)
            if horizon in per_horizon
        )
        return total, per_horizon
