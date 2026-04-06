"""Baseline sequence classifier (flatten + MLP)."""

from __future__ import annotations


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from src.models.encoder import MLPEncoder


if nn is not None:

    class BaselineMLP(nn.Module):
        """Baseline MLP that predicts the next-candle direction logit."""

        def __init__(
            self,
            *,
            window_size: int,
            num_features: int,
            hidden_dims: tuple[int, ...] = (256, 128),
            dropout: float = 0.1,
            multitask_output: bool = False,
            probabilistic_output: bool = False,
            include_rank_score: bool = False,
            include_regime_logits: bool = False,
            regime_classes: int = 3,
            distribution: str = "gaussian",
        ) -> None:
            super().__init__()
            if window_size <= 0 or num_features <= 0:
                raise ValueError("window_size and num_features must be > 0")

            input_dim = window_size * num_features
            self.encoder = MLPEncoder(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
            self.window_size = window_size
            self.num_features = num_features
            self.multitask_output = bool(multitask_output)
            self.probabilistic_output = bool(probabilistic_output)
            self.include_rank_score = bool(include_rank_score)
            self.include_regime_logits = bool(include_regime_logits)
            self.distribution = distribution
            self.direction_head = nn.Linear(self.encoder.output_dim, 1)
            self.mean_head = nn.Linear(self.encoder.output_dim, 1)
            self.log_scale_head = nn.Linear(self.encoder.output_dim, 1)
            self.threshold_head = nn.Linear(self.encoder.output_dim, 3)
            self.rank_head = nn.Linear(self.encoder.output_dim, 1) if include_rank_score else None
            self.regime_head = nn.Linear(self.encoder.output_dim, regime_classes) if include_regime_logits else None

        def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
            flattened = inputs.reshape(inputs.size(0), -1)
            encoded = self.encoder(flattened)
            direction_logit = self.direction_head(encoded).squeeze(-1)
            if not self.multitask_output:
                return direction_logit

            outputs: dict[str, torch.Tensor] = {
                "direction_logit": direction_logit,
                "threshold_logits": self.threshold_head(encoded),
            }
            if self.probabilistic_output:
                outputs["mean_return"] = self.mean_head(encoded).squeeze(-1)
                outputs["log_scale"] = self.log_scale_head(encoded).squeeze(-1)
            if self.rank_head is not None:
                outputs["rank_score"] = self.rank_head(encoded).squeeze(-1)
            if self.regime_head is not None:
                outputs["regime_logits"] = self.regime_head(encoded)
            return outputs

else:

    class BaselineMLP:  # type: ignore[override]
        """Missing torch fallback."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use BaselineMLP")
