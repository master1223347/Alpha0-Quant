"""Transformer-style panel model for MINN-style multi-head prediction."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _coerce_sequence_input(inputs: Any) -> Any:
    if inputs.dim() == 2:
        return inputs.unsqueeze(1)
    if inputs.dim() == 4:
        return inputs.mean(dim=1)
    if inputs.dim() != 3:
        raise ValueError(f"Expected 2D, 3D, or 4D input tensor, got shape {tuple(inputs.shape)}")
    return inputs


if nn is not None:

    class panel_transformer(nn.Module):
        """Transformer encoder with direction, distribution, and auxiliary heads."""

        def __init__(
            self,
            *,
            window_size: int,
            num_features: int,
            d_model: int = 128,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            multitask_output: bool = True,
            include_rank_score: bool = False,
            include_regime_logits: bool = False,
            regime_classes: int = 3,
            distribution: str = "gaussian",
            probabilistic_output: bool = True,
        ) -> None:
            super().__init__()
            if window_size <= 0 or num_features <= 0:
                raise ValueError("window_size and num_features must be > 0")
            if d_model <= 0:
                raise ValueError("d_model must be > 0")

            self.window_size = window_size
            self.num_features = num_features
            self.multitask_output = multitask_output
            self.include_rank_score = include_rank_score
            self.include_regime_logits = include_regime_logits
            self.distribution = distribution
            self.probabilistic_output = bool(probabilistic_output)

            self.input_projection = nn.Linear(num_features, d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.final_norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )

            self.direction_head = nn.Linear(d_model, 1)
            self.mean_head = nn.Linear(d_model, 1)
            self.log_scale_head = nn.Linear(d_model, 1)
            self.threshold_head = nn.Linear(d_model, 3)
            self.rank_head = nn.Linear(d_model, 1) if include_rank_score else None
            self.regime_head = nn.Linear(d_model, regime_classes) if include_regime_logits else None

        def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
            sequence = _coerce_sequence_input(inputs)
            length = sequence.size(1)
            if length > self.position_embedding.size(1):
                raise ValueError(
                    f"Input sequence length {length} exceeds configured window_size {self.position_embedding.size(1)}"
                )
            tokens = self.input_projection(sequence)
            tokens = tokens + self.position_embedding[:, :length, :]
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self.head(self.final_norm(pooled))

        def _assemble_outputs(self, latent: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
            direction_logit = self.direction_head(latent).squeeze(-1)
            if not self.multitask_output:
                return direction_logit

            outputs: dict[str, torch.Tensor] = {
                "direction_logit": direction_logit,
                "threshold_logits": self.threshold_head(latent),
            }
            if self.probabilistic_output:
                outputs["mean_return"] = self.mean_head(latent).squeeze(-1)
                outputs["log_scale"] = self.log_scale_head(latent).squeeze(-1)
            if self.rank_head is not None:
                outputs["rank_score"] = self.rank_head(latent).squeeze(-1)
            if self.regime_head is not None:
                outputs["regime_logits"] = self.regime_head(latent)
            return outputs

        def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
            latent = self._encode(inputs)
            return self._assemble_outputs(latent)

else:

    class panel_transformer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use panel_transformer")
