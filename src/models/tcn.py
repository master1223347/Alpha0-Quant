"""Temporal convolutional models for MINN-style multi-head prediction."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None

from src.models.heads_event import EventMetaHead


def _coerce_sequence_input(inputs: Any) -> Any:
    if inputs.dim() == 2:
        return inputs.unsqueeze(1)
    if inputs.dim() == 4:
        return inputs.mean(dim=1)
    if inputs.dim() != 3:
        raise ValueError(f"Expected 2D, 3D, or 4D input tensor, got shape {tuple(inputs.shape)}")
    return inputs


if nn is not None:

    class _TemporalResidualBlock(nn.Module):
        def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            padding = ((kernel_size - 1) // 2) * dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            self.norm1 = nn.BatchNorm1d(channels)
            self.norm2 = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.activation = nn.GELU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            residual = inputs
            outputs = self.conv1(inputs)
            outputs = self.norm1(outputs)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            outputs = self.conv2(outputs)
            outputs = self.norm2(outputs)
            outputs = self.dropout(outputs)
            return self.activation(outputs + residual)


    class tcn_encoder(nn.Module):
        """Temporal convolutional encoder with optional multi-head outputs."""

        def __init__(
            self,
            *,
            window_size: int,
            num_features: int,
            hidden_dims: tuple[int, ...] = (64, 64, 128),
            dropout: float = 0.1,
            kernel_size: int = 3,
            multitask_output: bool = True,
            include_rank_score: bool = False,
            include_regime_logits: bool = False,
            include_event_heads: bool = False,
            regime_classes: int = 3,
            distribution: str = "student_t",
            probabilistic_output: bool = True,
        ) -> None:
            super().__init__()
            if window_size <= 0 or num_features <= 0:
                raise ValueError("window_size and num_features must be > 0")
            if not hidden_dims:
                raise ValueError("hidden_dims must not be empty")

            self.window_size = window_size
            self.num_features = num_features
            self.multitask_output = multitask_output
            self.include_rank_score = include_rank_score
            self.include_regime_logits = include_regime_logits
            self.include_event_heads = bool(include_event_heads)
            self.distribution = distribution
            self.probabilistic_output = bool(probabilistic_output)

            blocks: list[nn.Module] = []
            in_channels = num_features
            for index, channels in enumerate(hidden_dims):
                blocks.append(nn.Conv1d(in_channels, channels, kernel_size=1))
                blocks.append(nn.GELU())
                blocks.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
                for block_index in range(2):
                    blocks.append(
                        _TemporalResidualBlock(
                            channels,
                            kernel_size=kernel_size,
                            dilation=2**block_index,
                            dropout=dropout,
                        )
                    )
                in_channels = channels
            self.backbone = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.projection = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )

            latent_dim = hidden_dims[-1]
            self.direction_head = nn.Linear(latent_dim, 1)
            self.mean_head = nn.Linear(latent_dim, 1)
            self.log_scale_head = nn.Linear(latent_dim, 1)
            self.threshold_head = nn.Linear(latent_dim, 3)
            self.rank_head = nn.Linear(latent_dim, 1) if include_rank_score else None
            self.regime_head = nn.Linear(latent_dim, regime_classes) if include_regime_logits else None
            self.event_head = EventMetaHead(latent_dim) if include_event_heads else None

        def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
            sequence = _coerce_sequence_input(inputs)
            features = sequence.transpose(1, 2)
            encoded = self.backbone(features)
            pooled = self.pool(encoded).squeeze(-1)
            return self.projection(pooled)

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
            if self.event_head is not None:
                outputs.update(self.event_head(latent))
            return outputs

        def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
            latent = self._encode(inputs)
            return self._assemble_outputs(latent)

else:

    class tcn_encoder:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use tcn_encoder")
