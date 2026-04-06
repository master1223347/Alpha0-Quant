"""Graph-style panel model for MINN-style multi-head prediction."""

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


def _coerce_sequence_input(inputs: Any) -> Any:
    if inputs.dim() == 2:
        return inputs.unsqueeze(1)
    if inputs.dim() == 4:
        return inputs.mean(dim=1)
    if inputs.dim() != 3:
        raise ValueError(f"Expected 2D, 3D, or 4D input tensor, got shape {tuple(inputs.shape)}")
    return inputs


if nn is not None:

    class gnn_panel(nn.Module):
        """Lightweight graph/message-passing model over feature nodes."""

        def __init__(
            self,
            *,
            window_size: int,
            num_features: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            multitask_output: bool = True,
            include_rank_score: bool = False,
            include_regime_logits: bool = False,
            regime_classes: int = 3,
            distribution: str = "student_t",
            probabilistic_output: bool = True,
        ) -> None:
            super().__init__()
            if window_size <= 0 or num_features <= 0:
                raise ValueError("window_size and num_features must be > 0")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be > 0")

            self.window_size = window_size
            self.num_features = num_features
            self.multitask_output = multitask_output
            self.include_rank_score = include_rank_score
            self.include_regime_logits = include_regime_logits
            self.distribution = distribution
            self.probabilistic_output = bool(probabilistic_output)

            self.temporal_projection = nn.Sequential(
                nn.Linear(window_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.feature_embedding = nn.Parameter(torch.randn(num_features, hidden_dim) * 0.02)
            self.adjacency_logits = nn.Parameter(torch.zeros(num_features, num_features))
            self.message_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )

            self.direction_head = nn.Linear(hidden_dim, 1)
            self.mean_head = nn.Linear(hidden_dim, 1)
            self.log_scale_head = nn.Linear(hidden_dim, 1)
            self.threshold_head = nn.Linear(hidden_dim, 3)
            self.rank_head = nn.Linear(hidden_dim, 1) if include_rank_score else None
            self.regime_head = nn.Linear(hidden_dim, regime_classes) if include_regime_logits else None

        def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
            sequence = _coerce_sequence_input(inputs)
            if sequence.size(1) != self.window_size:
                raise ValueError(
                    f"Expected window_size {self.window_size}, received sequence length {sequence.size(1)}"
                )

            node_inputs = sequence.transpose(1, 2)
            node_states = self.temporal_projection(node_inputs)
            node_states = node_states + self.feature_embedding.unsqueeze(0)

            adjacency = torch.softmax(self.adjacency_logits, dim=-1)
            for layer in self.message_layers:
                message = torch.matmul(adjacency, node_states)
                node_states = self.final_norm(layer(node_states + message))

            mean_pool = node_states.mean(dim=1)
            max_pool = node_states.max(dim=1).values
            return self.readout(torch.cat([mean_pool, max_pool], dim=-1))

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

    class gnn_panel:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use gnn_panel")
