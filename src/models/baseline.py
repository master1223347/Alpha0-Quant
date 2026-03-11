"""Baseline sequence classifier (flatten + MLP)."""

from __future__ import annotations


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from src.models.classifier import BinaryClassifierHead
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
        ) -> None:
            super().__init__()
            if window_size <= 0 or num_features <= 0:
                raise ValueError("window_size and num_features must be > 0")

            input_dim = window_size * num_features
            self.encoder = MLPEncoder(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
            self.classifier = BinaryClassifierHead(
                input_dim=self.encoder.output_dim,
                hidden_dim=None,
                dropout=dropout,
            )
            self.window_size = window_size
            self.num_features = num_features

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            flattened = inputs.reshape(inputs.size(0), -1)
            encoded = self.encoder(flattened)
            return self.classifier(encoded)

else:

    class BaselineMLP:  # type: ignore[override]
        """Missing torch fallback."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use BaselineMLP")
