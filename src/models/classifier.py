"""Classifier heads for binary direction prediction."""

from __future__ import annotations


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class BinaryClassifierHead(nn.Module):
        """Project latent features into a single direction logit."""

        def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be > 0")

            if hidden_dim is None or hidden_dim <= 0:
                self.network = nn.Linear(input_dim, 1)
            else:
                layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden_dim, 1))
                self.network = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs).squeeze(-1)

else:

    class BinaryClassifierHead:  # type: ignore[override]
        """Missing torch fallback."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use BinaryClassifierHead")
