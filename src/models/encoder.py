"""Encoders for sequence classification models."""

from __future__ import annotations


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class MLPEncoder(nn.Module):
        """Simple MLP encoder for flattened sequence inputs."""

        def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128), dropout: float = 0.1) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be > 0")
            if not hidden_dims:
                raise ValueError("hidden_dims must not be empty")

            layers: list[nn.Module] = []
            previous_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(previous_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous_dim = hidden_dim

            self.network = nn.Sequential(*layers)
            self.output_dim = previous_dim

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs)

else:

    class MLPEncoder:  # type: ignore[override]
        """Missing torch fallback."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use MLPEncoder")
