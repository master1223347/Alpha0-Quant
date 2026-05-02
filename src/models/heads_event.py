"""Event-meta output heads for two-stage tradeability prediction."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class ConditionalDirectionHead(nn.Module):
        """Predict P(up | event) from a shared latent representation."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(int(input_dim), 1)

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            return self.linear(latent).squeeze(-1)


    class EventMetaHead(nn.Module):
        """Two-head event block: P(event) and P(up | event)."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.event = nn.Linear(int(input_dim), 1)
            self.conditional_direction = ConditionalDirectionHead(int(input_dim))

        def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
            event_logit = self.event(latent).squeeze(-1)
            direction_logit = self.conditional_direction(latent)
            return {
                "event_logit": event_logit,
                "event_direction_logit": direction_logit,
                "direction_logit_cond_event": direction_logit,
            }

else:

    class ConditionalDirectionHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use ConditionalDirectionHead")


    class EventMetaHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use EventMetaHead")
