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


    class EventConditionedHeads(nn.Module):
        """Event-conditioned heads with backward-compatible aliases."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.event = nn.Linear(int(input_dim), 1)
            self.conditional_direction = ConditionalDirectionHead(int(input_dim))
            self.opportunity = nn.Linear(int(input_dim), 1)
            self.drift = nn.Linear(int(input_dim), 3)
            self.reversal = nn.Linear(int(input_dim), 1)
            self.absorption = nn.Linear(int(input_dim), 1)

        def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
            event_logit = self.event(latent).squeeze(-1)
            direction_logit = self.conditional_direction(latent)
            return {
                "event_logit": event_logit,
                "event_direction_logit": direction_logit,
                "direction_logit_cond_event": direction_logit,
                "event_opportunity_logit": self.opportunity(latent).squeeze(-1),
                "event_drift_logits": self.drift(latent),
                "event_reversal_logit": self.reversal(latent).squeeze(-1),
                "event_absorption_logit": self.absorption(latent).squeeze(-1),
            }


    EventMetaHead = EventConditionedHeads


    class EventFeatureScalarEmbeddings(nn.Module):
        """Replace categorical event feature scalars with learned scalar embeddings."""

        def __init__(self, event_feature_indices: dict[str, int] | None = None) -> None:
            super().__init__()
            self.event_feature_indices = dict(event_feature_indices or {})
            self.embeddings = nn.ModuleDict(
                {
                    "event_type_id": nn.Embedding(16, 1),
                    "event_scope_id": nn.Embedding(16, 1),
                    "event_scheduled_id": nn.Embedding(4, 1),
                    "time_to_event_bucket": nn.Embedding(16, 1),
                    "time_since_event_bucket": nn.Embedding(16, 1),
                }
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if not self.event_feature_indices:
                return inputs
            output = inputs.clone()
            for name, index in self.event_feature_indices.items():
                if name not in self.embeddings or index < 0 or index >= output.size(-1):
                    continue
                ids = torch.clamp(output[..., index].round().long(), min=0, max=self.embeddings[name].num_embeddings - 1)
                output[..., index] = self.embeddings[name](ids).squeeze(-1)
            return output

else:

    class ConditionalDirectionHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use ConditionalDirectionHead")


    class EventMetaHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use EventMetaHead")


    class EventConditionedHeads:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use EventConditionedHeads")


    class EventFeatureScalarEmbeddings:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use EventFeatureScalarEmbeddings")
